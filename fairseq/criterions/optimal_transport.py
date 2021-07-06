import torch
import math
import logging
import warnings
import json
import torch.nn.functional as F

from torch import nn
from os.path import join
from dataclasses import dataclass, field

from fairseq import utils, metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
)

logger = logging.getLogger(__name__)


def IPOT_distance(C, beta=1.0, iteration=10, step=1):
    b, n, m = C.size()
    sigma = torch.ones([b, m, 1], device=C.device) / float(m)
    T = torch.ones([b, n, m], device=C.device)
    A = torch.exp(-C / beta)

    for t in range(iteration):
        Q = A * T  # [b, n, m]
        for k in range(step):
            delta = 1 / (float(n) * torch.bmm(Q, sigma))  # [b, n, 1]
            sigma = 1 / (float(m) * torch.bmm(Q.transpose(1, 2), delta))  # [b, m, 1]
        T = delta * Q * sigma.transpose(1, 2)  # [b, n, m]

    distance = torch.bmm(C.transpose(1, 2), T)
    return sum([torch.trace(matrix) for matrix in distance])


def compute_ot_loss(input_embed, target_embed, output_embed, iteration=10,
                    copy_weight=0.0, match_weight=0.1, beta=1.0, step=1):
    if input_embed is not None:
        cosine_cost_copy = 1 - torch.bmm(input_embed, output_embed.permute([0, 2, 1]))
    cosine_cost_match = 1 - torch.bmm(target_embed, output_embed.permute([0, 2, 1]))

    total_loss = torch.tensor(0., device=target_embed.device)
    if input_embed is not None and copy_weight > 0:
        copy_distance = IPOT_distance(cosine_cost_copy, beta=beta, iteration=iteration, step=step)
        total_loss += copy_distance * copy_weight

    if match_weight > 0:
        match_distance = IPOT_distance(cosine_cost_match, beta=beta, iteration=iteration, step=step)
        total_loss += match_weight * match_distance

    return total_loss


def reduce_model(model, old_vocab, new_vocab, path_to_save):
    logger.info(" - parameters before reducing : {}".format(model.num_parameters()))
    logger.info(" - tokens before reducing : {}".format(len(old_vocab)))

    old_embeddings = model.get_input_embeddings()  # only need to change the word_embedding
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    assert old_num_tokens == len(old_vocab), 'len(old_vocab) != len(model.old_embeddings)'

    if len(new_vocab) == 0:
        warnings.warn("No word in target dictionary! BERT model will not be reduced!")
        return old_embeddings

    new_embeddings = nn.Embedding(len(new_vocab), old_embedding_dim).to(old_embeddings.weight.device)
    vocab = []
    for i, token in enumerate(new_vocab):
        if i >= len(new_vocab):
            break
        idx = old_vocab.get(token, None)

        if idx is None:
            if token == new_vocab.bos_word:
                idx = old_vocab.get("[CLS]", None)
                vocab.append("[CLS]")
            elif token == new_vocab.eos_word:
                idx = old_vocab.get("[SEP]", None)
                vocab.append("[SEP]")
            elif token == new_vocab.pad_word:
                idx = old_vocab.get("[PAD]", None)
                vocab.append("[PAD]")
            elif token == new_vocab.unk_word:
                idx = old_vocab.get("[UNK]", None)
                vocab.append("[UNK]")
            elif "madeupword" in token:
                idx = old_vocab.get("[UNK]", None)
                vocab.append(token)
            else:
                raise ValueError("Unknown token in target dictionary")
        else:
            vocab.append(token)
        assert idx is not None, "Unknown index in BERT dictionary"
        new_embeddings.weight.data[i, :] = old_embeddings.weight.data[idx, :]

    assert len(vocab) == len(new_vocab)
    model.set_input_embeddings(new_embeddings)
    model.config.vocab_size = len(new_vocab)
    model.vocab_size = len(new_vocab)
    model.tie_weights()
    model.save_pretrained(path_to_save)

    # save vocab
    with open(join(path_to_save, "vocab.txt"), "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + '\n')

    with open(join(path_to_save, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"do_lower_case": False, "model_max_length": 512}, f)
    return model


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig_OT(LabelSmoothedCrossEntropyCriterionConfig):
    ot_copy_weight: float = field(
        default=0.0,
        metadata={'help': "weight of Optimal Transport loss in copy term"}
    )
    ot_match_weight: float = field(
        default=1,
        metadata={'help': "weight of Optimal Transport loss in match term"}
    )
    ot_beta: float = field(
        default=1.0,
        metadata={'help': "beta in Optimal Transport"}
    )
    ot_iteration: int = field(
        default=10,
        metadata={'help': 'number of iteration for calculating OT distance'}
    )
    ot_k_step: int = field(
        default=1,
        metadata={'help': 'number of step for calculating OT distance in every iteration'}
    )


@register_criterion(
    'label_smoothed_cross_entropy_ot',
    dataclass=LabelSmoothedCrossEntropyCriterionConfig_OT
)
class LabelSmoothedCrossEntropyCriterion_OT(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ot_copy_weight=0.0,
            ot_match_weight=0.1,
            ot_beta=0.5,
            ot_iteration=50,
            ot_k_step=1,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.copy_weight = ot_copy_weight
        self.match_weight = ot_match_weight
        self.beta = ot_beta
        self.iteration = ot_iteration
        self.k_step = ot_k_step

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, label_smoothed_loss, nll_loss, ot_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (sample["nsentences"] if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "loss": loss.data,
            "label_smoothed_loss": label_smoothed_loss.data,
            "nll_loss": nll_loss.data,
            "ot_loss": ot_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_ot_loss(self, model, net_output, sample):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        if self.ignore_prefix_size > 0:
            if getattr(probs, "batch_first", False):
                probs = probs[:, self.ignore_prefix_size:, :].contiguous()
            else:
                probs = probs[self.ignore_prefix_size:, :, :].contiguous()

        assert hasattr(model, 'decoder')
        if getattr(model.decoder, 'fc_out', None):
            decoder_embedding = model.decoder.fc_out.weight
        else:
            decoder_embedding = model.decoder.embed_tokens.weight

        output_embed = torch.einsum("aij,jk->aik", probs, decoder_embedding)
        output_embed_norm = F.normalize(output_embed, p=2, dim=-1, eps=1e-12)  # take the average embedding

        target_sentence = F.one_hot(sample["target"], num_classes=probs.size(-1)).float()
        target_embed = torch.einsum("aij,jk->aik", target_sentence, decoder_embedding)
        target_embed_norm = F.normalize(target_embed, p=2, dim=-1, eps=1e-12)

        assert hasattr(model, 'encoder')
        src_tokens = sample["net_input"]["src_tokens"]
        if getattr(model.encoder, "left_pad", True):
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )
        encoder_embedding = model.encoder.embed_tokens
        input_embed = encoder_embedding(src_tokens)
        input_embed_norm = F.normalize(input_embed, p=2, dim=-1, eps=1e-12)

        ot_loss = compute_ot_loss(input_embed_norm, target_embed_norm, output_embed_norm,
                                  iteration=self.iteration, copy_weight=self.copy_weight,
                                  match_weight=self.match_weight, beta=self.beta, step=self.k_step)

        return ot_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        label_smoothed_loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)
        ot_loss = self.compute_ot_loss(model, net_output, sample)
        loss = label_smoothed_loss + ot_loss

        return loss, label_smoothed_loss, nll_loss, ot_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        label_smoothed_loss_sum = sum(log.get("label_smoothed_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ot_loss_sum = sum(log.get("ot_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "label_smoothed_loss", label_smoothed_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ot_loss", ot_loss_sum / nsentences / math.log(2), nsentences, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )


@register_criterion(
    "label_smoothed_cross_entropy_bertot",
    dataclass=LabelSmoothedCrossEntropyCriterionConfig_OT
)
class LabelSmoothedCrossEntropyCriterion_BERTOT(LabelSmoothedCrossEntropyCriterion_OT):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ot_copy_weight=0.0,
            ot_match_weight=0.1,
            ot_beta=0.5,
            ot_iteration=50,
            ot_k_step=1,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ot_copy_weight, ot_match_weight,
                         ot_beta, ot_iteration, ot_k_step, ignore_prefix_size, report_accuracy)

        # start to reduce the BERT scale.
        self.bert_tokenizer = task.bert_tokenizer
        bert_model = task.bert_model

        logger.info("start to reduce the {} model.".format(task.cfg.bert_model))

        assert hasattr(task, "tgt_dict"), "task must contain target dictionary"
        new_vocab = task.tgt_dict
        old_vocab = self.bert_tokenizer.vocab
        if len(new_vocab) == len(old_vocab):
            logger.info(" - Model is already reduced")
            self.reduced_model = bert_model
        else:
            self.reduced_model = reduce_model(bert_model, old_vocab, new_vocab, task.cfg.bert_model_dir)

        logger.info(" - Reduced Model Parameters: {}".format(self.reduced_model.num_parameters()))
        logger.info(" - Reduced Tokens number: {}".format(
            self.reduced_model.get_input_embeddings().weight.size(0)))

        for p in self.reduced_model.parameters():
            p.requires_grad = False

        batch_bos = torch.zeros([1, 1, len(new_vocab)])
        batch_bos[:, :, new_vocab.bos_index] = 1
        self.batch_bos = batch_bos

    def compute_ot_loss(self, model, net_output, sample):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        if self.ignore_prefix_size > 0:
            if getattr(probs, "batch_first", False):
                probs = probs[:, self.ignore_prefix_size:, :].contiguous()
            else:
                probs = probs[self.ignore_prefix_size:, :, :].contiguous()

        self.reduced_model.to(probs.device)
        self.reduced_model.eval()

        bert_word_embedding = self.reduced_model.get_input_embeddings().weight

        batch_bos = self.batch_bos.repeat([sample["nsentences"], 1, 1]).to(probs.device)

        target_sentence = F.one_hot(sample["target"], num_classes=probs.size(-1)).float()
        target_embed = torch.einsum("aij,jk->aik",
                                    torch.cat([batch_bos, target_sentence], dim=1),
                                    bert_word_embedding)
        # target_contextual_embedding = F.normalize(target_embed, p=2, dim=-1, eps=1e-12)
        target_contextual_embedding = self.reduced_model(inputs_embeds=target_embed)[0]
        target_contextual_embedding = F.normalize(target_contextual_embedding, p=2, dim=-1, eps=1e-12)

        output_embed = torch.einsum("aij,jk->aik",
                                    torch.cat([batch_bos, probs], dim=1),
                                    bert_word_embedding)
        # output_contextual_embedding = F.normalize(output_embed, p=2, dim=-1, eps=1e-12)
        output_contextual_embedding = self.reduced_model(inputs_embeds=output_embed)[0]
        output_contextual_embedding = F.normalize(output_contextual_embedding, p=2, dim=-1, eps=1e-12)

        # todo, make the input compatible with BERT embedding
        assert hasattr(model, 'encoder')
        src_tokens = sample["net_input"]["src_tokens"]
        if getattr(model.encoder, "left_pad", True):
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )
        # encoder_embedding = model.encoder.embed_tokens
        # input_embed = encoder_embedding(src_tokens)
        # input_embed_norm = F.normalize(input_embed, p=2, dim=-1, eps=1e-12)
        input_embed_norm = None

        ot_loss = compute_ot_loss(input_embed_norm,
                                  target_contextual_embedding,
                                  output_contextual_embedding,
                                  iteration=self.iteration, copy_weight=self.copy_weight,
                                  match_weight=self.match_weight, beta=self.beta, step=self.k_step)
        return ot_loss
