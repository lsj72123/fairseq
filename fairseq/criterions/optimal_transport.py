import torch
import math
from dataclasses import field
from omegaconf import II
from dataclasses import dataclass
from torch.nn.functional import normalize, one_hot

from fairseq import utils, metrics
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


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
    cosine_cost_copy = 1 - torch.bmm(input_embed, output_embed.permute([0, 2, 1]))
    cosine_cost_match = 1 - torch.bmm(target_embed, output_embed.permute([0, 2, 1]))

    total_loss = torch.tensor(0., device=input_embed.device)
    if copy_weight > 0:
        copy_distance = IPOT_distance(cosine_cost_copy, beta=beta, iteration=iteration, step=step)
        total_loss += copy_distance * copy_weight

    if match_weight > 0:
        match_distance = IPOT_distance(cosine_cost_match, beta=beta, iteration=iteration, step=step)
        total_loss += match_weight * match_distance

    return total_loss


@dataclass
class OptimalTransportConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ot_weight_copy: float = field(
        default=0.0,
        metadata={'help': "weight of Optimal Transport loss in copy term"}
    )
    ot_weight_match: float = field(
        default=0.1,
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
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion('label_smoothed_cross_entropy_with_ot', dataclass=OptimalTransportConfig)
class LabelSmoothedCrossEntropyCriterion_OT(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ot_weight_copy=0.0,
            ot_weight_match=0.1,
            ot_beta=0.5,
            ot_iteration=50,
            ot_k_step=1,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.weight_copy = ot_weight_copy
        self.weight_match = ot_weight_match
        self.beta = ot_beta
        self.iteration = ot_iteration
        self.k_step = ot_k_step
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

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

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, probs, target = self.get_lprobs_and_target(model, net_output, sample)
        label_smoothed_loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        assert hasattr(model, 'decoder')
        if getattr(model.decoder, 'fc_out', None):
            decoder_embedding = model.decoder.fc_out.weight
        else:
            decoder_embedding = model.decoder.embed_tokens.weight

        output_embed = torch.matmul(probs, decoder_embedding)
        output_embed_norm = normalize(output_embed, p=2, dim=-1, eps=1e-12)  # take the average embedding
        output_embed_norm = output_embed_norm.view([sample["nsentences"], -1, output_embed_norm.size(-1)]).contiguous()

        target_sentence = one_hot(sample["target"], num_classes=decoder_embedding.size(0)).float()
        target_embed = torch.einsum("aij,jk->aik", target_sentence, decoder_embedding)
        target_embed_norm = normalize(target_embed, p=2, dim=-1, eps=1e-12)

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
        input_embed_norm = normalize(input_embed, p=2, dim=-1, eps=1e-12)

        ot_loss = compute_ot_loss(input_embed_norm, target_embed_norm, output_embed_norm,
                                  iteration=self.iteration, copy_weight=self.weight_copy,
                                  match_weight=self.weight_match, beta=self.beta, step=self.k_step)

        loss = label_smoothed_loss + ot_loss
        return loss, label_smoothed_loss, nll_loss, ot_loss

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        probs = model.get_normalized_probs(net_output, log_probs=False)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                probs = probs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                probs = probs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), probs.view(-1, probs.size(-1)), target.view(-1)

    def compute_accuracy(self, model, net_output, sample):
        lprobs, _, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

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


@register_criterion('optimal_transport', dataclass=OptimalTransportConfig)
class OptimalTransportCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        nll_loss = self.compute_nll_loss(model, net_output, sample, reduce=reduce)
        ot_loss = self.compute_ot_loss()

    def compute_ot_loss(self, model, net_output, sample, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)

    def compute_nll_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss
