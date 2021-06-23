import torch
import logging
import os
import sys
from tqdm import tqdm
from dataclasses import field
from omegaconf import II
from dataclasses import dataclass
from torch.nn.functional import normalize

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq.criterions.optimal_transport")

from fairseq import utils
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


def IPOT(C, n, m, beta=0.5, iteration=50):
    sigma = torch.ones([m, 1]).to(C.device) / m
    T = torch.ones([n, m]).to(C.device)
    A = torch.exp(-C / beta)
    for t in range(iteration):
        Q = torch.multiply(A, T)
        for k in range(1):
            delta = 1 / (n * torch.matmul(Q, sigma))
            sigma = 1 / (m * torch.matmul(Q.permute([1, 0]), delta))
        tmp = torch.matmul(torch.diag(delta.squeeze()), Q)
        T = torch.matmul(tmp, torch.diag(sigma.squeeze()))
    return T


def IPOT_distance(C, n, m, beta=0.5, iteration=50):
    T = IPOT(C, n, m, beta=beta, iteration=iteration)
    distance = torch.trace(torch.matmul(C.permute([1, 0]), T))
    return distance


def compute_ot_loss(input_embed, output_embed, nsentences, beta=0.5, iteration=50):
    logger.info("start calculating OT distance")
    cosine_cost = 1 - torch.bmm(input_embed, output_embed.permute([0, 2, 1]))

    ot_loss = torch.tensor(0.).to(cosine_cost.device)
    for sentence in tqdm(range(nsentences), ncols=80):
        n = cosine_cost[sentence].size(0)
        m = cosine_cost[sentence].size(1)
        distance = IPOT_distance(cosine_cost[sentence], n, m, beta=beta, iteration=iteration)
        ot_loss += distance

    # ot_loss /= nsentences
    return ot_loss


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
    ot_weight: float = field(
        default=0.1,
        metadata={'help': "weight of Optimal Transport loss"}
    )
    ot_beta: float = field(
        default=0.5,
        metadata={'help': "beta in Optimal Transport"}
    )
    ot_iteration: int = field(
        default=50,
        metadata={'help': 'number of iteration for calculating OT distance'}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion('label_smoothed_cross_entropy_with_ot', dataclass=OptimalTransportConfig)
class LabelSmoothedCrossEntropyCriterion_OT(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ot_weight=0.1,
            ot_beta=0.5,
            ot_iteration=50,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.weight = ot_weight
        self.beta = ot_beta
        self.iteration = ot_iteration
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

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "label_smoothed_loss": label_smoothed_loss.data,
            "nll_loss": nll_loss.data,
            "ot_loss": ot_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
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

        ot_loss = compute_ot_loss(input_embed_norm, output_embed_norm, sample["nsentences"],
                                  beta=self.beta, iteration=self.iteration)

        loss = label_smoothed_loss + self.weight * ot_loss
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
