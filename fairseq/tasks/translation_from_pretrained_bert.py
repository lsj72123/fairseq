from multiprocessing import Pool
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from transformers import BertTokenizer

from fairseq import tokenizer
from fairseq.data import Dictionary
from fairseq.file_io import PathManager
from . import register_task
from .translation import TranslationTask, TranslationConfig


@dataclass
class TranslationFromPretrainedBERTConfig(TranslationConfig):
    bert_model: Optional[str] = field(
        default=None,
        metadata={"help": "BERT model name"}
    )
    bert_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained BERT model checkpoint directory"}
    )


@register_task("translation_from_pretrained_bert", dataclass=TranslationFromPretrainedBERTConfig)
class TranslationFromPretrainedBERTTask(TranslationTask):
    cfg: TranslationFromPretrainedBERTConfig

    @classmethod
    def build_dictionary(cls, filenames, tokenizer, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        assert isinstance(tokenizer, BertTokenizer)
        d = Dictionary(
            bos=tokenizer.cls_token,
            pad=tokenizer.pad_token,
            eos=tokenizer.sep_token,
            unk=tokenizer.unk_token,
        )
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

