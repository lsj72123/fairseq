import logging
from os.path import join, exists
from dataclasses import dataclass, field
from distutils.version import LooseVersion
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from transformers import __version__ as trans_version

from fairseq import utils
from fairseq.data import Dictionary, data_utils
from . import register_task
from .translation import TranslationTask, TranslationConfig

BERT_CACHE_DIR = "D:/pretrained_model/huggingface/transformers"
logger = logging.getLogger(__name__)


@dataclass
class TranslationFromPretrainedBERTConfig(TranslationConfig):
    bert_model: Optional[str] = field(
        default=None,
        metadata={"help": "BERT model name"}
    )
    bert_model_dir: str = field(
        default="bert_models",
        metadata={"help": "path to store the reduced BERT model"}
    )


@register_task('translation_from_pretrained_bert', dataclass=TranslationFromPretrainedBERTConfig)
class TranslationFromPretrainedBERTTask(TranslationTask):
    cfg: TranslationFromPretrainedBERTConfig

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):  # load dict and add dictionary to class
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg:
        """
        if (
                exists(join(cfg.bert_model_dir, "vocab.txt")) and
                exists(join(cfg.bert_model_dir, "config.json")) and
                exists(join(cfg.bert_model_dir, "pytorch_model.bin"))
        ):
            bert_tokenizer = AutoTokenizer.from_pretrained(
                cfg.bert_model_dir, cache_dir=BERT_CACHE_DIR,
                use_fast=LooseVersion(trans_version) < LooseVersion("4.0.0"))
            bert_model = AutoModel.from_pretrained(
                cfg.bert_model_dir, cache_dir=BERT_CACHE_DIR)
        else:
            bert_tokenizer = AutoTokenizer.from_pretrained(
                cfg.bert_model, cache_dir=BERT_CACHE_DIR,
                use_fast=LooseVersion(trans_version) < LooseVersion("4.0.0"))
            bert_model = AutoModel.from_pretrained(
                cfg.bert_model, cache_dir=BERT_CACHE_DIR)

        paths = utils.split_paths(cfg.data)  # data can contains all the data directory
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            join(paths[0], "dict.{}.txt".format(cfg.source_lang)), bert_tokenizer
        )
        tgt_dict = cls.load_dictionary(
            join(paths[0], "dict.{}.txt".format(cfg.target_lang)), bert_tokenizer
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict, bert_tokenizer, bert_model)

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict, bert_tokenizer, bert_model):
        super().__init__(cfg, src_dict, tgt_dict)
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model

    @classmethod
    def load_dictionary(cls, filename, bert_tokenizer):
        all_special_tokens_extended = set(bert_tokenizer.all_special_tokens_extended)
        used_special_tokens = {bert_tokenizer.cls_token,
                               bert_tokenizer.pad_token,
                               bert_tokenizer.sep_token,
                               bert_tokenizer.unk_token}

        d = Dictionary(
            bos=bert_tokenizer.cls_token,
            pad=bert_tokenizer.pad_token,
            eos=bert_tokenizer.sep_token,
            unk=bert_tokenizer.unk_token,
            extra_special_symbols=all_special_tokens_extended - used_special_tokens
        )
        d.add_from_file(filename)
        return d

    @classmethod
    def build_dictionary(cls, filenames, bert_tokenizer, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        all_special_tokens_extended = set(bert_tokenizer.all_special_tokens_extended)
        used_special_tokens = {bert_tokenizer.cls_token,
                               bert_tokenizer.pad_token,
                               bert_tokenizer.sep_token,
                               bert_tokenizer.unk_token}

        d = Dictionary(
            bos=bert_tokenizer.cls_token,
            pad=bert_tokenizer.pad_token,
            eos=bert_tokenizer.sep_token,
            unk=bert_tokenizer.unk_token,
            extra_special_symbols=all_special_tokens_extended - used_special_tokens
        )
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, bert_tokenizer.tokenize, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d
