# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

import torch
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
from typing import List, Dict


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:

    @staticmethod
    def find_offsets(filename, num_chunks) -> List[int]:
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]

            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()

            return offsets

    @staticmethod
    def binarize(
            filename, dict, consumer, tokenize=tokenize_line,
            append_eos=True, reverse_order=False, offset=0, end=-1, already_numberized=False,
    ) -> Dict[str, int]:
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):  # contains all the OOV tokens that need to be replaced by NUK
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                # f.tell() does not always give the byte position in the file
                # sometimes it skips to a very large number
                # it is unlikely that through a normal read we go from
                # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
                # that the procedure breaks by the Nondeterministic behavior of
                # f.tell()
                if 0 < end < f.tell() < end + 2 ** 32:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(
            filename, alignment_parser, consumer, offset=0, end=-1
    ) -> Dict[str, int]:
        nseq = 0

        with open(PathManager.get_local_path(filename), "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if 0 < end < f.tell():
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def binarize_bert(
            filename, tokenizer, consumer, offset=0, end=-1,
            append_eos=True, reverse_order=False, already_numberized=False,
    ) -> Dict[str, int]:
        nseq, ntok = 0, 0
        replaced, used = Counter(), Counter()

        def replaced_consumer(word, idx):  # contains all the OOV tokens that need to be replaced by NUK
            if idx == tokenizer.unk_index and word != tokenizer.unk_word:
                replaced.update([word])

        def used_consumer(word):
            used.update([word])

        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                # f.tell() does not always give the byte position in the file
                # sometimes it skips to a very large number
                # it is unlikely that through a normal read we go from
                # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
                # that the procedure breaks by the Nondeterministic behavior of
                # f.tell()
                if 0 < end < f.tell() < end + 2 ** 32:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(tokenizer.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    line = '{} {} {}'.format('[CLS]', line.strip(), '[SEP]')
                    tokenized_line = tokenizer.tokenize(line)
                    if len(tokenized_line) > tokenizer.max_len:
                        tokenized_line = tokenized_line[:tokenizer.max_len - 1]
                        tokenized_line.append("[SEP]")  # make sure the last token is [SEP]
                    words = tokenizer.convert_tokens_to_ids(tokenized_line)
                    nwords = len(words)
                    ids = torch.IntTensor(nwords)
                    for i, word in enumerate(words):
                        ids[i] = word
                        replaced_consumer(tokenized_line[i], word)
                        used_consumer((tokenized_line[i], word))
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
            "used": used,
        }
