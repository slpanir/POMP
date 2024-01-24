#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple
from contextlib import redirect_stdout
from typing import List, Tuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq.dataclass.configs import FairseqConfig
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
import sentencepiece as spm
import tempfile
from fairseq_cli.preprocess import cli_main as preprocess_cli_main
from fairseq_cli.generate import cli_main as generate_cli_main
from io import StringIO
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


# Function to load models
def load_models(cfg: FairseqConfig) -> Tuple[List, TranslationMultiSimpleEpochTask]:
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    task = tasks.setup_task(cfg)
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, _ = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda:
            model.cuda()
        model.prepare_for_inference_(cfg)
    return models, task


def encode_sentence_with_spm(sentence, spm_model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)
    return sp.EncodeAsPieces(sentence, out_type=str)

def decode_sentence_with_spm(sentence, spm_model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)
    return "".join(sp.DecodePieces(sentence))

def preprocess_sentence(sp, sentence, src_lang, tgt_lang, destdir, dict_path, workers=1):
    encoded_sentence = sp.EncodeAsPieces(sentence)
    # encoded_sentence = encode_sentence_with_spm(sentence, spm_model_path)
    encoded_sentence_str = ' '.join(encoded_sentence)

    # Write the encoded sentence to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_src:
        tmp_src.write(encoded_sentence_str)
        tmp_src.flush()
        src_file_path = tmp_src.name
    os.rename(src_file_path, src_file_path + f'.{src_lang}')
    os.system(f'cp {src_file_path}.{src_lang} {src_file_path}.{tgt_lang}')  #useless

    # Construct the preprocess command arguments
    preprocess_args = [
        '--source-lang', src_lang,
        '--target-lang', tgt_lang,
        '--testpref', src_file_path,
        '--destdir', destdir,
        '--workers', str(workers),
        '--srcdict', dict_path,
        '--tgtdict', dict_path,
        '--dataset-impl', 'lazy',
    ]

    # Run the preprocess command
    sys.argv = ['fairseq-preprocess'] + preprocess_args
    preprocess_cli_main()

    # Clean up the temporary source file
    os.remove(src_file_path + f'.{src_lang}')
    os.remove(src_file_path + f'.{tgt_lang}')

def generate_translation(
        destdir, model_path, dict_path, src_lang, tgt_lang, extra_bt=None, beam=5
):
    # Construct the generate command arguments
    generate_args = [
        destdir,
        '--path', model_path,
        '--task', 'translation_multi_simple_epoch',
        '--source-lang', src_lang,
        '--target-lang', tgt_lang,
        '--gen-subset', 'test',
        '--beam', str(beam),
        '--remove-bpe',
        '--fp16',
        '--same-lang-per-batch',
        '--enable-lang-ids',
        '--max-tokens', '5000',
        '--remove-bpe',
        '--mplm-type', 'xlmrL',
        '--xlmr-task', 'xlmr_2stage_posdrop',
        '--model-overrides', "{'xlmr_modeldir':\"/mnt/e/unmt/acl22-sixtp/models/xlmrL_base\", 'enable_lang_proj':\"True\"}",
        '--langs', "en,de,es,fi,hi,ru,zh",
        '--enable-lang-proj',
        '--fixed-dictionary', dict_path
    ]

    # Capture the output of the generate command
    sys.argv = ['fairseq-generate'] + generate_args
    if extra_bt.get('return_models'):
        scorer, models, saved_cfg = generate_cli_main(extra_bt)
        output = scorer
    else:
        output = generate_cli_main()
    # with StringIO() as buf, redirect_stdout(buf):
    #     if extra_bt.get('return_models'):
    #         scorer, models, saved_cfg = generate_cli_main(extra_bt)
    #         output = scorer
    #     else:
    #         generate_cli_main()
    #         output = buf.getvalue()
    # get H content
    output = "".join(output.split('\n')[2].split('\t')[2:]).replace(" ","").replace("▁","", 1).replace("▁"," ")
    # output=sp.DecodePieces(output)

    if extra_bt.get('return_models'):
        return output, models, saved_cfg
    return output

def bt(
        spm_model_path, sentence, src_lang, tgt_lang, model_path, dict_path, extra_bt=None
):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocess_sentence(sp, sentence, src_lang, tgt_lang, tmpdir, dict_path)
        if extra_bt.get('return_models'):
            translation, models, saved_cfg = generate_translation(
                tmpdir, model_path, dict_path, src_lang, tgt_lang, extra_bt
            )
            return translation, models, saved_cfg
        else:
            translation = generate_translation(tmpdir, model_path, dict_path, src_lang, tgt_lang, extra_bt)
            return translation

# Example usage:
source_list = [
    "Ғажайып құлаққап.",
    "Жүлделі бірінші орынды Қызылорда қала­сынан Дастан Айтбайдың 'Инновациялық құлақ­қап 'Safe headphones' жобасы жеңіп алды.",
    "Оған арнайы диплом мен 300 мың теңгенің сертификаты табыс етілді.",
    "Дастанның құлаққабын нағыз инновациялық жоба деп айтуға болады."
]
first_source = source_list.pop(0)
extra_bt = {
    "return_models": True,
    "models": None,
    "saved_cfg": None,
}
translated_sentence, models, saved_cfg = bt(
    "/mnt/e/unmt/acl22-sixtp/models/x2x/sentencepiece.bpe.model",
    first_source,
    src_lang="kk",
    tgt_lang="en",
    model_path="/mnt/e/unmt/acl22-sixtp/models/x2x/x2x.pt",
    dict_path="/mnt/e/unmt/acl22-sixtp/models/x2x/dict.txt",
    extra_bt=extra_bt,
)
extra_bt["return_models"] = False
extra_bt["models"] = models
extra_bt["saved_cfg"] = saved_cfg
for source in source_list:
    translated_sentence = bt(
       "/mnt/e/unmt/acl22-sixtp/models/x2x/sentencepiece.bpe.model",
        source,
        src_lang="kk",
        tgt_lang="en",
        model_path="/mnt/e/unmt/acl22-sixtp/models/x2x/x2x.pt",
        dict_path="/mnt/e/unmt/acl22-sixtp/models/x2x/dict.txt",
        extra_bt=extra_bt,
    )
    print(translated_sentence)