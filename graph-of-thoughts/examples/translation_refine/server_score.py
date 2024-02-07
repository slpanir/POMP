import logging
from typing import Dict, List
import sacrebleu
import math
import random
import argparse
import json
import pandas as pd
import numpy as np
from comet import download_model, load_from_checkpoint
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import os

def compute_bleurt_for_batch(ref_batch, input_batch, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(ref_batch, input_batch, padding=True, return_tensors='pt', max_length=512,
                           truncation=True).to('cuda')
        logits = model(**inputs).logits.flatten().tolist()
    torch.cuda.empty_cache()
    return logits

def bleurt_20(ref=None, input=None, hyp=None, bleurt_model=None, bleurt_tokenizer=None):
    batch_size = 32
    bleurt_en_res = []
    bleurt_hyp_res = []

    for i in range(0, len(hyp), batch_size):
        end_idx = min(i + batch_size, len(hyp))
        bleurt_en_res.extend(compute_bleurt_for_batch(ref[i:end_idx], input[i:end_idx], bleurt_model, bleurt_tokenizer))
        bleurt_hyp_res.extend(compute_bleurt_for_batch(ref[i:end_idx], hyp[i:end_idx], bleurt_model, bleurt_tokenizer))

        # calculate the average of the en_res and hyp_res

    # calculate the average of the en_res and hyp_res
    bleurt_input_avg = sum(bleurt_en_res) / len(bleurt_en_res)
    bleurt_hyp_avg = sum(bleurt_hyp_res) / len(bleurt_hyp_res)
    print("bleurt-20_en_avg: " + str(bleurt_input_avg))
    print("bleurt-20_hyp_avg: " + str(bleurt_hyp_avg))
    return bleurt_input_avg, bleurt_hyp_avg

def bleurt_d12(ref=None, input=None, hyp=None, bleurt_model=None, bleurt_tokenizer=None):
    batch_size = 32
    bleurt_en_res = []
    bleurt_hyp_res = []

    for i in range(0, len(hyp), batch_size):
        end_idx = min(i + batch_size, len(hyp))
        bleurt_en_res.extend(compute_bleurt_for_batch(ref[i:end_idx], input[i:end_idx], bleurt_model, bleurt_tokenizer))
        bleurt_hyp_res.extend(compute_bleurt_for_batch(ref[i:end_idx], hyp[i:end_idx], bleurt_model, bleurt_tokenizer))

        # calculate the average of the en_res and hyp_res

    # calculate the average of the en_res and hyp_res
    bleurt_input_avg = sum(bleurt_en_res) / len(bleurt_en_res)
    bleurt_hyp_avg = sum(bleurt_hyp_res) / len(bleurt_hyp_res)
    print("bleurt-d12_en_avg: " + str(bleurt_input_avg))
    print("bleurt-d12_hyp_avg: " + str(bleurt_hyp_avg))
    return bleurt_input_avg, bleurt_hyp_avg

def comet_0(ref=None, input=None, hyp=None, comet_model=None):
    comet_hyp = []
    comet_input = []
    for i in range(len(ref)):
        comet_hyp.append({'src': ref[i], 'mt': hyp[i], 'ref': ref[i]})
        comet_input.append({'src': ref[i], 'mt': input[i], 'ref': ref[i]})

    model_hyp = comet_model.predict(comet_hyp, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
    model_input = comet_model.predict(comet_input, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
    print("comet-0_hyp_avg: " + str(model_hyp))
    print("comet-0_input_avg: " + str(model_input))

    return model_hyp, model_input

def comet_x(ref=None, input=None, hyp=None, comet_model=None):
    comet_hyp = []
    comet_input = []
    for i in range(len(ref)):
        comet_hyp.append({'src': ref[i], 'mt': hyp[i], 'ref': ref[i]})
        comet_input.append({'src': ref[i], 'mt': input[i], 'ref': ref[i]})

    model_hyp = comet_model.predict(comet_hyp, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
    model_input = comet_model.predict(comet_input, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]

    print("comet-x_hyp_avg: " + str(model_hyp))
    print("comet-x_input_avg: " + str(model_input))

    return model_hyp, model_input

def evaluate_got_refine_results(
        hyp, data, result_path,
        is_bleurt_20=False, is_bleurt_d12=False,
        is_comet_0=False, is_comet_x=False,
        is_bleu=False,
        gt4pseudo=None, pseudo=False):
    src = [x[1] for x in data]
    input = [x[2] for x in data]
    if pseudo:
        ref = [x[0] for x in gt4pseudo]
    else:
        ref = [x[3] for x in data]

    logging.info(f"len of src: {len(src)}, len of input: {len(input)}, len of ref: {len(ref)}, len of hyp: {len(hyp)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_bleurt_20:
        print("Caclulating bleurt-20")
        bleurt_20_model = BleurtForSequenceClassification.from_pretrained('bleurt/BLEURT-20').to(device).eval()
        bleurt_20_tokenizer = BleurtTokenizer.from_pretrained('bleurt/BLEURT-20')
        bleurt_20_hyp_avg, bleurt_20_input_avg = bleurt_20(ref, input, hyp, bleurt_20_model, bleurt_20_tokenizer)
        with open(os.path.join(result_path, 'bleurt_20.json'), 'w', encoding='utf8') as f:
            json.dump({'bleurt_20_hyp_avg': bleurt_20_hyp_avg, 'bleurt_20_input_avg': bleurt_20_input_avg}, f)
    if is_bleurt_d12:
        print("Caclulating bleurt-20-d12")
        bleurt_d12_model = BleurtForSequenceClassification.from_pretrained('bleurt/BLEURT-20-D12').to(device).eval()
        bleurt_d12_tokenizer = BleurtTokenizer.from_pretrained('bleurt/BLEURT-20-D12')
        bleurt_d12_hyp_avg, bleurt_d12_input_avg = bleurt_d12(ref, input, hyp, bleurt_d12_model, bleurt_d12_tokenizer)
        with open(os.path.join(result_path, 'bleurt_d12.json'), 'w', encoding='utf8') as f:
            json.dump({'bleurt_d12_hyp_avg': bleurt_d12_hyp_avg, 'bleurt_d12_input_avg': bleurt_d12_input_avg}, f)
    if is_bleu:
        print("Caclulating bleu")
        bleu_hyp = sacrebleu.corpus_bleu(hyp, [ref]).score
        bleu_input = sacrebleu.corpus_bleu(input, [ref]).score
        with open(os.path.join(result_path, 'bleu.json'), 'w', encoding='utf8') as f:
            json.dump({'bleu_hyp': bleu_hyp, 'bleu_input': bleu_input}, f)

    if is_comet_0:
        print("Caclulating comet")
        comet_model_path = '/data/xcomet/models/checkpoints/model.ckpt'
        comet_0_model = load_from_checkpoint(comet_model_path, reload_hparams=True).eval()
        comet_0_hyp, comet_0_input = comet_0(ref, input, hyp, comet_0_model)
        with open(os.path.join(result_path, 'comet_0.json'), 'w', encoding='utf8') as f:
            json.dump({'comet_0_hyp': comet_0_hyp, 'comet_0_input': comet_0_input}, f)
    if is_comet_x:
        print("Caclulating xcomet")
        comet_model_path = '/data/xcomet/models/checkpoints/model.ckpt'
        comet_x_model = load_from_checkpoint(comet_model_path, reload_hparams=True).eval()
        comet_x_hyp, comet_x_input = comet_x(ref, input, hyp, comet_x_model)
        with open(os.path.join(result_path, 'comet_x.json'), 'w', encoding='utf8') as f:
            json.dump({'comet_x_hyp': comet_x_hyp, 'comet_x_input': comet_x_input}, f)


def evaluate_from_file(
        src_ref_path: str,
        data_path: str,
        is_bleurt_20=False, is_bleurt_d12=False,
        is_comet_0=False, is_comet_x=False,
        is_bleu=False):
    # bleurt_cache_dir = r'/data/xcomet/bleurt'
    # bleurt_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    # bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20',
    #                                                                cache_dir=bleurt_cache_dir).to(device)
    # bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)

    langs = ["et", "gu", "kk", "lv", "ne", "si"]

    for lang in langs:
        src_path = os.path.join(src_ref_path, f"{lang}2en", "src")
        trans_path = os.path.join(src_ref_path, f"{lang}2en", "hyp")
        ref_path = os.path.join(src_ref_path, f"{lang}2en", "ref")

        metric_path = os.path.join(data_path, f"{lang}2en")
        got_refine_hyp = os.path.join(data_path, f"{lang}2en", 'test_got_refine')
        # got_refine_eval = os.path.join(data_path, 'test_metrics.json')
        with open(got_refine_hyp, 'r', encoding='utf8') as f:
            refine_results = f.readlines()
        refine_results = [i.strip() for i in refine_results]

        data = []
        with open(src_path, "r", encoding='utf8') as f:
            src_lines = f.readlines()
            with open(trans_path, "r", encoding='utf8') as g:
                trans_lines = g.readlines()
                with open(ref_path, "r", encoding='utf8') as r:
                    ref_lines = r.readlines()
                    assert len(src_lines) == len(trans_lines), f"src lines: {len(src_lines)}, but trans lines: {len(trans_lines)}"
                    assert len(src_lines) == len(ref_lines), f"src lines: {len(src_lines)}, but ref lines: {len(ref_lines)}"
                    for i in range(len(src_lines)):
                        data.append([i, src_lines[i].strip(), trans_lines[i].strip(), ref_lines[i].strip()])
        # evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, refine_results, data, got_refine_eval)
        evaluate_got_refine_results(refine_results, data, metric_path,
                                    is_bleurt_20=is_bleurt_20, is_bleurt_d12=is_bleurt_d12,
                                    is_comet_0=is_comet_0, is_comet_x=is_comet_x,
                                    is_bleu=is_bleu)
    # return evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, refine_results, data, got_refine_eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-ref-path", type=str, default="/home/ubuntu/test/src-ref")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--bleurt-20", action="store_true")
    parser.add_argument("--bleurt-d12", action="store_true")
    parser.add_argument("--comet-0", action="store_true")
    parser.add_argument("--comet-x", action="store_true")
    parser.add_argument("--bleu", action="store_true")
    args = parser.parse_args()
    evaluate_from_file(args.src_ref_path, args.data_path,
                       is_bleurt_20=args.bleurt_20, is_bleurt_d12=args.bleurt_d12,
                       is_comet_0=args.comet_0, is_comet_x=args.comet_x,
                       is_bleu=args.bleu)