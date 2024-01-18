# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

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
from graph_of_thoughts import operations
from fairseq.models.transformer import TransformerModel

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def string_to_list(string: str) -> List[int]:
    """
    Helper function to convert a list encoded inside a string into a Python
    list object of string elements.

    :param string: Input string containing a list.
    :type string: str
    :return: List of string elements.
    :rtype: List[str]
    :raise AssertionError: If input string does not contain a list.
    """

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return [int(num) for num in string[1:-1].split(",")]


def test_sorting(state: Dict) -> bool:
    """
    Function to test whether the final solution matches ground truth.

    :param state: Thought state that represents the final solution.
    :type state: Dict
    :return: Returns whether the solution matches the ground truth.
    :rtype: bool
    """

    try:
        correct_list = sorted(string_to_list(state["original"]))
        sorted_list = string_to_list(state["current"])
        return sorted_list == correct_list
    except:
        return False


def num_errors(state: Dict) -> float:
    """
    Function to locally count the number of errors that serves as a score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """

    try:
        unsorted_list = state["original"]
        if (
            "unsorted_sublist" in state
            and state["unsorted_sublist"] != ""
            and state["unsorted_sublist"] is not None
            and len(state["unsorted_sublist"]) < len(unsorted_list) - 5
        ):
            unsorted_list = state["unsorted_sublist"]
        correct_list = sorted(string_to_list(unsorted_list))
        current_list = string_to_list(state["current"])
        num_errors = 0
        for i in range(10):
            num_errors += abs(
                sum([1 for num in current_list if num == i])
                - sum([1 for num in correct_list if num == i])
            )
        num_errors += sum(
            [1 for num1, num2 in zip(current_list, current_list[1:]) if num1 > num2]
        )
        return num_errors
    except:
        return 300


def get_random_indices(total_lines: int, num_samples: int = 4) -> List[int]:
    """
    Function to generate a list of random indices.

    :param total_lines: Total number of lines in the input file.
    :type total_lines: int
    :param num_samples: Number of samples to be generated, defaults to 4.
    :type num_samples: int, optional
    """

    # Randomly select indices based on a normal distribution
    # mean = total_lines // 2
    # std_dev = total_lines // 4
    indices = np.random.choice(np.arange(total_lines), size=num_samples, replace=False)

    # Clip indices to ensure they are within the valid range
    indices = np.clip(indices, 0, total_lines - 1)

    return indices.tolist()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def swish(x):
    return x * sigmoid(x)

def origin_symmetric_swish(x):
    if x <= 0:
        return swish(x)
    else:
        return -swish(-x)


def evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, hyp, data, result_path, gt4pseudo=None, pseudo=False):
    src = [x[1] for x in data]
    input = [x[2] for x in data]
    if pseudo:
        ref = [x[0] for x in gt4pseudo]
    else:
        ref = [x[3] for x in data]

    batch_size = 32
    bleurt_en_res = []
    bleurt_hyp_res = []

    for i in range(0, len(hyp), batch_size):
        end_idx = min(i + batch_size, len(ref))
        bleurt_en_res.extend(compute_bleurt_for_batch(ref[i:end_idx], input[i:end_idx], bleurt_model, bleurt_tokenizer))
        bleurt_hyp_res.extend(compute_bleurt_for_batch(ref[i:end_idx], hyp[i:end_idx], bleurt_model, bleurt_tokenizer))

        # calculate the average of the en_res and hyp_res

    # calculate the average of the en_res and hyp_res
    bleurt_input_avg = sum(bleurt_en_res) / len(bleurt_en_res)
    bleurt_hyp_avg = sum(bleurt_hyp_res) / len(bleurt_hyp_res)
    print("bleurt_en_avg: " + str(bleurt_input_avg))
    print("bleurt_hyp_avg: " + str(bleurt_hyp_avg))


    comet_hyp = []
    comet_input = []
    for i in range(len(ref)):
        comet_hyp.append({'src': src[i], 'mt': hyp[i], 'ref': ref[i]})
        comet_input.append({'src': src[i], 'mt': input[i], 'ref': ref[i]})

    bleu_hyp = sacrebleu.corpus_bleu(hyp, [ref]).score
    bleu_input = sacrebleu.corpus_bleu(input, [ref]).score

    model_hyp = comet_model.predict(comet_hyp, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
    model_input = comet_model.predict(comet_input, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]

    eval_results = {'bleu_hyp': bleu_hyp, 'bleu_input': bleu_input, 'comet_hyp': model_hyp, 'comet_input': model_input,
                    'bleurt_hyp': bleurt_hyp_avg, 'bleurt_input': bleurt_input_avg}
    with open(result_path, 'w', encoding='utf8') as f:
        json.dump(eval_results, f)


def compute_bleurt_for_batch(ref_batch, input_batch, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(ref_batch, input_batch, padding=True, return_tensors='pt', max_length=512,
                           truncation=True).to('cuda')
        logits = model(**inputs).logits.flatten().tolist()
    torch.cuda.empty_cache()
    return logits


def combined_score_auto_metric(states: List[Dict]) -> List[float]:
    scores = []
    for state in states:
        scores.append(auto_metric(state))
    return scores


def auto_metric(state: Dict) -> float:
    """
    Function to locally count the number of errors that serves as a score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """
    original = state["original"]
    previous = state["previous"]
    current = state["current"]
    if state['pseudo']:
        if previous != "":
            original = previous
        else:
            return 0.0

    bleurt_model = state["bleurt"]["model"]
    bleurt_tokenizer = state["bleurt"]["tokenizer"]

    bleurt_references = [original[3]]  # [id, src, trans, ref]
    bleurt_orig_trans = [original[2]]
    bleurt_refine_trans = [current[2]]

    with torch.no_grad():
        orig_inputs = bleurt_tokenizer(bleurt_references, bleurt_orig_trans, padding=True, return_tensors='pt', truncation=True).to(bleurt_model.device)
        refine_inputs = bleurt_tokenizer(bleurt_references, bleurt_refine_trans, padding=True, return_tensors='pt', truncation=True).to(bleurt_model.device)
        bleurt_orig_res = bleurt_model(**orig_inputs).logits.flatten().tolist()
        bleurt_refine_res = bleurt_model(**refine_inputs).logits.flatten().tolist()

    comet_model = state["comet"]["model"]
    comet_orig = [{'src': original[1], 'mt': original[2], 'ref': original[3]}]
    comet_refine = [{'src': original[1], 'mt': current[2], 'ref': original[3]}]
    comet_orig_res = comet_model.predict(comet_orig, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
    comet_refine_res = comet_model.predict(comet_refine, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]

    bleu_orig_res = sacrebleu.corpus_bleu([original[2]], [[original[3]]]).score
    bleu_refine_res = sacrebleu.corpus_bleu([current[2]], [[original[3]]]).score

    # normalize
    unnormed_orig = [bleu_orig_res, comet_orig_res*100, bleurt_orig_res[0]*100]
    unnormed_refine = [bleu_refine_res, comet_refine_res*100, bleurt_refine_res[0]*100]

    # normed_orig = [(x - min(unnormed_orig)) / (max(unnormed_orig) - min(unnormed_orig)) for x in unnormed_orig]
    # normed_refine = [(x - min(unnormed_refine)) / (max(unnormed_refine) - min(unnormed_refine)) for x in unnormed_refine]

    normed_orig = z_score_normlization(unnormed_orig)  # [<0, 1.x, 0.x]
    normed_refine = z_score_normlization(unnormed_refine)

    bleu_weight = state["sacrebleu"]["weight"]
    comet_weight = state["comet"]["weight"]
    bleurt_weight = state["bleurt"]["weight"]
    weight_sum = bleu_weight + comet_weight + bleurt_weight

    # weighted average
    orig_score = (normed_orig[0] * bleu_weight + normed_orig[1] * comet_weight + normed_orig[2] * bleurt_weight) / weight_sum
    refine_score = (normed_refine[0] * bleu_weight + normed_refine[1] * comet_weight + normed_refine[2] * bleurt_weight) / weight_sum

    delta = refine_score - orig_score
    if delta > 0:
        return math.exp(refine_score - orig_score) - 1
    else:
        return -math.exp(orig_score - refine_score) + 1
    # return math.exp(refine_score - orig_score) - 1


def combined_score_auto_metric_v2(states: List[Dict]) -> List[float]:
    scores = []
    for state in states:
        scores.append(auto_metric_v2(state))
    return scores


def auto_metric_v2(state: Dict) -> float:
    """
    Function to locally count the number of errors that serves as a score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """
    original = state["original"]
    previous = state["previous"]
    current = state["current"]
    if state['pseudo']:
        if previous != "":
            original = previous

    bleurt_model = state["bleurt"]["model"]
    bleurt_tokenizer = state["bleurt"]["tokenizer"]

    bleurt_references = [original[3]]  # [id, src, trans, ref]
    bleurt_refine_trans = [current[2]]

    with torch.no_grad():
        refine_inputs = bleurt_tokenizer(bleurt_references, bleurt_refine_trans, padding=True, return_tensors='pt', truncation=True).to(bleurt_model.device)
        bleurt_refine_res = bleurt_model(**refine_inputs).logits.flatten().tolist()

    comet_model = state["comet"]["model"]
    comet_refine = [{'src': original[1], 'mt': current[2], 'ref': original[3]}]
    comet_refine_res = comet_model.predict(comet_refine, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]

    bleu_refine_res = sacrebleu.corpus_bleu([current[2]], [[original[3]]]).score

    # normalize
    unnormed_refine = [bleu_refine_res, comet_refine_res*100, bleurt_refine_res[0]*100]

    # normed_orig = [(x - min(unnormed_orig)) / (max(unnormed_orig) - min(unnormed_orig)) for x in unnormed_orig]
    # normed_refine = [(x - min(unnormed_refine)) / (max(unnormed_refine) - min(unnormed_refine)) for x in unnormed_refine]

    normed_refine = z_score_normlization(unnormed_refine)

    bleu_weight = state["sacrebleu"]["weight"]
    comet_weight = state["comet"]["weight"]
    bleurt_weight = state["bleurt"]["weight"]
    weight_sum = bleu_weight + comet_weight + bleurt_weight

    # weighted average
    refine_score = (normed_refine[0] * bleu_weight + normed_refine[1] * comet_weight + normed_refine[2] * bleurt_weight) / weight_sum

    return refine_score


def combined_score_auto_metric_v2_test_only_bleurt(states: List[Dict]) -> List[float]:
    scores = []
    for state in states:
        for s in states:
            if s == state:
                continue
            state["other_results"].append(s["current"][2])
        scores.append(auto_metric_v2_test_only_bleurt(state))
    return scores


def auto_metric_v2_test_only_bleurt(state: Dict) -> float:  # reference-free
    """
    Function to locally count the number of errors that serves as a score.

    :param state: Thought state to be scored.waa
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """
    bleurt_references = [state["current"][2]]
    bleurt_refine_trans = state["other_results"]
    
    bleurt_model = state["bleurt"]["model"]
    bleurt_tokenizer = state["bleurt"]["tokenizer"]

    total_score = 0.0
    with torch.no_grad():
        for t in bleurt_refine_trans:
            refine_inputs = bleurt_tokenizer(bleurt_references, [t], padding=True, return_tensors='pt',
                                         truncation=True).to(bleurt_model.device)
            bleurt_refine_res = bleurt_model(**refine_inputs).logits.flatten().tolist()
            total_score += bleurt_refine_res[0]


    return total_score / len(bleurt_refine_trans)


def z_score_normlization(data):
    mean = np.mean(data)
    std = np.std(data)

    assert std != 0, "std is 0"
    return [(i - mean) / std for i in data]


def evaluate_from_file(timestamp: str, test_lang: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bleurt_cache_dir = r'/mnt/e/unmt/stanford_alpaca/models/bleurt'
    bleurt_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20',
                                                                   cache_dir=bleurt_cache_dir).to(device)
    bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    bleurt_model.eval()

    comet_model_path = '/home/slpan/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
    comet_model = load_from_checkpoint(comet_model_path).eval()

    data_path = os.path.join(os.path.dirname(__file__), "data", "x2x", f"{test_lang}2en")
    src_path = os.path.join(data_path, "src")
    trans_path = os.path.join(data_path, "hyp")
    ref_path = os.path.join(data_path, "ref")

    pair_name = f"results/{test_lang}2en"
    folder_name = pair_name + f"/{timestamp}"
    got_refine_hyp = os.path.join(os.path.dirname(__file__), folder_name, 'got_refine')
    got_refine_eval = os.path.join(os.path.dirname(__file__), folder_name, 'metrics.json')
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
    evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, refine_results, data, got_refine_eval)


def output_sample_graph(graph: operations.GraphOfOperations, path: str) -> None:
    """
    Serialize the state and results of the operations graph to a JSON file.

    :param path: The path to the output file.
    :type path: str
    """
    output = []
    for operation in graph.operations:
        operation_serialized = {
            "operation": operation.operation_type.name,
        }
        if operation.operation_type.name == "generate":
            operation_serialized["aux_lang"] = operation.lang
        if operation.operation_type.name == "aggregate":
            operation_serialized["aux_lang"] = []
            for op in operation.predecessors:
                if op.operation_type.name == "generate":
                    operation_serialized["aux_lang"].append(op.lang)
                else:
                    for o in op.predecessors:
                        if o.operation_type.name == "generate":
                            operation_serialized["aux_lang"].append(o.lang)
            # operation_serialized["aux_lang"] = [op.lang for op in operation.predecessors.predecessors]

        output.append(operation_serialized)


    with open(path, "w") as file:
        file.write(json.dumps(output, indent=2))


def bt_nmt(state: Dict):
    """
    back translate target sentences to source language with NMT model

    Returns: bt source sentences

    """
    # Load model
    
