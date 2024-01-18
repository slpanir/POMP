# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import os
import logging
import datetime
import json
import csv
import numpy as np
import math
from copy import deepcopy
from typing import Dict, List, Callable, Union, Tuple, Any
from collections import OrderedDict
import threading

import torch
from comet import download_model, load_from_checkpoint
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

from graph_of_thoughts import controller, operations, prompter, parser
from graph_of_thoughts.operations.operations import KeepBestN, Generate, Aggregate, Score, Operation
from graph_of_thoughts.operations.thought import Thought
import utils
import faulthandler
faulthandler.enable()


# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

class RefineTranslationPrompter(prompter.Prompter):
    """
    SortingPrompter provides the generation of prompts specific to the sorting
    example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    def generate_prompt(self, **kwargs) -> str:
        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        ref_path = kwargs["ref_path"]
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        aux_lang_list = kwargs["aux_lang"]
        indices = kwargs["indices"]

        # aux_lang = similarity.pop(0)[0]
        # kwargs['aux_lang'] = aux_lang
        assert len(aux_lang_list) == 1, "Expected exactly one auxiliary language in Generation Operation."
        aux_lang = aux_lang_list[0]
        aux_path = os.path.join(data_path, f"{language}2{aux_lang}", aux_lang)

        prompt_path = os.path.join(prompt_path, f"add_{aux_lang}_prompt.txt")

        # Extract sentences from the files based on the selected indices
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        aux_sentences = [line.strip() for idx, line in enumerate(open(aux_path)) if idx in indices]
        input_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]
        ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]

        # Construct the prompt
        LANG = lang_map[language]
        AUX_LANG = lang_map[aux_lang]
        prompt = ""
        for s, a, i, r in zip(src_sentences, aux_sentences, input_sentences, ref_sentences):
            prompt += f"<{LANG} source>: {s}\n<{AUX_LANG} translation>: {a}\n<English translation>: {i}\n<Refined translation>: {r}\n\n"
        # src = "{" + "{language}_text".format(language=language.lower()) + "}"
        src = "{src_text}"
        au = "{aux_translation}"
        en = "{english_translation}"
        prompt += f"<{LANG} source>: {src}\n<{AUX_LANG} translation>: {au}\n<English translation>: {en}\n<Refined translation>:"

        with open(aux_path, 'r') as f:
            aux_text = f.readlines()[kwargs["original"][0]].strip()
        prompt = prompt.format(src_text=kwargs["original"][1], aux_translation=aux_text,
                      english_translation=kwargs["original"][2])
        # if prompt.txt does not exist, create it
        if not os.path.exists(prompt_path):
            with open(prompt_path, 'w', encoding='utf8') as pp:
                pp.write(str(indices) + '\n')
                pp.write(prompt)

        return prompt


    def generate_prompt_pseudo(self, **kwargs) -> str:
        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        ref_path = kwargs["ref_path"]
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        aux_lang_list = kwargs["aux_lang"]
        indices = kwargs["indices"]

        # aux_lang = similarity.pop(0)[0]
        # kwargs['aux_lang'] = aux_lang
        assert len(aux_lang_list) == 1, "Expected exactly one auxiliary language in Generation Operation."
        aux_lang = aux_lang_list[0]
        aux_path = os.path.join(data_path, f"{language}2{aux_lang}", aux_lang)

        prompt_path = os.path.join(prompt_path, f"add_{aux_lang}_prompt.txt")

        # Extract sentences from the files based on the selected indices
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        aux_sentences = [line.strip() for idx, line in enumerate(open(aux_path)) if idx in indices]
        input_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]
        ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]

        # Construct the prompt
        LANG = lang_map[language]
        AUX_LANG = lang_map[aux_lang]
        prompt = ""
        for s, a, i in zip(src_sentences, aux_sentences, input_sentences):
            prompt += f"<{LANG} source>: {s}\n<{AUX_LANG} translation>: {a}\n<English translation>: {i}\n\n"
        # src = "{" + "{language}_text".format(language=language.lower()) + "}"
        src = "{src_text}"
        au = "{aux_translation}"
        en = "{english_translation}"
        prompt += f"<{LANG} source>: {src}\n<{AUX_LANG} translation>: {au}\n<English translation>:"

        with open(aux_path, 'r') as f:
            aux_text = f.readlines()[kwargs["original"][0]].strip()
        prompt = prompt.format(src_text=kwargs["original"][1], aux_translation=aux_text)
        # if prompt.txt does not exist, create it
        if not os.path.exists(prompt_path):
            with open(prompt_path, 'w', encoding='utf8') as pp:
                pp.write(str(indices) + '\n')
                pp.write(prompt)

        return prompt


    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If not exactly two thought states are provided.
        """
        assert len(state_dicts) >= 2, "Expected two states at least for aggregation prompt."

        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        ref_path = kwargs["ref_path"]
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        indices = kwargs["indices"]

        aux_langs = [lang for state in state_dicts for lang in state['aux_lang']]
        aux_langs = list(dict.fromkeys(aux_langs))
        prompt_path = os.path.join(prompt_path, f"add_{str('-'.join(aux_langs))}_prompt.txt")

        # Extract sentences from the files based on the selected indices
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        input_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]
        ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]

        # Construct the prompt
        LANG = lang_map[language]
        prompt = ""
        aux = "{}"
        for s, i, r in zip(src_sentences, input_sentences, ref_sentences):
            prompt += f"<{LANG} source>: {s}\n{aux}<English translation>: {i}\n<Refined translation>: {r}\n\n"
        prompt += f"<{LANG} source>: {state_dicts[0]['original'][1]}\n{aux}<English translation>: {state_dicts[0]['current'][2]}\n<Refined translation>:"
        aux_list = []
        aux_template = "<{AUX_LANG} translation>: {aux_translation}\n"

        # for state in state_dicts:
        #     for aux_lang in state["aux_lang"]:
        for aux_lang in aux_langs:
            AUX_LANG = lang_map[aux_lang]
            aux_path = os.path.join(data_path, f"{language}2{aux_lang}", aux_lang)
            aux_sentences = [line.strip() for idx, line in enumerate(open(aux_path)) if idx in indices]
            if aux_list == []:
                for sent in aux_sentences:
                    aux_list.append(aux_template.format(AUX_LANG=AUX_LANG, aux_translation=sent))
                with open(aux_path, 'r') as f:
                    aux_list.append(aux_template.format(AUX_LANG=AUX_LANG, aux_translation=f.readlines()[kwargs["original"][0]].strip()))
            else:
                for idx, sent in enumerate(aux_sentences):
                    aux_list[idx] += aux_template.format(AUX_LANG=AUX_LANG, aux_translation=sent)
                with open(aux_path, 'r') as f:
                    aux_list[idx + 1] += aux_template.format(AUX_LANG=AUX_LANG, aux_translation=f.readlines()[kwargs["original"][0]].strip())
        # 检查prompt中{aux}数量与aux_list长度是否一致
        assert prompt.count("{}") == len(aux_list), "The number of {aux_prompt} in prompt is not equal to the length of aux_list."
        prompt = prompt.format(*aux_list)

        if not os.path.exists(prompt_path):
            with open(prompt_path, 'w', encoding='utf8') as pp:
                pp.write(str(indices) + '\n')
                pp.write(prompt)

        return prompt


    def aggregation_prompt_pseudo(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If not exactly two thought states are provided.
        """
        assert len(state_dicts) >= 2, "Expected two states at least for aggregation prompt."

        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        ref_path = kwargs["ref_path"]
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        indices = kwargs["indices"]

        aux_langs = [lang for state in state_dicts for lang in state['aux_lang']]
        prompt_path = os.path.join(prompt_path, f"add_{str('-'.join(aux_langs))}_prompt.txt")

        # Extract sentences from the files based on the selected indices
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        input_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]
        ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]

        # Construct the prompt
        LANG = lang_map[language]
        prompt = ""
        aux = "{}"
        for s, i, r in zip(src_sentences, input_sentences):
            prompt += f"<{LANG} source>: {s}\n{aux}<English translation>: {i}\n<Refined translation>: {r}\n\n"
        prompt += f"<{LANG} source>: {state_dicts[0]['original'][1]}\n{aux}<English translation>: {state_dicts[0]['current'][2]}\n<Refined translation>:"
        aux_list = []
        aux_template = "<{AUX_LANG} translation>: {aux_translation}\n"
        for state in state_dicts:
            for aux_lang in state["aux_lang"]:
                AUX_LANG = lang_map[aux_lang]
                aux_path = os.path.join(data_path, f"{language}2{aux_lang}", aux_lang)
                aux_sentences = [line.strip() for idx, line in enumerate(open(aux_path)) if idx in indices]
                if aux_list == []:
                    for sent in aux_sentences:
                        aux_list.append(aux_template.format(AUX_LANG=AUX_LANG, aux_translation=sent))
                    with open(aux_path, 'r') as f:
                        aux_list.append(aux_template.format(AUX_LANG=AUX_LANG, aux_translation=f.readlines()[kwargs["original"][0]].strip()))
                else:
                    for idx, sent in enumerate(aux_sentences):
                        aux_list[idx] += aux_template.format(AUX_LANG=AUX_LANG, aux_translation=sent)
                    with open(aux_path, 'r') as f:
                        aux_list[idx + 1] += aux_template.format(AUX_LANG=AUX_LANG, aux_translation=f.readlines()[kwargs["original"][0]].strip())
        # 检查prompt中{aux}数量与aux_list长度是否一致
        assert prompt.count("{}") == len(aux_list), "The number of {aux_prompt} in prompt is not equal to the length of aux_list."
        prompt = prompt.format(*aux_list)

        if not os.path.exists(prompt_path):
            with open(prompt_path, 'w', encoding='utf8') as pp:
                pp.write(str(indices) + '\n')
                pp.write(prompt)

        return prompt


    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        """
        pass


class RefineTranslationParser(parser.Parser):
    """
    SortingParser provides the parsing of language model reponses specific to
    the sorting example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        :raise AssertionError: If not exactly two thought states are provided.
        """

        assert len(states) >= 2, "Expected two states at least for aggregation answer."
        assert len(texts) == 1, "Expected exactly one response for aggregation answer."
        new_states = []
        text = texts[0]
        try:
            new_state = states[0].copy()
            if states[0]['current'] != "":
                pre = states[0]['current'].copy()
                new_state['previous'] = pre
            cur = states[0]["original"].copy()
            # cur[2] = text.strip()
            cur[2] = text.strip().split("\n")[0]
            new_state["current"] = cur
            # aux_lang list
            aux_langs = [lang for state in states for lang in state['aux_lang']]
            aux_langs = list(dict.fromkeys(aux_langs))
            new_state["aux_lang"] = aux_langs
            new_states.append(new_state)
        except Exception as e:
            logging.error(
                f"Could not parse step answer: {text}. Encountered exception: {e}"
            )
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            try:
                new_state = state.copy()
                if state['current'] != "":
                    pre = state['current'].copy()
                    new_state['previous'] = pre
                cur = state["original"].copy()
                # cur[2] = text.strip()
                cur[2] = text.strip().split("\n")[0]
                new_state["current"] = cur
                new_states.append(new_state)
            except Exception as e:
                logging.error(
                    f"Could not parse step answer: {text}. Encountered exception: {e}"
                )
            # if state["method"] == "got" and state["current"] == "":
            #     # We expect a json which contains the four lists named "List 1" to "List 4"
            #     # cut everything until the opening bracket and everything after the closing bracket
            #     try:
            #         new_state = state.copy()
            #         cur = state["original"].copy()
            #         cur[2] = text.strip()
            #         new_state["current"] = cur
            #         new_states.append(new_state)
            #     except Exception as e:
            #         logging.error(
            #             f"Could not parse step answer: {text}. Encountered exception: {e}"
            #         )
            # else:  #
            #
            #     answers = text.strip().split("\n")
            #     answers = [
            #         answer for answer in answers if "[" in answer and "]" in answer
            #     ]
            #     if any(["Output" in answer for answer in answers]):
            #         # cut elements until last output is found
            #         for answer in reversed(answers):
            #             if "Output" in answer:
            #                 answers = answers[answers.index(answer) :]
            #                 break
            #
            #     answers = [
            #         answer[answer.index("[") : answer.index("]") + 1]
            #         for answer in answers
            #     ]
            #     if len(answers) == 0:
            #         logging.warning(
            #             f"Could not parse step answer: {text}. Returning empty list."
            #         )
            #         answer = "[]"
            #     else:
            #         if len(answers) > 1:
            #             logging.warning(
            #                 f"Multiple answers found for step answer: {text}. Using the first one."
            #             )
            #         answer = answers[0]
            #
            #     new_state = state.copy()
            #     new_state["current"] = answer
            #     new_state["phase"] = 2
            #     new_states.append(new_state)
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 20))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(1):
        operations_graph.append_operation(operations.Generate(1, 20))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT2 method.
    ToT2 uses a tree with more levels, but with fewer branches per level.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, 10))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def got(gen_aux=3, merge_aux=2) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    gen1 = operations.Generate(1, 1)
    score1 = operations.Score(1, False, utils.auto_metric)
    keep_best1 = operations.KeepBestN(1, True)
    score1.add_predecessor(gen1)
    keep_best1.add_predecessor(score1)

    gen2 = operations.Generate(1, 1)
    score2 = operations.Score(1, False, utils.auto_metric)
    keep_best2 = operations.KeepBestN(1, True)
    score2.add_predecessor(gen2)
    keep_best2.add_predecessor(score2)

    gen3 = operations.Generate(1, 1)
    score3 = operations.Score(1, False, utils.auto_metric)
    keep_best3 = operations.KeepBestN(1, True)
    score3.add_predecessor(gen3)
    keep_best3.add_predecessor(score3)

    agg12 = operations.Aggregate(num_merges=merge_aux)
    agg12.add_predecessor(keep_best1)
    agg12.add_predecessor(keep_best2)

    agg23 = operations.Aggregate(num_merges=merge_aux)
    agg23.add_predecessor(keep_best2)
    agg23.add_predecessor(keep_best3)

    agg13 = operations.Aggregate(num_merges=merge_aux)
    agg13.add_predecessor(keep_best1)
    agg13.add_predecessor(keep_best3)

    # PLAN A
    scoreALL = operations.Score(1, True, utils.combined_score_auto_metric)
    keep_bestALL = operations.KeepBestN(1, True)
    scoreALL.add_predecessor(agg12)
    scoreALL.add_predecessor(agg23)
    scoreALL.add_predecessor(agg13)
    keep_bestALL.add_predecessor(scoreALL)

    operations_graph.add_operation(gen1)
    operations_graph.add_operation(score1)
    operations_graph.add_operation(keep_best1)
    operations_graph.add_operation(gen2)
    operations_graph.add_operation(score2)
    operations_graph.add_operation(keep_best2)
    operations_graph.add_operation(gen3)
    operations_graph.add_operation(score3)
    operations_graph.add_operation(keep_best3)
    operations_graph.add_operation(agg12)
    operations_graph.add_operation(agg23)
    operations_graph.add_operation(agg13)
    operations_graph.add_operation(scoreALL)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph


def full_got(max_gen_aux=6, max_merge_aux=2) -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()
    score_list = []
    for i in range(0, max_gen_aux):
        gen = operations.Generate(1, 1)
        score = operations.Score(1, False, utils.auto_metric)
        score.add_predecessor(gen)
        score_list.append(gen)

        operations_graph.add_operation(gen)
        operations_graph.add_operation(score)

    assert max_merge_aux <= len(score_list), "max_merge_aux should be less than or equal to max_gen_aux."

    scoreALL = operations.Score(1, True, utils.combined_score_auto_metric)
    for i in range(2, max_merge_aux + 1):
        for j in range(0, len(score_list) - i + 1):
            agg = operations.Aggregate(num_merges=i)
            for k in range(j, j + i):
                agg.add_predecessor(score_list[k])
            scoreALL.add_predecessor(agg)
            operations_graph.add_operation(agg)
    operations_graph.add_operation(scoreALL)

    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreALL)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph


def sample_gen_agg_keys(gen_probability) -> Tuple[List[Any], List[Any]]:
    # normalize probability
    # gen_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}
    # randomly sample keys according to probability
    sample_gen_keys = []
    for key in gen_probability.keys():
        if np.random.random() < gen_probability[key]:
            sample_gen_keys.append(key)
    len_gen_aux = len(sample_gen_keys)

    agg_probability = {}
    # aggregate all possible combinations
    for i in range(2, len_gen_aux + 1):
        for j in range(0, len_gen_aux - i + 1):
            agg_prob = 1.0
            for key in sample_gen_keys[j:j+i]:
                agg_prob *= gen_probability[key]
            agg_key = tuple(sample_gen_keys[j:j+i])
            agg_probability[agg_key] = agg_prob

    # normalize probability
    # agg_probability = {k: v / sum(agg_probability.values()) for k, v in agg_probability.items()}
    # randomly sample keys according to probability
    sample_agg_keys = []
    for key in agg_probability.keys():
        if np.random.random() < agg_probability[key]:
            sample_agg_keys.append(key)

    return sample_gen_keys, sample_agg_keys


def sample_got(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    score_operations = {}
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        score = operations.Score(1, False, utils.auto_metric)
        score.add_predecessor(gen)
        operations_graph.add_operation(gen)
        operations_graph.add_operation(score)
        score_operations[sample_gen_keys[i]] = score

    scoreALL = operations.Score(1, True, utils.combined_score_auto_metric)
    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreALL)
    added_score_operations = []
    if not sample_agg_keys:
        for key in sample_gen_keys:
            score_op = score_operations[key]
            scoreALL.add_predecessor(score_op)
    else:
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key))
            for key in agg_key:
                score_op = score_operations[key]
                agg.add_predecessor(score_op)
                added_score_operations.append(score_op)
            agg.add_successor(scoreALL)
            operations_graph.add_operation(agg)
    for score in score_operations.values():
        if score not in added_score_operations:
            keep_bestALL.add_predecessor(score)
    operations_graph.add_operation(scoreALL)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

async def multi_threads_train(
    train_lang: str,
    tgt_lang: str,
    similarity: dict,
    lang_map: dict,
    methods: List[Callable[..., operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
    threads_num: int = 1,
    lr: float = 1e-2,
    pseudo: bool = False,
    aux_probs_from_file: str = None,
) -> float:

    orig_budget = budget
    # need lock: refine_results, lm.cost, lm.api_key_list
    lock = threading.Lock()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bleurt_cache_dir = r'/mnt/e/unmt/stanford_alpaca/models/bleurt'
    bleurt_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20',
                                                                   cache_dir=bleurt_cache_dir).to(device)
    bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    bleurt_model.eval()

    comet_model_path = '/home/slpan/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
    comet_model = load_from_checkpoint(comet_model_path).eval().to(device)

    data_path = os.path.join(os.path.dirname(__file__), "data", "train-set4test-langs", f"{train_lang}2en")
    src_path = os.path.join(data_path, "src")
    trans_path = os.path.join(data_path, "hyp")
    ref_path = os.path.join(data_path, "ref")
    # data = []
    # with open(src_path, "r", encoding='utf8') as f:
    #     src_lines = f.readlines()
    #     with open(trans_path, "r", encoding='utf8') as g:
    #         trans_lines = g.readlines()
    #         assert len(src_lines) == len(
    #             trans_lines), f"src lines: {len(src_lines)}, but trans lines: {len(trans_lines)}"
    #         if pseudo:
    #             for i in range(len(src_lines)):
    #                 data.append([i, src_lines[i].strip(), trans_lines[i].strip(), trans_lines[i].strip()])
    #         else:
    #             with open(ref_path, "r", encoding='utf8') as r:
    #                 ref_lines = r.readlines()
    #                 assert len(src_lines) == len(
    #                     ref_lines), f"src lines: {len(src_lines)}, but ref lines: {len(ref_lines)}"
    #                 for i in range(len(src_lines)):
    #                     data.append([i, src_lines[i].strip(), trans_lines[i].strip(), ref_lines[i].strip()])
    data = []
    gt4pseudo = []
    # 使用zip函数同时迭代三个文件的每一行
    with open(src_path, "r", encoding='utf8') as f_src, \
            open(trans_path, "r", encoding='utf8') as f_trans, \
            open(ref_path, "r", encoding='utf8') as f_ref:

        for i, (src_line, trans_line, ref_line) in enumerate(zip(f_src, f_trans, f_ref)):
            src_line = src_line.strip()
            trans_line = trans_line.strip()
            ref_line = ref_line.strip()

            # 确保源文件和参考文件的行数相同
            assert src_line and trans_line and ref_line, f"Mismatch at line {i}"

            if pseudo:
                # 如果是伪翻译，参考文本就是翻译文本
                data.append([i, src_line, trans_line, trans_line])
                gt4pseudo.append([ref_line])
            else:
                data.append([i, src_line, trans_line, ref_line])

    # 如果文件行数不匹配，zip会停止在最短文件的末尾，所以如果你需要确保所有文件具有相同的行数，
    # 你应该在循环结束后检查文件是否已经到达末尾

    data = data[300:500]
    gt4pseudo = gt4pseudo[300:500]

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.join(os.path.dirname(__file__), "results")))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    pair_name = f"results/{train_lang}2{tgt_lang}"
    os.makedirs(os.path.join(os.path.dirname(__file__), pair_name), exist_ok=True)
    folder_name = pair_name + f"/{extra_info}_{timestamp}"
    os.makedirs(os.path.join(os.path.dirname(__file__), folder_name))

    config = {
        "data": data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    # support Chinese
    with open(
        os.path.join(os.path.join(os.path.dirname(__file__), folder_name, "config.json")), "w",
    ) as f:
        json.dump(config, f, ensure_ascii=False)

    logging.basicConfig(
        filename=os.path.join(os.path.dirname(__file__), folder_name, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(
            os.path.join(os.path.dirname(__file__), folder_name, method.__name__)
        )

    indices = utils.get_random_indices(len(data))
    logging.info(f"Using indices: {indices} for {train_lang}2{tgt_lang} in few-shot setting.")

    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
    )
    orig_num_keys = len(lm.api_key_list)

    # sim = deepcopy(similarity[train_lang])
    # convert similarity to probability
    aux_probability = {}
    if aux_probs_from_file:
        aux_probs_from_file = os.path.join(aux_probs_from_file, f'{train_lang}_probs.json')
        with open(aux_probs_from_file, 'r', encoding='utf8') as f:
            aux_probability = json.load(f)
    else:
        for item in similarity[train_lang]:  # [(lang, sim),]
            k = item[0]
            v = np.exp(-1.0 + item[1])  # exp
            # v = 1 / (1 + np.exp(-1 * (item[1] - 0.5)))  # sigmoid
            aux_probability[k] = v
    method = methods[0]
    # operations_graph = method(aux_probability)

    state = {
                "original": "",
                "previous": "",
                "current": "",
                "method": method.__name__,
                "data_path": data_path,
                "results_path": os.path.join(os.path.dirname(__file__), folder_name),
                "prompt_path": os.path.join(os.path.dirname(__file__), folder_name, method.__name__),
                "src_path": src_path,
                "trans_path": trans_path,
                "ref_path": ref_path,
                "indices": indices,
                "language": train_lang,
                # "similarity": sim,
                "lang_map": lang_map,
                "aux_lang": [],
                "bleurt": {
                    "model": bleurt_model,
                    "tokenizer": bleurt_tokenizer,
                    "config": bleurt_config,
                    "weight": 0.3,
                },
                "comet": {
                    "model": comet_model,
                    "weight": 0.1,
                },
                "sacrebleu": {
                    "weight": 0.6,
                },
                "score": 0.0,
                "pseudo": pseudo,
                "lock": lock,
            }
    refine_results = OrderedDict()
    unchanged_num = 0
    threads = []
    total_api_key_num = 0
    total_cost = 0.0
    for i in range(threads_num):
        start_index = int(len(data) / threads_num * i)
        end_index = int(len(data) / threads_num * (i + 1)) if i < threads_num - 1 else len(data)
        api_key_start_index = int(len(lm.api_key_list) / threads_num * i)
        api_key_end_index = int(len(lm.api_key_list) / threads_num * (i + 1)) if i < threads_num - 1 else len(lm.api_key_list)
        api_key_list = lm.api_key_list[api_key_start_index:api_key_end_index]
        thread_state = {
            "original": deepcopy(state["original"]),
            "previous": deepcopy(state["previous"]),
            "current": deepcopy(state["current"]),
            "method": state["method"],
            "data_path": state["data_path"],
            "results_path": state["results_path"],
            "prompt_path": state["prompt_path"],
            "src_path": state["src_path"],
            "trans_path": state["trans_path"],
            "ref_path": state["ref_path"],
            "indices": state["indices"],
            "language": state["language"],
            # "similarity": deepcopy(state["similarity"]),
            "lang_map": state["lang_map"],
            "aux_lang": deepcopy(state["aux_lang"]),
            "bleurt": state["bleurt"],
            "comet": state["comet"],
            "sacrebleu": state["sacrebleu"],
            "score": deepcopy(state["score"]),
            "pseudo": state["pseudo"],
            "lock": state["lock"],
        }
        t = threading.Thread(target=train, args=(lm_name, api_key_list, start_index, end_index, data, method,
                                                 aux_probability, thread_state, refine_results, unchanged_num,
                                                 lock, total_api_key_num, total_cost, budget, lr))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # save aux_probability
    logging.info(aux_probability)
    got_refine_probability = os.path.join(os.path.dirname(__file__), folder_name, f'{train_lang}_probs.json')
    got_refine_probability_json = json.dumps(aux_probability, indent=2)
    f = open(got_refine_probability, 'w', encoding='utf8')
    f.write(got_refine_probability_json)

    refine_outputs = [refine_results[i] for i in sorted(refine_results.keys())]
    got_refine_hyp = os.path.join(os.path.dirname(__file__), folder_name, 'train_got_refine')
    got_refine_eval = os.path.join(os.path.dirname(__file__), folder_name, 'train_metrics.json')
    with open(got_refine_hyp, 'w', encoding='utf8') as f:
        for result in refine_outputs:
            f.write(result + '\n')
    try:
        utils.evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, refine_outputs, data, got_refine_eval, gt4pseudo, pseudo)
    except Exception as e:
        logging.error(f"Exception: {e}")
        logging.warning(f"Failed to evaluate got refine results.")

    # check keys
    if total_api_key_num != orig_num_keys:
        logging.warning(
            f"Number of keys has changed from {orig_num_keys} to {len(lm.api_key_list)}.")
        with open(os.path.join(os.path.dirname(__file__), folder_name, method.__name__, 'new_key_list.txt'), 'w', encoding='utf8') as f:
            f.write("[\n")
            for key in lm.api_key_list:
                f.write(key + '\n')
            f.write("]\n")

    return orig_budget - total_cost


async def train(
    lm_name: str,
    api_key_list: List[str],
    start_index: int,
    end_index: int,
    data: List[Any],
    method: Callable[..., operations.GraphOfOperations],
    # operations_graph: operations.GraphOfOperations,
    aux_probability: Dict,
    state: Dict,
    refine_results: OrderedDict,
    unchanged_num: int,
    lock: threading.Lock,
    total_api_key_num: int,
    total_cost: float,
    budget: float,
    lr: float,
    score_max: float = 1.40,
) -> None:

    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
        api_key_list=api_key_list,
    )

    item = start_index
    # orig_sim = deepcopy(state['similarity'])
    while item < end_index:  # [id, src, en_trans, ref]
        # state['similarity'] = deepcopy(orig_sim)
        with lock:
            operations_graph = method(aux_probability)
        state["original"] = data[item]
        state["previous"] = ""
        state["current"] = ""
        state["aux_lang"] = []
        state["score"] = 0.0
        logging.info(f"Running sample {item}")
        # if budget <= 0.0:
        if total_cost > budget:
            logging.error(
                f"Budget has been depleted, stopping. Sample {item} has not been run."
            )
            break
        # for method in methods:
        logging.info(f"Running method {method.__name__}")
        logging.info(f"Budget left: {budget - total_cost}")
        # if budget <= 0.0:
        if lm.cost > budget:
            logging.error(
                f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
            )
            break
        executor = controller.Controller(
            lm,
            operations_graph,
            RefineTranslationPrompter(),
            RefineTranslationParser(),
            state,
            lock,
        )
        try:
            executor.run()
        except Exception as e:
            logging.error(f"Exception: {e}")
        # path = os.path.join(
        #     os.path.dirname(__file__),
        #     folder_name,
        #     method.__name__,
        #     f"{data.index(d)}.json",
        # )
        # executor.output_graph(path)
        # budget -= lm.cost

        # update aux_probability
        # state["similarity"] = None
        best_score = -1000.05
        scores = []
        try:
            last_operation: KeepBestN = executor.graph.leaves[-1]
            previous_thoughts: List[Thought] = last_operation.get_previous_thoughts()
            assert all(
                previous_thought.scored for previous_thought in previous_thoughts
            ), "Not all thoughts have been scored"
            best_thought: Thought = last_operation.get_best_n()[0]
            best_score = best_thought.score
            best_score = min(best_score, score_max)
            with lock:
                for thought in previous_thoughts:
                    score = thought.score
                    scores.append(score)
                    aux_lang = thought.state['aux_lang']
                    for aux in aux_lang:
                        # temp = aux_probability[aux] + lr * abs(score)
                        # temp = aux_probability[aux] * math.exp(score)
                        temp = aux_probability[aux] * (1 + score * lr)
                        temp = temp * 0.2 + aux_probability[aux] * 0.8
                        temp = min(temp, 1.0)
                        aux_probability[aux] = temp
                    # if thought == best_thought:
                    #     for aux in aux_lang:
                    #         # temp = aux_probability[aux] + lr * abs(score)
                    #         temp = aux_probability[aux] * math.exp(score)
                    #         temp = temp * 0.2 + aux_probability[aux] * 0.8
                    #         temp = min(temp, 1.0)
                    #         if math.isnan(temp):
                    #             continue
                    #         else:
                    #             aux_probability[aux] = temp
                    # else:
                    #     for aux in aux_lang:
                    #         # temp = aux_probability[aux] - lr * abs(score)
                    #         temp = aux_probability[aux] * math.exp(score)
                    #         temp = temp * 0.2 + aux_probability[aux] * 0.8
                    #         temp = min(temp, 1.0)
                    #         if math.isnan(temp):
                    #             continue
                    #         else:
                    #             aux_probability[aux] = temp
            logging.warning(f"aux probability success to update for {item}")
        except Exception as e:
            logging.error(f"Exception: {e}")
            logging.warning(f"aux probability fails to update for {item}")

        # record local results
        try:
            # refine_results.append(executor.graph.leaves[0].thoughts[0].state["current"][2])
            with lock:
                refine_results[item] = executor.graph.leaves[-1].thoughts[0].state["current"][2]
                if refine_results[item] == data[item][2] or refine_results[item] == data[item][3]:
                    unchanged_num += 1
        except:
            # refine_results.append(d[2])
            with lock:
                if state['pseudo']:
                    refine_results[item] = data[item][3]
                else:
                    refine_results[item] = data[item][2]
                unchanged_num += 1
        # save temp results
        temp_res = {
            "aux_probs": aux_probability,
            "output": refine_results[item],
            "best_score": best_score,
            "scores": scores,
        }
        temp_res_json = json.dumps(temp_res, indent=2)
        f = open(os.path.join(state['results_path'], f'{item}'), 'w', encoding='utf8')
        f.write(temp_res_json)
        f.close()

        item += 1
        with lock:
            lr = lr * 0.99

    with lock:
        total_api_key_num += len(lm.api_key_list)
        total_cost += lm.cost

def run(
    test_lang: str,
    tgt_lang: str,
    similarity: dict,
    lang_map: dict,
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bleurt_cache_dir = r'/mnt/e/unmt/stanford_alpaca/models/bleurt'
    bleurt_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20',
                                                                   cache_dir=bleurt_cache_dir).to(device)
    bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20', cache_dir=bleurt_cache_dir)
    bleurt_model.eval()

    comet_model_path = '/home/slpan/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
    comet_model = load_from_checkpoint(comet_model_path).eval().to(device)

    data_path = os.path.join(os.path.dirname(__file__), "data", "x2x", f"{test_lang}2en")
    src_path = os.path.join(data_path, "src")
    trans_path = os.path.join(data_path, "hyp")
    ref_path = os.path.join(data_path, "ref")
    data = []
    with open(src_path, "r", encoding='utf8') as f_src, \
            open(trans_path, "r", encoding='utf8') as f_trans, \
            open(ref_path, "r", encoding='utf8') as f_ref:

        for i, (src_line, trans_line, ref_line) in enumerate(zip(f_src, f_trans, f_ref)):
            src_line = src_line.strip()
            trans_line = trans_line.strip()
            ref_line = ref_line.strip()

            # 确保源文件和参考文件的行数相同
            assert src_line and trans_line and ref_line, f"Mismatch at line {i}"
            data.append([i, src_line, trans_line, ref_line])

    # with open(src_path, "r", encoding='utf8') as f:
    #     src_lines = f.readlines()
    #     with open(trans_path, "r", encoding='utf8') as g:
    #         trans_lines = g.readlines()
    #         with open(ref_path, "r", encoding='utf8') as r:
    #             ref_lines = r.readlines()
    #             assert len(src_lines) == len(trans_lines), f"src lines: {len(src_lines)}, but trans lines: {len(trans_lines)}"
    #             assert len(src_lines) == len(ref_lines), f"src lines: {len(src_lines)}, but ref lines: {len(ref_lines)}"
    #             for i in range(len(src_lines)):
    #                 data.append([i, src_lines[i].strip(), trans_lines[i].strip(), ref_lines[i].strip()])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.join(os.path.dirname(__file__), "results")))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    pair_name = f"results/{test_lang}2{tgt_lang}"
    os.makedirs(os.path.join(os.path.dirname(__file__), pair_name), exist_ok=True)
    folder_name = pair_name + f"/{extra_info}_{timestamp}"
    os.makedirs(os.path.join(os.path.dirname(__file__), folder_name), exist_ok=True)

    config = {
        "data": data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    # support Chinese
    with open(
        os.path.join(os.path.join(os.path.dirname(__file__), folder_name, "config.json")), "w",
    ) as f:
        json.dump(config, f, ensure_ascii=False)

    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), folder_name, "log.log"))
    # handler.setLevel(logging.DEBUG)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    # handler.setFormatter(formatter)
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.addHandler(stream_handler)

    logging.basicConfig(
        filename=os.path.join(os.path.dirname(__file__), folder_name, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(
            os.path.join(os.path.dirname(__file__), folder_name, method.__name__), exist_ok=True
        )

    indices = utils.get_random_indices(len(data))
    logging.info(f"Using indices: {indices} for {test_lang}2{tgt_lang} in few-shot setting.")

    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
    )
    orig_num_keys = len(lm.api_key_list)

    refine_results = []
    unchaged_num = 0
    for d in data:  # [id, src, en_trans, ref]
        logging.info(f"Running sample {data.index(d)}")
        # if budget <= 0.0:
        if lm.cost > budget:
            logging.error(
                f"Budget has been depleted, stopping. Sample {data.index(d)} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget - lm.cost}")
            # if budget <= 0.0:
            if lm.cost > budget:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            operations_graph = method()
            sim = deepcopy(similarity[test_lang])
            executor = controller.Controller(
                lm,
                operations_graph,
                RefineTranslationPrompter(),
                RefineTranslationParser(),
                {
                    "original": d,
                    "previous": "",
                    "current": "",
                    "method": method.__name__,
                    "data_path": data_path,
                    "results_path": os.path.join(os.path.dirname(__file__), folder_name),
                    "prompt_path": os.path.join(os.path.dirname(__file__), folder_name, method.__name__),
                    "src_path": src_path,
                    "trans_path": trans_path,
                    "ref_path": ref_path,
                    "indices": indices,
                    "language": test_lang,
                    "similarity": sim,
                    "lang_map": lang_map,
                    "aux_lang": [],
                    "bleurt": {
                        "model": bleurt_model,
                        "tokenizer": bleurt_tokenizer,
                        "config": bleurt_config,
                        "weight": 0.2,
                    },
                    "comet": {
                        "model": comet_model,
                        "weight": 0.2,
                    },
                    "sacrebleu": {
                        "weight": 0.6,
                    },
                    "score": 0.0,
                    "pseudo": False,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            # path = os.path.join(
            #     os.path.dirname(__file__),
            #     folder_name,
            #     method.__name__,
            #     f"{data.index(d)}.json",
            # )
            # executor.output_graph(path)
            # budget -= lm.cost
            try:
                refine_results.append(executor.graph.leaves[-1].thoughts[0].state["current"][2])
                if executor.graph.leaves[-1].thoughts[0].state["current"][2] == d[2]:
                    unchaged_num += 1
            except:
                refine_results.append(d[2])
                unchaged_num += 1
    unchaged_ratio = unchaged_num / len(data)
    logging.info(f"Unrefined data ratio: {unchaged_ratio}")
    got_refine_hyp = os.path.join(os.path.dirname(__file__), folder_name, 'got_refine')
    got_refine_eval = os.path.join(os.path.dirname(__file__), folder_name, 'metrics.json')
    with open(got_refine_hyp, 'w', encoding='utf8') as f:
        for result in refine_results:
            f.write(result + '\n')
    try:
        utils.evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, refine_results, data, got_refine_eval)
    except Exception as e:
        logging.error(f"Exception: {e}")
        logging.warning(f"Failed to evaluate got refine results.")

    # check keys
    if len(lm.api_key_list) != orig_num_keys:
        logging.warning(
            f"Number of keys has changed from {orig_num_keys} to {len(lm.api_key_list)}.")
        with open(os.path.join(os.path.dirname(__file__), folder_name, method.__name__, 'new_key_list.txt'), 'w', encoding='utf8') as f:
            f.write("[\n")
            for key in lm.api_key_list:
                f.write(key + '\n')
            f.write("]\n")


    return orig_budget - lm.cost


if __name__ == "__main__":
    """
    Input (x)   : an unordered list of 32 numbers between 0 and 9 (inclusive)
    Output (y)  : a sorted list of 32 numbers between 0 and 9 (inclusive)
    Correct     : y == sorted(x)
    Input Example:
        [0, 1, 9, 4, 2, 2, 0, 5, 1...]
    Output Example:
        [0, 0, 0, 0, 1, 1, 1, 1, 2...]
    """
    # budget = 5.0 * 84 - 13.73317 - 9.29297 - 18.1199 - 2.115021 - 1.817194 - 2.00988
    # gu: 13.73317
    # kk一次的失败，没记录
    # kk: 9.29297  0.4:0.4:0.2
    # lv: 18.1199  0.1:0.1:0.8
    # ne: 一次失败
    # ne: 1.817194  44.96% unrefined  0.2:0.2:0.6
    # si: 2.00988  16.90% unrefined 4.5:1:4.5

    # budget = 1000-9.74574-283.85823
    budget = 120

    # test_langs = ["et", "gu", "kk", "lv", "ne", "si"]
    # test_langs = ["ne"]
    test_langs = ["si"]
    train_langs = ["de", "es", "fi", "hi", "ru", "zh"]
    lang_map = {"et": "Estonian", "gu": "Gujarati", "kk": "Kazakh", "lv": "Latvian", "ne": "Nepali", "si": "Sinhala",
                "de": "German", "es": "Spanish", "fi": "Finnish", "hi": "Hindi", "ru": "Russian", "zh": "Chinese"
                }
    tgt = "en"
    # samples = [item for item in range(0, 100)]
    approaches = [sample_got]
    # approaches = [got]
    similarity = torch.load("similarity_genres.pt", map_location=torch.device('cpu'))
    sorted_similarity = {}
    for test_lang in test_langs:
        to_tgt_sim = []
        for train_lang in train_langs:
            if test_lang == train_lang:
                continue
            to_tgt_sim.append(tuple([train_lang, similarity[f"{test_lang}-{train_lang}"]["cos_sim"]]))
        to_tgt_sim = sorted(to_tgt_sim, key=lambda x: x[1], reverse=True)
        sorted_similarity[f"{test_lang}"] = to_tgt_sim
    # {"src-tgt": {"src": tensor, "tgt": tensor, "cos_sim": float}}
    for test_lang in test_langs:
        # spent = run(test_lang, tgt, sorted_similarity, lang_map, approaches, budget, "chatgpt4")
        spent = multi_threads_train(test_lang, tgt, sorted_similarity, lang_map, approaches, budget, "chatgpt-16k-super",
                                    threads_num=10, lr=math.pow(0.99, 300), pseudo=True,
                                    aux_probs_from_file="/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/si2en/chatgpt-16k-super_sample_got_2023-11-10_06-50-39")
        budget -= spent

    logging.info(f"Spent {spent} out of {budget} budget.")

    # utils.evaluate_from_file('chatgpt_got_2023-10-22_20-05-50', 'kk')
