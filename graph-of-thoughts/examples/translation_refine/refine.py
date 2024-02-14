# coding: utf-8
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
import random

import numpy as np
import math
from copy import deepcopy
from typing import Dict, List, Callable, Union, Tuple, Any
from collections import OrderedDict
import threading
import matplotlib.pyplot as plt

import torch
from comet import download_model, load_from_checkpoint
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from fairseq.models.transformer import TransformerModel

from graph_of_thoughts import controller, operations, prompter, parser
from graph_of_thoughts.operations import GraphOfOperations
from graph_of_thoughts.operations.operations import KeepBestN, Generate, Aggregate, Score, Operation
from graph_of_thoughts.operations.thought import Thought
from remote_score import send_scoring_request
from back_translate import bt
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
    def generate_prompt_bt(self, **kwargs) -> str:
        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        indices = kwargs["indices"]

        prompt_path = os.path.join(prompt_path, f"bt_prompt.txt")

        # Extract sentences from the files based on the selected indices
        # few shot still pseudo
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        # ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]
        ref_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]

        # Construct the prompt
        LANG = lang_map[language]
        prompt = ""
        for s, r in zip(src_sentences, ref_sentences):
            prompt += f"<English source>: {r}\n<{LANG} translation>: {s}\n\n"

        src = "{src_text}"
        prompt += f"<English source>: {src}\n<{LANG} translation>:"

        with kwargs["lock"]:
            if kwargs["current"]:
                prompt = prompt.format(src_text=kwargs["current"][2])  # [id, src, trans, ref]
            else:
                prompt = prompt.format(src_text=kwargs["original"][2])
        # if prompt.txt does not exist, create it
            if not os.path.exists(prompt_path):
                with open(prompt_path, 'w', encoding='utf8') as pp:
                    pp.write(str(indices) + '\n')
                    pp.write(prompt)

        return prompt


    def generate_prompt_direct_trans(self, **kwargs) -> str:
        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        ref_path = kwargs["ref_path"]
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        indices = kwargs["indices"]

        prompt_path = os.path.join(prompt_path, f"direct_trans_prompt.txt")

        # Extract sentences from the files based on the selected indices
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]

        # Construct the prompt
        LANG = lang_map[language]
        prompt = ""
        for s, r in zip(src_sentences, ref_sentences):
            prompt += f"<{LANG} source>: {s}\n<English translation>: {r}\n\n"

        src = "{src_text}"
        prompt += f"<{LANG} source>: {src}\n<English translation>:"

        prompt = prompt.format(src_text=kwargs["original"][1])
        # if prompt.txt does not exist, create it
        if not os.path.exists(prompt_path):
            with open(prompt_path, 'w', encoding='utf8') as pp:
                pp.write(str(indices) + '\n')
                pp.write(prompt)

        return prompt


    def generate_prompt(self, **kwargs) -> str:
        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        ref_path = kwargs["ref_path"]
        # refine_path = kwargs["refine_path"]
        refine_path = kwargs.get("refine_path", None)
        hard_few_shot = kwargs.get("hard_few_shot", None)
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
        # ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]
        if refine_path:
            refine_sentences = [line.strip() for idx, line in enumerate(open(refine_path)) if idx in indices]
        else:
            refine_sentences = [line.strip() for line in hard_few_shot]

        # Construct the prompt
        LANG = lang_map[language]
        AUX_LANG = lang_map[aux_lang]
        prompt = ""
        for s, a, i, r in zip(src_sentences, aux_sentences, input_sentences, refine_sentences):
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
            with kwargs["lock"]:
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
        # assert len(state_dicts) >= 2, "Expected two states at least for aggregation prompt."

        data_path = kwargs["data_path"]
        src_path = kwargs["src_path"]
        trans_path = kwargs["trans_path"]
        ref_path = kwargs["ref_path"]
        # refine_path = kwargs["refine_path"]
        refine_path = kwargs.get("refine_path", None)
        hard_few_shot = kwargs.get("hard_few_shot", None)
        prompt_path = kwargs["prompt_path"]
        language = kwargs["language"]
        lang_map = kwargs["lang_map"]
        indices = kwargs["indices"]

        aux_langs = [lang for state in state_dicts for lang in state['aux_lang']]
        if isinstance(aux_langs[0], list):
            aux_langs = aux_langs[0]
        aux_langs = list(dict.fromkeys(aux_langs))
        prompt_path = os.path.join(prompt_path, f"add_{str('-'.join(aux_langs))}_prompt.txt")

        # Extract sentences from the files based on the selected indices
        src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
        input_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]
        # ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]
        if refine_path:
            refine_sentences = [line.strip() for idx, line in enumerate(open(refine_path)) if idx in indices]
        else:
            refine_sentences = [line.strip() for line in hard_few_shot]

        # Construct the prompt
        LANG = lang_map[language]
        prompt = ""
        aux = "{}"
        for s, i, r in zip(src_sentences, input_sentences, refine_sentences):
            prompt += f"<{LANG} source>: {s}\n{aux}<English translation>: {i}\n<Refined translation>: {r}\n\n"
        if state_dicts[0]['current'] != "":
            prompt += f"<{LANG} source>: {state_dicts[0]['original'][1]}\n{aux}<English translation>: {state_dicts[0]['current'][2]}\n<Refined translation>:"
        else:
            prompt += f"<{LANG} source>: {state_dicts[0]['original'][1]}\n{aux}<English translation>: {state_dicts[0]['original'][2]}\n<Refined translation>:"
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
            with kwargs["lock"]:
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

        # assert len(states) >= 2, "Expected two states at least for aggregation answer."
        assert len(texts) == 1, "Expected exactly one response for aggregation answer."
        new_states = []
        text = texts[0]
        try:
            new_state = states[0].copy()  # states是按照score的降序排列，取score最高的state
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

    def parse_aggregation_answer_bt(
        self, states: List[Dict], texts: List[str], texts_bt: List[str]=None
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

        # assert len(states) >= 2, "Expected two states at least for aggregation answer."
        assert len(texts) == 1, "Expected exactly one response for aggregation answer."
        new_states = []
        text = texts[0]
        if states[0]['original'][4] == '~!@#$%^&*()_+' and texts_bt:
            states[0]['original'][4] = texts_bt[0].strip().split("\n")[0]
        try:
            new_state = states[0].copy()  # states是按照score的降序排列，取score最高的state
            if states[0]['current'] != "":
                pre = states[0]['current'].copy()
                new_state['previous'] = pre
            cur = states[0]["original"].copy()
            # cur[2] = text.strip()
            cur[2] = text.strip().split("\n")[0]
            if texts_bt:
                cur[4] = texts_bt[0].strip().split("\n")[0]
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

    def parse_generate_answer_bt(self, state: Dict, texts: List[str], texts_bt: List[str]=None) -> List[Dict]:
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
        for id in range(len(texts)):
            try:
                new_state = state.copy()
                if state['original'][4] == '~!@#$%^&*()_+' and texts_bt:
                    state['original'][4] = texts_bt[id].strip().split("\n")[0]
                if state['current'] != "":
                    pre = state['current'].copy()
                    new_state['previous'] = pre
                cur = state["original"].copy()
                # cur[2] = text.strip()
                cur[2] = texts[id].strip().split("\n")[0]
                if texts_bt:
                    cur[4] = texts_bt[id].strip().split("\n")[0]
                new_state["current"] = cur
                # new_state["results"].append(cur[2])
                new_states.append(new_state)
            except Exception as e:
                logging.error(
                    f"Could not parse step answer: {texts[id]}. Encountered exception: {e}"
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
                # new_state["results"].append(cur[2])
                new_states.append(new_state)
            except Exception as e:
                logging.error(
                    f"Could not parse step answer: {text}. Encountered exception: {e}"
                )

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

def random_gen_agg_keys(gen_probability) -> Tuple[List[Any], List[Any]]:
    sample_gen_keys = np.random.choice(list(gen_probability.keys()), 3, replace=False)
    len_gen_aux = len(sample_gen_keys)
    agg_list = []
    for i in range(2, len_gen_aux + 1):
        for j in range(0, len_gen_aux - i + 1):
            agg_list.append(tuple(sample_gen_keys[j:j+i]))
    random_index = np.random.choice(len(agg_list))
    sample_agg_keys = [agg_list[random_index]]
    return sample_gen_keys, sample_agg_keys

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
            agg_probability[agg_key] = math.pow(agg_prob, 1.0 / i)

    # normalize probability
    # agg_probability = {k: v / sum(agg_probability.values()) for k, v in agg_probability.items()}
    # randomly sample keys according to probability
    sample_agg_keys = []
    for key in agg_probability.keys():
        if np.random.random() < agg_probability[key]:
            sample_agg_keys.append(key)

    return sample_gen_keys, sample_agg_keys


def sample_got_test(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    sample_agg_keys = sample_agg_keys_v2

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


def sample_got_test_v2(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    sample_agg_keys = sample_agg_keys_v2

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    gen_operations = {}
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)  # to be the reference in turn
    scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)  # bt score
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        scoreGen.add_predecessor(gen)
        operations_graph.add_operation(gen)
        gen_operations[sample_gen_keys[i]] = gen
    operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key))
            for key in agg_key:
                gen_op = gen_operations[key]
                agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

def sample_got_test_v2_aux_random(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = random_gen_agg_keys(aux_probability)

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    gen_operations = {}
    scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)  # to be the reference in turn
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)  # bt score
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        scoreGen.add_predecessor(gen)
        operations_graph.add_operation(gen)
        gen_operations[sample_gen_keys[i]] = gen
    operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key), langs=agg_key)
            for key in agg_key:
                gen_op = gen_operations[key]
                agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

def sample_got_test_v2_fixed_probs(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    sample_agg_keys = sample_agg_keys_v2

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    gen_operations = {}
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)  # to be the reference in turn
    scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)  # bt score
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        scoreGen.add_predecessor(gen)
        operations_graph.add_operation(gen)
        gen_operations[sample_gen_keys[i]] = gen
    operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key))
            for key in agg_key:
                gen_op = gen_operations[key]
                agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

def sample_got_test_v2_select_random(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    sample_agg_keys = sample_agg_keys_v2

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    gen_operations = {}
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)  # to be the reference in turn
    scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)  # bt score
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        scoreGen.add_predecessor(gen)
        operations_graph.add_operation(gen)
        gen_operations[sample_gen_keys[i]] = gen
    operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepRandomN(1, True)
    keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key))
            for key in agg_key:
                gen_op = gen_operations[key]
                agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

def sample_got_test_v2_only_gen(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    # sample_agg_keys = sample_agg_keys_v2
    sample_agg_keys = None

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    gen_operations = {}
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)  # to be the reference in turn
    scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)  # bt score
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        scoreGen.add_predecessor(gen)
        operations_graph.add_operation(gen)
        gen_operations[sample_gen_keys[i]] = gen
    operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key))
            for key in agg_key:
                gen_op = gen_operations[key]
                agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

def sample_got_test_v2_only_agg(aux_probability: Dict) -> operations.GraphOfOperations:
    # normalize probability
    # aux_probability = {k: v / sum(aux_probability.values()) for k, v in aux_probability.items()}  # no need?
    # randomly sample keys according to probability
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    # sample_agg_keys = sample_agg_keys_v2

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    # gen_operations = {}
    # # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)  # to be the reference in turn
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)  # bt score
    # for i in range(0, len_gen_aux):
    #     gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
    #     scoreGen.add_predecessor(gen)
    #     operations_graph.add_operation(gen)
    #     gen_operations[sample_gen_keys[i]] = gen
    # operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepBestN(1, True)
    # keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key), langs=agg_key)
            # for key in agg_key:
            #     gen_op = gen_operations[key]
            #     agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph

def read_from_file_got_test(got_file: str, only_gen=False, only_agg=False, select_random=False) -> operations.GraphOfOperations:
    assert not (only_gen and only_agg), "only_gen and only_agg cannot be True at the same time."
    with open(got_file, 'r') as f:
        got_json = json.load(f)
    operations_graph = operations.GraphOfOperations()
    gen_operations = {}

    if select_random:
        keepALL = operations.KeepRandomN(1, True)
    else:
        keepALL = operations.KeepBestN(1, True)

    if not only_agg:
        scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        scoreAgg = None
    else:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
    for operation in got_json:
        if operation['operation'] == 'generate' and not only_agg:
            gen = operations.Generate(1, 1, lang=operation['aux_lang'])
            scoreGen.add_predecessor(gen)
            operations_graph.add_operation(gen)
            gen_operations[operation['aux_lang']] = gen
        if operation['operation'] == 'aggregate' and not only_gen:
            if scoreAgg is None:
                scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
                # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
            agg = operations.Aggregate(num_merges=len(operation['aux_lang']), langs=operation['aux_lang'])
            if not only_agg:
                for key in operation['aux_lang']:
                    gen_op = gen_operations[key]
                    agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
    if not only_agg:
        keepALL.add_predecessor(scoreGen)
        operations_graph.add_operation(scoreGen)
    if scoreAgg and not only_gen:
        keepALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keepALL)

    return operations_graph


def prompt_direct_trans(prompt_path: str, src_path: str, ref_path: str, language: str, lang_map: dict, indices: List[int]):
    prompt_path = os.path.join(prompt_path, f"direct_trans_prompt.txt")

    # Extract sentences from the files based on the selected indices
    src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
    ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]

    # Construct the prompt
    LANG = lang_map[language]
    prompt = ""
    for s, r in zip(src_sentences, ref_sentences):
        prompt += f"<{LANG} source>: {s}\n<English translation>: {r}\n\n"

    src = "{src_text}"
    prompt += f"<{LANG} source>: {src}\n<English translation>:"

    # if prompt.txt does not exist, create it
    if not os.path.exists(prompt_path):
        with open(prompt_path, 'w', encoding='utf8') as pp:
            pp.write(str(indices) + '\n')
            pp.write(prompt)
    return prompt


def prompt_direct_refine(prompt_path: str, src_path: str, trans_path: str, ref_path: str, language: str, lang_map: dict, indices: List[int]):
    prompt_path = os.path.join(prompt_path, f"direct_refine_prompt.txt")

    # Extract sentences from the files based on the selected indices
    src_sentences = [line.strip() for idx, line in enumerate(open(src_path)) if idx in indices]
    input_sentences = [line.strip() for idx, line in enumerate(open(trans_path)) if idx in indices]
    # ref_sentences = [line.strip() for idx, line in enumerate(open(ref_path)) if idx in indices]
    ref_sentences = [
        "Civil rights group warns against travel to Missouri",
        "On December 7, 2014, 11 children were singing Christmas carols around a fire in Enarotal when two Indonesian soldiers on a motorcycle appeared out of the darkness.",
        "I am never far from my creations and I am not afraid of work.",
        "Of course, the old pros go there with their discs, but I haven't gotten to play there yet.",
    ]  # temp for et, from google translate

    # Construct the prompt
    LANG = lang_map[language]
    prompt = ""
    for s, i, r in zip(src_sentences, input_sentences, ref_sentences):
        prompt += f"<{LANG} source>: {s}\n<English translation>: {i}\n<Refined translation>: {r}\n\n"
    # src = "{" + "{language}_text".format(language=language.lower()) + "}"
    src = "{src_text}"
    en = "{english_translation}"
    prompt += f"<{LANG} source>: {src}\n<English translation>: {en}\n<Refined translation>:"

    # if prompt.txt does not exist, create it
    if not os.path.exists(prompt_path):
        with open(prompt_path, 'w', encoding='utf8') as pp:
            pp.write(str(indices) + '\n')
            pp.write(prompt)

    return prompt


def direct_trans_got() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()
    gen = operations.Generate(1, 1)
    keep_best = operations.KeepBestN(1, True)
    keep_best.add_predecessor(gen)
    operations_graph.add_operation(gen)
    operations_graph.add_operation(keep_best)

    return operations_graph

def direct_refine_got() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()
    gen = operations.Generate(1, 1)
    keep_best = operations.KeepBestN(1, True)
    keep_best.add_predecessor(gen)
    operations_graph.add_operation(gen)
    operations_graph.add_operation(keep_best)

    return operations_graph

def sample_got_v2(aux_probability: Dict) -> operations.GraphOfOperations:
    sample_gen_keys, sample_agg_keys = sample_gen_agg_keys(aux_probability)
    sample_agg_keys_v2 = []

    # agg no more than 3
    for key in sample_agg_keys:
        if len(key) <= 3:
            sample_agg_keys_v2.append(key)
    sample_agg_keys = sample_agg_keys_v2

    operations_graph = operations.GraphOfOperations()
    len_gen_aux = len(sample_gen_keys)

    gen_operations = {}
    # scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v3)
    scoreGen = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
    for i in range(0, len_gen_aux):
        gen = operations.Generate(1, 1, lang=sample_gen_keys[i])
        scoreGen.add_predecessor(gen)
        operations_graph.add_operation(gen)
        gen_operations[sample_gen_keys[i]] = gen
    operations_graph.add_operation(scoreGen)

    keep_bestALL = operations.KeepBestN(1, True)
    keep_bestALL.add_predecessor(scoreGen)
    if sample_agg_keys:
        # scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v3)
        scoreAgg = operations.Score(1, True, utils.combined_score_auto_metric_v2_test_only_bleurt)
        for agg_key in sample_agg_keys:
            agg = operations.Aggregate(num_merges=len(agg_key), langs=agg_key)
            for key in agg_key:
                gen_op = gen_operations[key]
                agg.add_predecessor(gen_op)
            scoreAgg.add_predecessor(agg)
            operations_graph.add_operation(agg)
        keep_bestALL.add_predecessor(scoreAgg)
        operations_graph.add_operation(scoreAgg)
    operations_graph.add_operation(keep_bestALL)

    return operations_graph



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
        score = operations.Score(1, False, utils.auto_metric_v3)
        score.add_predecessor(gen)
        operations_graph.add_operation(gen)
        operations_graph.add_operation(score)
        score_operations[sample_gen_keys[i]] = score

    scoreALL = operations.Score(1, True, utils.combined_score_auto_metric_v3)
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

def multi_threads_train(
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
    hard_indices: List[int] = None,
    hard_few_shot: List[str] = None,
    output_from_file: str = None,
) -> str:

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

    # comet_model_path = '/home/slpan/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
    # comet_model = load_from_checkpoint(comet_model_path).eval().to(device)

    extra_bt = {
        "back_translate": True,
        "return_models": True,
        "models": None,
        "saved_cfg": None,
    }
    _, sixtp_models, saved_cfg = bt(
        "/mnt/e/unmt/acl22-sixtp/models/x2x/sentencepiece.bpe.model",
        "an example for loading model",
        src_lang=train_lang,
        tgt_lang=tgt_lang,
        model_path="/mnt/e/unmt/acl22-sixtp/models/x2x/x2x.pt",
        dict_path="/mnt/e/unmt/acl22-sixtp/models/x2x/dict.txt",
        extra_bt=extra_bt,
    )
    extra_bt = {
        "back_translate": True,
        "return_models": False,
        "models": sixtp_models,
        "saved_cfg": saved_cfg,
    }

    # nmt_model_path = '/mnt/e/unmt/acl22-sixtp/models/x2x'
    # nmt_model = TransformerModel.from_pretrained(
    #     nmt_model_path,
    #     checkpoint_file='x2x.pt',
    #     bpe='sentencepiece',
    #     bpe_codes=os.path.join(nmt_model_path, 'sentencepiece.bpe.model'),
    #     fixed_dictionary=os.path.join(nmt_model_path, 'dict.txt'),
    #     device=device,
    # ).eval().to(device)

    data_path = os.path.join(os.path.dirname(__file__), "data", "train-set4test-langs", f"{train_lang}2en")
    src_path = os.path.join(data_path, "src")
    trans_path = os.path.join(data_path, "hyp")
    ref_path = os.path.join(data_path, "ref")
    refine_path = None

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
                data.append([i, src_line, trans_line, trans_line, "~!@#$%^&*()_+"])  # last one for bt, bad enough first
                gt4pseudo.append([ref_line])
            else:
                data.append([i, src_line, trans_line, ref_line, "~!@#$%^&*()_+"])  # last one for bt, bad enough first

    # 如果文件行数不匹配，zip会停止在最短文件的末尾，所以如果你需要确保所有文件具有相同的行数，
    # 你应该在循环结束后检查文件是否已经到达末尾

    data = data[0:300]
    gt4pseudo = gt4pseudo[0:300]

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

    if hard_few_shot:
        indices = hard_indices
    else:
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
    last_graph = operations.GraphOfOperations()

    state = {
                "original": "",
                "previous": "",
                "current": "",
                "other_results": [],
                "method": method.__name__,
                "data_path": data_path,
                "results_path": os.path.join(os.path.dirname(__file__), folder_name),
                "prompt_path": os.path.join(os.path.dirname(__file__), folder_name, method.__name__),
                "src_path": src_path,
                "trans_path": trans_path,
                "ref_path": ref_path,
                "refine_path": refine_path,
                "indices": indices,
                "hard_few_shot": hard_few_shot,
                "language": train_lang,
                "tgt_lang": tgt_lang,
                # "similarity": sim,
                "lang_map": lang_map,
                "aux_lang": [],
                # "nmt": {
                #     "model": nmt_model,
                #     "weight": 0.0,
                # },
                "bleurt": {
                    "model": bleurt_model,
                    "tokenizer": bleurt_tokenizer,
                    "config": bleurt_config,
                    "weight": 0.5,
                },
                "comet": {
                    "weight": 0.5,
                },
                "sacrebleu": {
                    "weight": 0.0,
                },
                "score": 0.0,
                "pseudo": pseudo,
                "lock": lock,
                "extra_bt": extra_bt,
            }
    refine_results = OrderedDict()
    if output_from_file:
        for i in range(len(data)):
            if os.path.exists(os.path.join(output_from_file, f'{i}')):
                with open(os.path.join(output_from_file, f'{i}'), 'r', encoding='utf8') as f:
                    temp_res = json.load(f)
                    refine_results[i] = temp_res['output']
    aux_prob_points = {
        "de": [],
        "zh": [],
        "fi": [],
        "ru": [],
        "hi": [],
        "es": [],
    }
    lr_points = []
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
        random.shuffle(api_key_list)
        thread_state = {
            "original": deepcopy(state["original"]),
            "previous": deepcopy(state["previous"]),
            "current": deepcopy(state["current"]),
            "other_results": deepcopy(state["other_results"]),
            "method": state["method"],
            "data_path": state["data_path"],
            "results_path": state["results_path"],
            "prompt_path": state["prompt_path"],
            "src_path": state["src_path"],
            "trans_path": state["trans_path"],
            "ref_path": state["ref_path"],
            "refine_path": state["refine_path"],
            "indices": state["indices"],
            "hard_few_shot": state["hard_few_shot"],
            "language": state["language"],
            "tgt_lang": state["tgt_lang"],
            # "similarity": deepcopy(state["similarity"]),
            "lang_map": state["lang_map"],
            "aux_lang": deepcopy(state["aux_lang"]),
            "bleurt": state["bleurt"],
            "comet": state["comet"],
            "sacrebleu": state["sacrebleu"],
            "score": deepcopy(state["score"]),
            "pseudo": state["pseudo"],
            "lock": state["lock"],
            "extra_bt": state["extra_bt"],
        }
        t = threading.Thread(target=train, args=(lm_name, api_key_list, start_index, end_index, data, method,
                                                 aux_probability, thread_state, refine_results, unchanged_num,
                                                 lock, total_api_key_num, total_cost, budget, lr, aux_prob_points, lr_points, last_graph))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # save last_graph
    utils.output_sample_graph(last_graph, os.path.join(os.path.dirname(__file__), folder_name, 'last_graph.json'))

    # save aux_probability
    logging.info(aux_probability)
    got_refine_probability = os.path.join(os.path.dirname(__file__), folder_name, f'{train_lang}_probs.json')
    got_refine_probability_json = json.dumps(aux_probability, indent=2)
    f = open(got_refine_probability, 'w', encoding='utf8')
    f.write(got_refine_probability_json)
    f.close()

    refine_outputs = [refine_results[i] for i in sorted(refine_results.keys())]
    logging.info(f"total number of refine outputs: {len(refine_outputs)}")
    got_refine_hyp = os.path.join(os.path.dirname(__file__), folder_name, 'train_got_refine')
    got_refine_eval = os.path.join(os.path.dirname(__file__), folder_name, 'train_metrics.json')
    with open(got_refine_hyp, 'w', encoding='utf8') as f:
        for result in refine_outputs:
            f.write(result + '\n')
    try:
        # utils.evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, refine_outputs, data, got_refine_eval, gt4pseudo, pseudo) # bleurt and comet
        utils.evaluate_got_refine_results_v2(bleurt_model, bleurt_tokenizer, refine_outputs, data,
                                          got_refine_eval, gt4pseudo, pseudo) # bleurt and xcomet
    except Exception as e:
        logging.error(f"Exception: {e}")
        logging.warning(f"Failed to evaluate got refine results.")

    logging.info(f"Unchanged ratio: {unchanged_num / len(refine_outputs)}")

    horizon = {}
    for aux in aux_prob_points.keys():
        horizon[aux] = [i for i in range(len(aux_prob_points[aux]))]

    # plot
    for aux in aux_prob_points.keys():
        plt.plot(horizon[aux], aux_prob_points[aux], label=aux)
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.title('Probability Trends over Iterations')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), folder_name, 'aux_prob_trends.png'))
    plt.close()

    plt.plot([i for i in range(len(lr_points))], lr_points, label='lr')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Trends over Iterations')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), folder_name, 'lr_trends.png'))
    plt.close()

    # check keys
    if total_api_key_num != orig_num_keys:
        logging.warning(
            f"Number of keys has changed from {orig_num_keys} to {len(lm.api_key_list)}.")
        with open(os.path.join(os.path.dirname(__file__), folder_name, method.__name__, 'new_key_list.txt'), 'w', encoding='utf8') as f:
            f.write("[\n")
            for key in lm.api_key_list:
                f.write(key + '\n')
            f.write("]\n")

    # release memory
    del bleurt_model
    del bleurt_tokenizer
    # del comet_model
    del state
    del refine_results
    del threads
    del total_api_key_num
    del aux_probability
    del data
    del gt4pseudo

    # test
    # spent = multi_threads_test(train_lang, tgt_lang, similarity, lang_map, methods, budget, lm_name, threads_num,
    #                            aux_probs_from_file=os.path.join(os.path.dirname(__file__), folder_name))

    return os.path.join(os.path.dirname(__file__), folder_name)


def train(
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
    aux_prob_points: Dict,
    lr_points: List[float],
    last_graph: operations.GraphOfOperations,
) -> None:

    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
        api_key_list=api_key_list,
    )

    item = start_index
    # orig_sim = deepcopy(state['similarity'])
    while item < end_index:  # [id, src, en_trans, ref]
        if refine_results.get(item):
            item += 1
            continue
        if len(lm.api_key_list) == 0:
            logging.warning(f"Run out of keys.")
            break
        with lock:
            operations_graph = method(aux_probability)
        state["original"] = data[item]
        state["previous"] = ""
        state["current"] = ""
        state["aux_lang"] = []
        state["score"] = 0.0
        logging.info(f"Running sample {item}")
        logging.info(f"Running method {method.__name__}")
        logging.info(f"Budget left: {budget - total_cost}")
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
        rewards = {}
        try:
            last_operation: KeepBestN = executor.graph.leaves[-1]
            scoreALL: Score = last_operation.predecessors[-1]  # the first score for gen, the second for agg
            aggregate_operations: List[Aggregate] = scoreALL.predecessors
            for aggregate_operation in aggregate_operations:
                if aggregate_operation.operation_type.name != 'aggregate':
                    continue
                sigma = 0.0
                agg_score = aggregate_operation.thoughts[0].state['score']
                agg_aux_lang = aggregate_operation.thoughts[0].state['aux_lang']
                rewards[str(agg_aux_lang)] = {}
                rewards[str(agg_aux_lang)]["score"] = agg_score
                previous_thoughts: List[Thought] = aggregate_operation.get_previous_thoughts()
                for thought in previous_thoughts:
                    sigma += (agg_score - thought.state['score'])
                with lock:
                    for thought in previous_thoughts:  # generate score thought
                        score = thought.state['score']
                        aux = thought.state['aux_lang'][0]
                        reward = (sigma - (agg_score - score)) / (len(previous_thoughts) - 1)
                        reward = lr * utils.origin_symmetric_swish(reward)
                        rewards[str(agg_aux_lang)][str(aux)] = reward
                        if abs(reward) >= 0.1:
                            logging.warning(f"Reward too large: {reward}, skip.")
                            continue
                        temp = aux_probability[aux] * (1 + reward)
                        temp = 0.5 * temp + 0.5 * aux_probability[aux]
                        temp = min(temp, 1.0)
                        aux_probability[aux] = temp
            with lock:
                for aux in aux_probability.keys():
                    aux_prob_points[aux].append(aux_probability[aux])
            # last_operation: KeepBestN = executor.graph.leaves[-1]
            # previous_thoughts: List[Thought] = last_operation.get_previous_thoughts()
            # assert all(
            #     previous_thought.scored for previous_thought in previous_thoughts
            # ), "Not all thoughts have been scored"
            # best_thought: Thought = last_operation.get_best_n()[0]
            # best_score = best_thought.score
            # # best_score = min(best_score, score_max)
            # with lock:
            #     for thought in previous_thoughts:
            #         score = thought.score
            #         aux_lang = thought.state['aux_lang']
            #         scores[str(aux_lang)] = score
            #         for aux in aux_lang:
            #             # temp = aux_probability[aux] + lr * abs(score)
            #             # temp = aux_probability[aux] * math.exp(score)
            #             if abs(score * lr) >= 0.1:
            #                 continue
            #             temp = aux_probability[aux] * (1 + max(-0.1, min(score * lr, 0.1)) / len(aux_lang))  # 防止个别超高分数的影响
            #             temp = 0.5 * temp + 0.5 * aux_probability[aux]
            #             temp = min(temp, 1.0)
            #             aux_probability[aux] = temp
                        # 1.topK
                        # 2.Aggregate 多个语种的提升： 除以语种数量的平均值
                        # 3.人为限制Agg语种数量
                        # 4.如果是这个原因，仍然未缓解，单语种提升加成，多个语种提升减成

                        # 5.分值都很低且很相似，说明表现都不好，反之表现都好，辅助语种都用，根据效果判断使用辅助语种的数量

                        # if score > 0.0:
                        #     temp = temp * 0.8 + aux_probability[aux] * 0.2
                        # else:
                        #     temp = temp * 0.2 + aux_probability[aux] * 0.8
                # for aux in aux_probability.keys():
                #     aux_prob_points[aux].append(aux_probability[aux])
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
        except Exception as e:
            # refine_results.append(d[2])
            logging.error(f"Exception: {e}")
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
            "reward": rewards,
            # "best_score": best_score,
            "lr": lr,
            # "scores": scores,
        }
        temp_res_json = json.dumps(temp_res, indent=2)
        f = open(os.path.join(state['results_path'], f'{item}'), 'w', encoding='utf8')
        f.write(temp_res_json)
        f.close()

        item += 1
        with lock:
            lr = lr * 0.9
            lr_points.append(lr)

    with lock:
        total_api_key_num += len(lm.api_key_list)
        total_cost += lm.cost
        last_graph = operations_graph
        utils.output_sample_graph(operations_graph, os.path.join(state['results_path'], f'{item}_graph.json'))


def multi_threads_test(
        test_lang: str,
        tgt_lang: str,
        similarity: dict,
        lang_map: dict,
        methods: List[Callable[..., operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
        threads_num: int = 1,
        pseudo: bool = False,
        opertions_graph_path: str = None,
        aux_probs_from_file: str = None,
        output_from_file: str = None,
        only_gen: bool = False,
        only_agg: bool = False,
        select_random: bool = False,
        fixed_probs: bool = False,
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

    # comet_model_path = '/home/slpan/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
    # comet_model = load_from_checkpoint(comet_model_path).eval().to(device)

    data_path = os.path.join(os.path.dirname(__file__), "data", "x2x", f"{test_lang}2en")
    src_path = os.path.join(data_path, "src")
    trans_path = os.path.join(data_path, "hyp")
    ref_path = os.path.join(data_path, "ref")
    refine_path = os.path.join(data_path, "llm_direct_refine")

    data = []
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

            data.append([i, src_line, trans_line, ref_line, "~!@#$%^&*()_+"])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.join(os.path.dirname(__file__), "results")))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    pair_name = f"results/{test_lang}2{tgt_lang}"
    os.makedirs(os.path.join(os.path.dirname(__file__), pair_name), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), pair_name, 'test'), exist_ok=True)
    folder_name = pair_name + f"/test/{extra_info}_{timestamp}"
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
    logging.info(f"Using indices: {indices} for {test_lang}2{tgt_lang} in few-shot setting.")

    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
    )
    orig_num_keys = len(lm.api_key_list)

    # sim = deepcopy(similarity[train_lang])
    # convert similarity to probability
    aux_probability = {}
    if aux_probs_from_file:
        aux_probs_from_file = os.path.join(aux_probs_from_file, f'{test_lang}_probs.json')
        with open(aux_probs_from_file, 'r', encoding='utf8') as f:
            aux_probability = json.load(f)
    else:
        for item in similarity[test_lang]:  # [(lang, sim),]
            k = item[0]
            v = np.exp(-1.0 + item[1])  # exp
            # v = 1 / (1 + np.exp(-1 * (item[1] - 0.5)))  # sigmoid
            aux_probability[k] = v
    method = methods[0]
    if opertions_graph_path:
        logging.info(f"Using given operations graph from {opertions_graph_path}.")
        operations_graph = read_from_file_got_test(opertions_graph_path, only_gen=only_gen, only_agg=only_agg, select_random=select_random)
    else:
        operations_graph = method(aux_probability)
    utils.output_sample_graph(operations_graph, os.path.join(os.path.dirname(__file__), folder_name, 'graph.json'))

    state = {
        "original": "",
        "previous": "",
        "current": "",
        "other_results": [],
        "method": method,
        "data_path": data_path,
        "results_path": os.path.join(os.path.dirname(__file__), folder_name),
        "prompt_path": os.path.join(os.path.dirname(__file__), folder_name, method.__name__),
        "src_path": src_path,
        "trans_path": trans_path,
        "ref_path": ref_path,
        "refine_path": refine_path,
        "indices": indices,
        "language": test_lang,
        # "similarity": sim,
        "lang_map": lang_map,
        "aux_lang": [],
        "bleurt": {
            "model": bleurt_model,
            "tokenizer": bleurt_tokenizer,
            "config": bleurt_config,
            "weight": 0.5,
        },
        "comet": {
            # "model": comet_model,
            "weight": 0.5,
        },
        "sacrebleu": {
            "weight": 0,
        },
        "score": 0.0,
        "pseudo": pseudo,
        "lock": lock,
    }
    logging.info(f"bleurt weight: {state['bleurt']['weight']}, comet weight: {state['comet']['weight']}, sacrebleu weight: {state['sacrebleu']['weight']}")
    refine_results = OrderedDict()
    if output_from_file:
        for i in range(len(data)):
            if os.path.exists(os.path.join(output_from_file, f'{i}')):
                with open(os.path.join(output_from_file, f'{i}'), 'r', encoding='utf8') as f:
                    temp_res = json.load(f)
                    refine_results[i] = temp_res['output']
    unchanged_num = 0
    threads = []
    total_api_key_num = 0
    total_cost = 0.0
    for i in range(threads_num):
        start_index = int(len(data) / threads_num * i)
        end_index = int(len(data) / threads_num * (i + 1)) if i < threads_num - 1 else len(data)
        api_key_start_index = int(len(lm.api_key_list) / threads_num * i)
        api_key_end_index = int(len(lm.api_key_list) / threads_num * (i + 1)) if i < threads_num - 1 else len(
            lm.api_key_list)
        api_key_list = lm.api_key_list[api_key_start_index:api_key_end_index]
        thread_state = {
            "original": deepcopy(state["original"]),
            "previous": deepcopy(state["previous"]),
            "current": deepcopy(state["current"]),
            "other_results": deepcopy(state["other_results"]),
            "method": state["method"],
            "data_path": state["data_path"],
            "results_path": state["results_path"],
            "prompt_path": state["prompt_path"],
            "src_path": state["src_path"],
            "trans_path": state["trans_path"],
            "ref_path": state["ref_path"],
            "refine_path": state["refine_path"],
            "indices": state["indices"],
            "language": state["language"],
            "lang_map": state["lang_map"],
            "aux_lang": deepcopy(state["aux_lang"]),
            "bleurt": state["bleurt"],
            "comet": state["comet"],
            "sacrebleu": state["sacrebleu"],
            "score": deepcopy(state["score"]),
            "pseudo": state["pseudo"],
            "lock": state["lock"],
        }
        thread_ops_graph = deepcopy(operations_graph)
        t = threading.Thread(target=test, args=(lm_name, api_key_list, start_index, end_index, data,
                                                method, aux_probability, thread_ops_graph, thread_state, refine_results, unchanged_num,
                                                 lock, total_api_key_num, total_cost, budget))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # results
    refine_outputs = [refine_results[i] for i in sorted(refine_results.keys())]
    got_refine_hyp = os.path.join(os.path.dirname(__file__), folder_name, 'test_got_refine')
    got_refine_eval = os.path.join(os.path.dirname(__file__), folder_name, 'test_metrics.json')
    with open(got_refine_hyp, 'w', encoding='utf8') as f:
        for result in refine_outputs:
            f.write(result + '\n')
    # metrics
    # try:
    #     # utils.evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, comet_model, refine_outputs, data,
    #     #                                   got_refine_eval)
    #     utils.evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, refine_outputs, data,
    #                                       got_refine_eval)
    # except Exception as e:
    #     logging.error(f"Exception: {e}")
    #     logging.warning(f"Failed to evaluate got refine results.")

    logging.info(f"Unchanged ratio: {unchanged_num / len(refine_outputs)}")

    # check keys
    if total_api_key_num != orig_num_keys:
        logging.warning(
            f"Number of keys has changed from {orig_num_keys} to {len(lm.api_key_list)}.")
        with open(os.path.join(os.path.dirname(__file__), folder_name, method.__name__, 'new_key_list.txt'), 'w',
                  encoding='utf8') as f:
            f.write("[\n")
            for key in lm.api_key_list:
                f.write(key + '\n')
            f.write("]\n")

    return orig_budget - total_cost


def test(
        lm_name: str,
        api_key_list: List[str],
        start_index: int,
        end_index: int,
        data: List[Any],
        method: Callable[..., operations.GraphOfOperations],
        aux_probability: Dict,
        operations_graph: operations.GraphOfOperations,
        state: Dict,
        refine_results: OrderedDict,
        unchanged_num: int,
        lock: threading.Lock,
        total_api_key_num: int,
        total_cost: float,
        budget: float,
) -> None:
    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
        api_key_list=api_key_list,
    )

    item = start_index
    while item < end_index:  # [id, src, en_trans, ref]
        if refine_results.get(item):
            item += 1
            continue
        if method.__name__ == 'sample_got_test_v2_aux_random':
            operations_graph = method(aux_probability)
        state["original"] = data[item]
        state["previous"] = ""
        state["current"] = ""
        state["other_results"] = []
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

        # record local results
        try:
            # refine_results.append(executor.graph.leaves[0].thoughts[0].state["current"][2])
            with lock:
                refine_results[item] = executor.graph.leaves[-1].thoughts[0].state["current"][2]
                if refine_results[item] == data[item][2] or refine_results[item] == data[item][3]:
                    unchanged_num += 1
        except Exception as e:
            logging.error(f"Exception: {e}")
            # refine_results.append(d[2])
            with lock:
                if state['pseudo']:
                    refine_results[item] = data[item][3]
                else:
                    refine_results[item] = data[item][2]
                unchanged_num += 1
        last_operation = executor.graph.leaves[-1]
        previous_thoughts = last_operation.get_previous_thoughts()
        previous_states = [thought.state for thought in previous_thoughts]
        all_gens_res = {}
        for i, state in enumerate(previous_states):
            all_gens_res[state['aux_lang'][0]] = state['current'][2]
        # save temp results
        temp_res = {
            "source": data[item][1],
            "reference": data[item][3],
            "output": refine_results[item],
            "all_gens": all_gens_res,
        }
        temp_res_json = json.dumps(temp_res, indent=2)
        f = open(os.path.join(state['results_path'], f'{item}'), 'w', encoding='utf8')
        f.write(temp_res_json)
        f.close()
        logging.info(f"Sample {item} output has been recorded.")

        executor.reset_graph()
        logging.info(f"Operation graph has been reset.")
        item += 1

    with lock:
        total_api_key_num += len(lm.api_key_list)
        total_cost += lm.cost


def multi_threads_direct_trans(
        test_lang: str,
        tgt_lang: str,
        similarity: dict,
        lang_map: dict,
        methods: List[Callable[..., operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
        threads_num: int = 1,
        pseudo: bool = False,
        opertions_graph_path: str = None,
        aux_probs_from_file: str = None,
        output_from_file: str = None,
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

    # comet_model_path = '/home/slpan/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
    # comet_model = load_from_checkpoint(comet_model_path).eval().to(device)

    data_path = os.path.join(os.path.dirname(__file__), "data", "x2x", f"{test_lang}2en")
    src_path = os.path.join(data_path, "src")
    trans_path = os.path.join(data_path, "hyp")
    ref_path = os.path.join(data_path, "ref")

    data = []
    # 使用zip函数同时迭代三个文件的每一行
    with open(src_path, "r", encoding='utf8') as f_src, \
          open(trans_path, "r", encoding='utf-8') as f_trans, \
            open(ref_path, "r", encoding='utf8') as f_ref:

        for i, (src_line, trans_line, ref_line) in enumerate(zip(f_src, f_trans, f_ref)):
            src_line = src_line.strip()
            trans_line = trans_line.strip()
            ref_line = ref_line.strip()

            # 确保源文件和参考文件的行数相同
            assert src_line and trans_line and ref_line, f"Mismatch at line {i}"

            data.append([i, src_line, trans_line, ref_line, '~!@#$%^&*()_+'])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.join(os.path.dirname(__file__), "results")))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    pair_name = f"results/{test_lang}2{tgt_lang}"
    os.makedirs(os.path.join(os.path.dirname(__file__), pair_name), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), pair_name, 'test'), exist_ok=True)
    folder_name = pair_name + f"/test/{extra_info}_{timestamp}"
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

    # indices = utils.get_random_indices(len(data))
    indices = [278, 1644, 615, 30]  # temp for et
    logging.info(f"Using indices: {indices} for {test_lang}2{tgt_lang} in few-shot setting.")

    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
    )
    orig_num_keys = len(lm.api_key_list)

    method = methods[0]
    operations_graph = method()
    if opertions_graph_path:
        logging.info(f"Using given operations graph from {opertions_graph_path}.")
        operations_graph = read_from_file_got_test(opertions_graph_path)
    utils.output_sample_graph(operations_graph, os.path.join(os.path.dirname(__file__), folder_name, 'graph.json'))

    state = {
        "original": "",
        "previous": "",
        "current": "",
        "method": method.__name__,
        "data_path": data_path,
        "results_path": os.path.join(os.path.dirname(__file__), folder_name),
        "prompt_path": os.path.join(os.path.dirname(__file__), folder_name, method.__name__),
        "prompt": "",
        "src_path": src_path,
        "trans_path": trans_path,
        "ref_path": ref_path,
        "indices": indices,
        "language": test_lang,
        # "similarity": sim,
        "lang_map": lang_map,
        "aux_lang": [],
        "score": 0.0,
        "pseudo": pseudo,
        "lock": lock,
    }

    if state['method'] == 'direct_trans_got':
        state['prompt'] = prompt_direct_trans(state['prompt_path'], state['src_path'], state['ref_path'],
                                              state['language'], state['lang_map'], state['indices'])
    elif state['method'] == 'direct_refine_got':
        state['prompt'] = prompt_direct_refine(state['prompt_path'], state['src_path'], state['trans_path'], state['ref_path'],
                                               state['language'], state['lang_map'], state['indices'])

    refine_results = OrderedDict()
    if output_from_file:
        for i in range(len(data)):
            if os.path.exists(os.path.join(output_from_file, f'{i}')):
                with open(os.path.join(output_from_file, f'{i}'), 'r', encoding='utf8') as f:
                    temp_res = json.load(f)
                    refine_results[i] = temp_res['output']
    unchanged_num = 0
    threads = []
    total_api_key_num = 0
    total_cost = 0.0
    for i in range(threads_num):
        start_index = int(len(data) / threads_num * i)
        end_index = int(len(data) / threads_num * (i + 1)) if i < threads_num - 1 else len(data)
        api_key_start_index = int(len(lm.api_key_list) / threads_num * i)
        api_key_end_index = int(len(lm.api_key_list) / threads_num * (i + 1)) if i < threads_num - 1 else len(
            lm.api_key_list)
        api_key_list = lm.api_key_list[api_key_start_index:api_key_end_index]
        thread_state = {
            "original": deepcopy(state["original"]),
            "previous": deepcopy(state["previous"]),
            "current": deepcopy(state["current"]),
            "method": state["method"],
            "data_path": state["data_path"],
            "results_path": state["results_path"],
            "prompt_path": state["prompt_path"],
            "prompt": state["prompt"],
            "src_path": state["src_path"],
            "trans_path": state["trans_path"],
            "ref_path": state["ref_path"],
            "indices": state["indices"],
            "language": state["language"],
            "lang_map": state["lang_map"],
            "aux_lang": deepcopy(state["aux_lang"]),
            "score": deepcopy(state["score"]),
            "pseudo": state["pseudo"],
            "lock": state["lock"],
        }
        thread_ops_graph = deepcopy(operations_graph)
        t = threading.Thread(target=direct_trans, args=(lm_name, api_key_list, start_index, end_index, data,
                                                method, thread_ops_graph, thread_state, refine_results, unchanged_num,
                                                 lock, total_api_key_num, total_cost, budget))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # results
    refine_outputs = [refine_results[i] for i in sorted(refine_results.keys())]
    got_refine_hyp = os.path.join(os.path.dirname(__file__), folder_name, 'test_got_refine')
    got_refine_eval = os.path.join(os.path.dirname(__file__), folder_name, 'test_metrics.json')
    with open(got_refine_hyp, 'w', encoding='utf8') as f:
        for result in refine_outputs:
            f.write(result + '\n')
    # metrics
    try:
        utils.evaluate_got_refine_results_v2(bleurt_model, bleurt_tokenizer, refine_outputs, data, got_refine_eval)
    except Exception as e:
        logging.error(f"Exception: {e}")
        logging.warning(f"Failed to evaluate got refine results.")

    logging.info(f"Unchanged ratio: {unchanged_num / len(refine_outputs)}")

    # check keys
    if total_api_key_num != orig_num_keys:
        logging.warning(
            f"Number of keys has changed from {orig_num_keys} to {len(lm.api_key_list)}.")
        with open(os.path.join(os.path.dirname(__file__), folder_name, method.__name__, 'new_key_list.txt'), 'w',
                  encoding='utf8') as f:
            f.write("[\n")
            for key in lm.api_key_list:
                f.write(key + '\n')
            f.write("]\n")

    return orig_budget - total_cost


def direct_trans(
        lm_name: str,
        api_key_list: List[str],
        start_index: int,
        end_index: int,
        data: List[Any],
        method: Callable[..., operations.GraphOfOperations],
        operations_graph: operations.GraphOfOperations,
        state: Dict,
        refine_results: OrderedDict,
        unchanged_num: int,
        lock: threading.Lock,
        total_api_key_num: int,
        total_cost: float,
        budget: float,
) -> None:
    lm = controller.ChatGPT(
        "../../graph_of_thoughts/controller/config.json",
        model_name=lm_name,
        api_key_list=api_key_list,
    )

    item = start_index
    while item < end_index:  # [id, src, en_trans, ref]
        if refine_results.get(item):
            item += 1
            continue
        state["original"] = data[item]
        state["previous"] = ""
        state["current"] = ""
        state["aux_lang"] = []
        state["score"] = 0.0
        logging.info(f"Running sample {item}")

        # for method in methods:
        logging.info(f"Running method {method.__name__}")
        logging.info(f"Budget left: {budget - total_cost}")
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

        is_failed = ""
        try:
            with lock:
                refine_results[item] = executor.graph.leaves[-1].thoughts[0].state["current"][2]
        except Exception as e:
            logging.error(f"Exception: {e}")
            logging.warning(f"Failed to get output for {item}")
            is_failed = "failed_"  # 用于区分是否是失败的样本
            with lock:
                refine_results[item] = ""

        # save temp results
        temp_res = {
            "output": refine_results[item],
        }
        temp_res_json = json.dumps(temp_res, indent=2)
        f = open(os.path.join(state['results_path'], is_failed+f'{item}'), 'w', encoding='utf8')
        f.write(temp_res_json)
        f.close()
        logging.info(f"Sample {item} output has been recorded.")

        executor.reset_graph()
        logging.info(f"Operation graph has been reset.")
        item += 1

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
        utils.evaluate_got_refine_results(bleurt_model, bleurt_tokenizer, refine_results, data, got_refine_eval)
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
    budget = 1000

    only_agg_test_langs = ["gu", "kk", "lv", "ne", "si"]
    gt_only_agg_test_langs = ["lv", "ne", "si"]
    only_gen_test_langs = ["kk", "lv", "ne", "si"]
    gt_only_gen_test_langs = ["et", "gu", "kk", "lv", "ne", "si"]
    select_random_test_langs = ["et", "gu", "kk", "lv", "ne", "si"]
    aux_random_test_langs = ["et", "gu", "kk", "lv", "ne", "si"]
    fixed_probs_test_langs = ["et", "gu", "kk", "lv", "ne", "si"]
    direct_test_langs = ["kk"]
    # test_langs = ["si"]
    test_langs = ["lv", "ne", "si"]
    all_test_langs = ["et", "gu", "kk", "lv", "ne", "si"]
    train_langs = ["de", "es", "fi", "hi", "ru", "zh"]
    lang_map = {"et": "Estonian", "gu": "Gujarati", "kk": "Kazakh", "lv": "Latvian", "ne": "Nepali", "si": "Sinhala",
                "de": "German", "es": "Spanish", "fi": "Finnish", "hi": "Hindi", "ru": "Russian", "zh": "Chinese"
                }
    tgt = "en"
    # samples = [item for item in range(0, 100)]
    train_approaches = [sample_got_v2]
    test_approaches = [sample_got_test_v2]
    test_only_gen_approaches = [sample_got_test_v2_only_gen]
    test_only_agg_approaches = [sample_got_test_v2_only_agg]
    test_select_random_approaches = [sample_got_test_v2_select_random]
    test_aux_random_approaches = [sample_got_test_v2_aux_random]
    test_fixed_probs_approaches = [sample_got_test_v2_fixed_probs]
    trans_approaches = [direct_trans_got]
    refine_approaches = [direct_refine_got]
    # approaches = [got]
    similarity = torch.load("similarity_genres.pt", map_location=torch.device('cpu'))
    sorted_similarity = {}
    for test_lang in all_test_langs:
        to_tgt_sim = []
        for train_lang in train_langs:
            if test_lang == train_lang:
                continue
            to_tgt_sim.append(tuple([train_lang, similarity[f"{test_lang}-{train_lang}"]["cos_sim"]]))
        to_tgt_sim = sorted(to_tgt_sim, key=lambda x: x[1], reverse=True)
        sorted_similarity[f"{test_lang}"] = to_tgt_sim
    # {"src-tgt": {"src": tensor, "tgt": tensor, "cos_sim": float}}
    graph_path_dict = {
        "et": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/et2en/chatgpt-super_sample_got_v2_2024-01-30_03-17-04",
        "gu": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/gu2en/chatgpt-super_sample_got_v2_2024-01-30_17-16-59",
        "kk": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/kk2en/chatgpt-super_sample_got_v2_2024-01-30_21-19-32",
        "lv": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/lv2en/chatgpt-super_sample_got_v2_2024-02-01_16-21-43",
        "ne": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/ne2en/chatgpt-super_sample_got_v2_2024-02-02_10-12-10",
        "si": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/si2en/chatgpt-super_sample_got_v2_2024-02-03_02-12-42"
    }
    test_output_from_file_dict = {
        "et": None,
        "gu": None,
        "kk": None,
        "lv": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/lv2en/test/chatgpt-super_sample_got_test_v2_2024-02-06_15-50-59",
        "ne": None,
        "si": None,
    }
    test_only_gen_output_from_file_dict = {
        "et": None,
        "gu": None,
        "kk": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/kk2en/test/chatgpt-super_sample_got_test_v2_only_gen_2024-02-06_09-55-10",
        "lv": None,
        "ne": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/ne2en/test/chatgpt-super_sample_got_test_v2_only_gen_2024-02-08_10-06-27",
        "si": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/si2en/test/chatgpt-super_sample_got_test_v2_only_gen_2024-02-08_12-32-12",
    }
    test_only_agg_output_from_file_dict = {
        "et": None,
        "gu": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/gu2en/test/chatgpt-super_sample_got_test_v2_only_agg_2024-02-06_16-15-15",
        "kk": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/kk2en/test/chatgpt-super_sample_got_test_v2_only_agg_2024-02-08_10-16-29",
        "lv": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/lv2en/test/chatgpt-super_sample_got_test_v2_only_agg_2024-02-08_12-43-49",
        "ne": None,
        "si": None,
    }
    test_select_random_output_from_file_dict = {
        "et": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/et2en/test/chatgpt-super_sample_got_test_v2_select_random_2024-02-05_13-22-02",
        "gu": None,
        "kk": None,
        "lv": None,
        "ne": None,
        "si": None,
    }
    test_fixed_probs_output_from_file_dict = {
        "et": "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/et2en/test/chatgpt-super_sample_got_test_v2_fixed_probs_2024-02-05_21-46-51",
        "gu": None,
        "kk": None,
        "lv": None,
        "ne": None,
        "si": None,
    }

    # train set in test languages
    hard_indices = [93, 397, 567, 851]  # random generate
    # translate from google as few-shot pseudo label
    et_few_shot = [
        "Degei sits on the mud floor - he has no furniture - and there he also does his daily household chores, for example washing clothes with cloudy water from the nearby swamp.",
        "When Bardina Degei cooks dinner, she does not use the stove.",
        "Laila Ali: I am inspired every day by my personal goals.",
        "\"We weren't used to wearing clothes like this,\" says Degei, referring to her colorful handwoven turban, dark blouse and bright skirt.",
    ]
    gu_few_shot = [
        "I was also happy to see her happy, but on the third morning her period started.",
        "Periods and entry into the house",
        "In 2006, the Indian Young Lawyers Association filed a Public Interest Litigation in the Supreme Court challenging the ban on the temple.",
        "My newly married sakhi told me about two weeks ago that she is preparing for Vrat before her third.",
    ]
    kk_few_shot = [
        "Meanwhile, Remainers marched through Birmingham before holding a two-hour campaign in the city centre.",
        "But there are enough reasons to reject this conclusion.",
        "However, the policies conducted in recent years did not leave room for intangible assets, unstable political situation, geopolitical conflicts, increased risk in financial markets - these can deviate the model from its direction.",
        "Let's rewrite the scenario of monetary policy",
    ]
    lv_few_shot = [
        "After that I realized that working in a team is sometimes dangerous.",
        "Dreams disappear in the sands of Rio.",
        "Olga has a higher legal education - a master's degree.",
        "\"This year we celebrate the ten-year anniversary of the \"Riga Detective Agency\".",
    ]
    ne_few_shot = [
        "This contradicts earlier reports which said that canceling the decisive election would be unconstitutional.",
        "It had a weak army and a weak navy, however they had just built four new ships just before the war.",
        "Depending on the needs of the organism, the eye has more than one structure.",
        "Hangi is often used to make traditional roast style dinners.",
    ]
    si_few_shot = [
        "This is contrary to previous reports which stated that canceling the by-election would be unconstitutional.",
        "Although four new ships were built before the outbreak of war, they had a weak army and navy.",
        "There are different forms of eyes that exist in a complex range depending on the sexual needs of the organism.",
        "Hangi is often used to cook a traditional roast style dinner.",
    ]
    hard_few_shot = {
        'et': et_few_shot,
        'gu': gu_few_shot,
        'kk': kk_few_shot,
        'lv': lv_few_shot,
        'ne': ne_few_shot,
        'si': si_few_shot,
    }


    # for test_lang in test_langs:
    # for test_lang in only_gen_test_langs:
    # for test_lang in gt_only_gen_test_langs:
    # for test_lang in only_agg_test_langs:
    # for test_lang in gt_only_agg_test_langs:
    # for test_lang in select_random_test_langs:
    # for test_lang in aux_random_test_langs:
    # for test_lang in fixed_probs_test_langs:
    # for test_lang in direct_test_langs:
    # for train_lang in train_langs:
        # spent = run(test_lang, tgt, sorted_similarity, lang_map, approaches, budget, "chatgpt4")
        # spent = multi_threads_train(test_lang, tgt, sorted_similarity, lang_map, approaches, budget, "chatgpt-16k-super",
        #                             threads_num=8, lr=1, pseudo=True,)
                                    # aux_probs_from_file=f"/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/{test_lang}2en/chatgpt-16k_sample_got_2023-11-13_17-32-04")

        #
        # folder_path = multi_threads_train('si', tgt, sorted_similarity, lang_map, train_approaches, budget, "chatgpt-super",
        #                             threads_num=10, lr=0.28242953648100017, hard_indices=hard_indices, hard_few_shot=hard_few_shot[test_lang],
        #                             aux_probs_from_file='/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/si2en/chatgpt-super_sample_got_v2_2024-02-02_19-54-09',
        #                             output_from_file='/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/si2en/chatgpt-super_sample_got_v2_2024-02-02_19-54-09')


        # last_train_graph_path = os.path.join(os.path.dirname(__file__), folder_path, 'last_graph.json')
        # kk
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_approaches, budget, "chatgpt-super",
        #                            threads_num=1, opertions_graph_path="/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/kk2en/chatgpt-16k-super_sample_got_2023-11-17_16-55-27/last_graph.json",
        #                            aux_probs_from_file="/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/kk2en/chatgpt-16k-super_sample_got_2023-11-17_16-55-27",)
                                   # output_from_file="/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/gu2en/test/chatgpt-16k_sample_got_test_2023-11-17_08-47-06")

        # test
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_approaches, budget,
        #                            "chatgpt-super",
        #                            threads_num=10,
        #                            opertions_graph_path= graph_path_dict[test_lang]+"/last_graph.json",
        #                            aux_probs_from_file=graph_path_dict[test_lang],
        #                            output_from_file=test_output_from_file_dict[test_lang])
        # budget -= spent

        # only gen
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_only_gen_approaches, budget,
        #                            "chatgpt-super",
        #                            threads_num=10,
        #                            opertions_graph_path=graph_path_dict[test_lang] + "/last_graph.json",
        #                            only_gen=True,
        #                            )
        # budget -= spent

        # only agg
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_only_agg_approaches, budget,
        #                            "chatgpt-super",
        #                            threads_num=10,
        #                            opertions_graph_path=graph_path_dict[test_lang] + "/last_graph.json",
        #                            only_agg=True,
        #                            output_from_file=test_only_agg_output_from_file_dict[test_lang],
        #                            )
        # budget -= spent

        # select random
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_select_random_approaches, budget,
        #                            "chatgpt-super",
        #                            threads_num=10,
        #                            opertions_graph_path=graph_path_dict[test_lang] + "/last_graph.json",
        #                            select_random=True,
        #                            output_from_file=test_select_random_output_from_file_dict[test_lang],
        #                            )

        # aux random
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_aux_random_approaches, budget,
        #                            "chatgpt-super",
        #                            threads_num=10,
        #                            )

        # fixed probs
        # spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_fixed_probs_approaches, budget,
        #                            "chatgpt-super",
        #                            threads_num=10,
        #                            opertions_graph_path=f"/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/{test_lang}2en/graph_from_sim" + "/last_graph.json",
        #                            output_from_file=test_fixed_probs_output_from_file_dict[test_lang],
        #                            )



        # spent = multi_threads_direct_trans(test_lang, tgt, sorted_similarity, lang_map, trans_approaches, budget,
        #                             "chatgpt-super",
        #                             threads_num=10, )
    # test_lang = "ne"
    # for test_lang in ["et", "lv"]:
    #     spent = multi_threads_test(test_lang, tgt, sorted_similarity, lang_map, test_approaches, budget,
    #                                "chatgpt-super",
    #                                threads_num=10,
    #                                opertions_graph_path= graph_path_dict[test_lang]+"/last_graph.json",
    #                                aux_probs_from_file=graph_path_dict[test_lang],)

    test_lang = "kk"
    folder_path = multi_threads_train(test_lang, tgt, sorted_similarity, lang_map, train_approaches, budget, "chatgpt-super",
                                threads_num=10, lr=1, hard_indices=hard_indices, hard_few_shot=hard_few_shot[test_lang],
                                )
    # utils.evaluate_from_file("chatgpt-super_sample_got_test_v2_2024-02-01_03-14-00", "et")
