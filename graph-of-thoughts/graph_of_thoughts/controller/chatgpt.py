# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import sys
import openai
import os
import random
import time
import re
from typing import List, Dict, Union
import requests
import concurrent.futures
from .abstract_language_model import AbstractLanguageModel



# def on_backoff_handler(details):
#     """在每次重试时被调用的函数"""
#     if details['retries'] >= 2:



class ChatGPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False, lock=None, threads: int=None, api_key_list: list=None
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatgpt'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for chatgpt, i.e. gpt-4, gpt-3.5-turbo, etc.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        self.stop: Union[str, List[str]] = self.config["stop"]
        # The account organization is the organization that is used for chatgpt.
        self.organization: str = self.config["organization"]
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION is not set")
        else:
            openai.organization = self.organization
        # The api key is the api key that is used for chatgpt. Env variable OPENAI_API_KEY takes precedence over config.
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")
        # openai.api_key = self.api_key

        self.api_key_list: list = self.config["api_key_list"]
        openai.api_key = self.api_key_list[0]
        if threads:
            assert api_key_list is not None
            self.api_key_list = api_key_list
            assert lock is not None
            self.lock = lock


    def query(self, query: str, num_responses: int = 1) -> Dict:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the OpenAI model.
        :rtype: Dict
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        try:
            total_attempts = 8
            while total_attempts > 0:
                try:
                    if num_responses == 1:
                        if total_attempts == 1:
                            # response = self.chat([{"role": "user", "content": query}], num_responses)
                            response = self.chat([{"role": "user",
                                                   "content": "Take a deep breath and work on this step by step. This is very import to my career. \n" + query}],
                                                 num_responses)
                            time.sleep(random.randint(1, 3))
                        else:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                # future = executor.submit(self.chat, [{"role": "user", "content": query}], num_responses)
                                future = executor.submit(self.chat, [{"role": "user",
                                                                      "content": "Take a deep breath and work on this step by step. This is very import to my career. \n" + query}],
                                                         num_responses)
                                try:
                                    response = future.result(timeout=60)
                                except concurrent.futures.TimeoutError:
                                    self.logger.warning("TimeoutError in chatgpt, trying again")
                                    future.cancel()
                                    time.sleep(random.randint(1, 3))
                                    total_attempts -= 1
                                    continue
                        # response = self.chat([{"role": "user", "content": query}], num_responses)
                        # time.sleep(random.randint(1, 3))
                    else:
                        response = []
                        next_try = num_responses
                        total_num_attempts = num_responses
                        while num_responses > 0 and total_num_attempts > 0:
                            try:
                                assert next_try > 0
                                res = self.chat([{"role": "user", "content": query}], next_try)
                                response.append(res)
                                num_responses -= next_try
                                next_try = min(num_responses, next_try)
                            except Exception as e:
                                next_try = (next_try + 1) // 2
                                self.logger.warning(
                                    f"Error in chatgpt: {e}, trying again with {next_try} samples"
                                )
                                time.sleep(random.randint(1, 3))
                                total_num_attempts -= 1
                    break
                except Exception as e:
                    if not 'quota' in str(e):
                        # sys.exit()

                        total_attempts -= 1
                    self.logger.warning(f"Error in chatgpt, trying again, {total_attempts} attempts left")
            if total_attempts == 0:
                raise Exception("Failed to get response from chatgpt")
        except Exception as e:
            self.logger.warning(f"Failed to get response from chatgpt, return the same as input")
            matches = re.findall(r'<English translation>: (.*?)(?=[<\n])', query)
            if matches:
                res = matches[-1].strip()
            else:
                res = None
            response = {"choices": [{"message": {"content": res}}]}

        if self.cache:
            self.respone_cache[query] = response
        return response

    # @backoff.on_exception(
    #     backoff.expo, openai.error.OpenAIError, max_time=10, max_tries=5
    # )
    def chat(self, messages: List[Dict], num_responses: int = 1) -> Dict:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: Dict
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=num_responses,
                stop=self.stop,
            )
                # openai.requestssession = None
            # response = response[0]
            self.prompt_tokens += response["usage"]["prompt_tokens"]
            self.completion_tokens += response["usage"]["completion_tokens"]
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0

            self.cost = (
                    self.prompt_token_cost * prompt_tokens_k
                    + self.response_token_cost * completion_tokens_k
            )

            self.logger.info(
                f"This is the response from chatgpt: {response}"
                f"\nThis is the cost of the response: {self.cost}"
            )
        except Exception as e:
            # openai error code 503: exit()
            time.sleep(random.randint(1, 3))
            if 'quota' in str(e):
                self.api_key_list.remove(openai.api_key)
                self.logger.warning(f"Key {openai.api_key} run out of quota, removed from api_key_list")
                # self.api_key_list.pop(0)
            if len(self.api_key_list) > 1:
                self.api_key_list.append(self.api_key_list.pop(0))
                openai.api_key = self.api_key_list[0]
                self.api_key = self.api_key_list[0]
                self.config['api_key'] = self.api_key
            # try:
            #     with self.lock:
            #         if 'quota' in str(e):
            #             self.api_key_list.pop(0)
            #         self.api_key_list.append(self.api_key_list.pop(0))
            #         openai.api_key = self.api_key_list[0]
            #         self.api_key = self.api_key_list[0]
            #         self.config['api_key'] = self.api_key
            # except:
            #     if 'quota' in str(e):
            #         self.api_key_list.pop(0)
            #     self.api_key_list.append(self.api_key_list.pop(0))
            #     openai.api_key = self.api_key_list[0]
            #     self.api_key = self.api_key_list[0]
            #     self.config['api_key'] = self.api_key
                self.logger.warning(f"Error in chatgpt: {e}\nChanging API key to {self.api_key}")
            if self.model_name in ["chatgpt", "chatgpt-16k"]:
                time.sleep(secs=60/len(self.api_key_list))
            raise e
        return response

    def get_response_texts(self, query_response: Union[List[Dict], Dict]) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the OpenAI model.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if isinstance(query_response, Dict):
            query_response = [query_response]
        return [
            choice["message"]["content"]
            for response in query_response
            for choice in response["choices"]
        ]
