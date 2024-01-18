import random
import heapq
from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from fairseq.models.transformer import TransformerModel
from fairseq import models, quantization_utils
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from fairseq.tasks import FairseqTask
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig
import logging
from tqdm import tqdm
import numpy as np
import torch
import argparse
from argparse import Namespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_encoder_output_similarity(lang_repre, en_repre, file_type):
    lang2en_similarity = {}
    for lang, repre in lang_repre.items():
        similarity = np.dot(repre, en_repre[lang]) / (np.linalg.norm(repre) * np.linalg.norm(en_repre[lang]))
        lang2en_similarity[lang] = similarity
    if file_type == 'test':
        return lang2en_similarity
    if file_type == 'train':
        lang2lang_similarity = {}
        for lang1, repre1 in lang_repre.items():
            for lang2, repre2 in lang_repre.items():
                if lang1 != lang2:
                    similarity = np.dot(repre1, repre2) / (np.linalg.norm(repre1) * np.linalg.norm(repre2))
                    lang2lang_similarity[(lang1, lang2)] = similarity
        return lang2en_similarity, lang2lang_similarity
        

def similarity_to_distance(similarity):
    return 1 / (similarity + 0.01)

def translation_step_cost():
    return 0.5

def dijkstra_shortest_path(start, target, vertices, edges):
    distances = {vertex: float('infinity') for vertex in vertices}
    distances[start] = 0
    unvisited = list(vertices)
    path = {}

    while unvisited:
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])
        if current_vertex == target:
            break
        for neighbor, cost in edges[current_vertex].items():
            new_route = distances[current_vertex] + cost
            if new_route < distances[neighbor]:
                distances[neighbor] = new_route
                path[neighbor] = current_vertex
        unvisited.remove(current_vertex)

    shortest_path = []
    while target:
        shortest_path.insert(0, target)
        target = path.get(target, None)

    return shortest_path, distances[shortest_path[-1]]

def compute_shortest_translation_path(train_languages, test_languages, lang2en_similarity, lang2lang_similarity):
    edges = {}
    for lang1 in train_languages + test_languages:
        for lang2 in train_languages + test_languages:
            if lang1 != lang2:
                if (lang1, lang2) in lang2lang_similarity:
                    similarity = lang2lang_similarity[(lang1, lang2)]
                else:
                    similarity = lang2en_similarity[lang1] * lang2en_similarity[lang2]
                distance = similarity_to_distance(similarity) + translation_step_cost()
                edges.setdefault(lang1, {})[lang2] = distance
                edges.setdefault(lang2, {})[lang1] = distance

    results = {}
    for test_lang in test_languages:
        shortest_path, distance = dijkstra_shortest_path(test_lang, 'en', train_languages + test_languages + ['en'], edges)
        results[test_lang] = (shortest_path, distance)

    return results


def get_sentence_list(args, file_type):
    bpe_pth = args.workpth
    sentence_list = {}
    en_list = {}
    langs = ['et', 'gu', 'kk', 'lv', 'si', 'ne'] if file_type == 'test' else ['de', 'es', 'fi', 'hi', 'ru', 'zh']
    for lang in langs:
        with open(f"{args.workpth}/bpe/{file_type}.{lang}-en.{lang}", 'r', encoding='utf8') as f:
            sentence_list[lang] = f.read().splitlines()
        with open(f"{args.workpth}/bpe/{file_type}.{lang}-en.en", 'r', encoding='utf8') as f:
            en_list[lang] = f.read().splitlines()
    return sentence_list, en_list


def _get_corpus_repre(sentence_list, model, tokenizer):
    corpus_repre = {}
    for lang, sentences in sentence_list.items():
        repre = []
        for sent in tqdm(sentences, desc='processing：' + lang):
            inputs = tokenizer(sent, return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # 生成句子的表示
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_state = outputs.encoder_last_hidden_state
            # 所有tokens的表示的平均值
            tokens_avg_repre = np.mean(last_hidden_state[0, :, :].cpu().numpy(), axis=0)
            # 将句子表示添加到列表中
            repre.append(tokens_avg_repre)
        corpus_repre[lang] = np.mean(repre, axis=0)
    return corpus_repre


def get_corpus_repre(args, file_type, model, tokenizer):
    sentence_list, en_list = get_sentence_list(args, file_type)
    sentence_repre = _get_corpus_repre(sentence_list, model, tokenizer)
    en_repre = _get_corpus_repre(en_list, model, tokenizer)
    return sentence_repre, en_repre


def load_model(state, task):

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
        )

    model = models.build_model(cfg.model, task)
    model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    return model, cfg

def main():
    parser = options.get_generation_parser()
    parser.add_argument('--encoder-out-only', action='store_true',
                        help='only need encoder output to calculate similarity')

    # parser = argparse.ArgumentParser()
    parser.add_argument("--workpth", required=True, default='/mnt/e/unmt/acl22-sixtp')
    parser.add_argument("--fixed-dictionary", default=None, help="path to a fixed dictionary (not checkpointed)")
    args = options.parse_args_and_arch(parser)
    if isinstance(args, Namespace):
        cfg = convert_namespace_to_omegaconf(args)

    utils.import_user_module(cfg.common)
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    task = tasks.setup_task(cfg)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.workpth + '/models/x2x/x2x.pt')
    model, saved_cfg = load_model(state, task)

    # model = TransformerModel.from_pretrained(
    #     model_name_or_path=args.workpth + '/models/x2x',
    #     checkpoint_file='x2x.pt',
    #     bpe='sentencepiece',
    #     sentencepiece_model='senrencepiece.bpe.model',
    #     xlmr_modeldir=args.workpth + '/models/xlmrL_base',
    #     enable_lang_proj=True,
    #     max_tokens=5000,
    #     eval_bleu=False,# translation_multi_simple_epoch任务下eval_bleu参数不会自动赋值
    #     fixed_dictionary=args.fixed_dictionary,
    #     )
    tokenizer = SentencepieceBPE(args.workpth + '/models/x2x/sentencepiece.bpe.model')

    # Read the text from the workpth's bpe and get representation
    test_repre, test_en_repre = get_corpus_repre(args, 'test', model, tokenizer)
    train_repre, train_en_repre = get_corpus_repre(args, 'train', model, tokenizer)

    test2en_similarity = compute_encoder_output_similarity(test_repre, test_en_repre, 'test')
    train2en_similarity, train2train_similarity = compute_encoder_output_similarity(train_repre, train_en_repre, 'trian')

    # Compute the shortest path
    results = compute_shortest_translation_path(['de', 'es', 'fi', 'hi', 'ru', 'zh'],
                                                ['et', 'gu', 'kk', 'lv', 'si', 'ne'],
                                                train2en_similarity, train2train_similarity)
    # Print the results
    for test_lang, (shortest_path, distance) in results.items():
        print(f"{test_lang} -> {' -> '.join(shortest_path)} (distance: {distance})")


    
    


if __name__ == "__main__":
    main()