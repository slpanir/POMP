#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain
from collections import OrderedDict

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig
from copy import deepcopy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import os
import heapq
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


def cal_similarity_wo_en(src_list, tgt_list, similarity):
    for l1 in src_list:
        for l2 in tgt_list:
            if l1 != l2:
                sim = {}
                src = l1
                tgt = l2
                sim[src] = similarity[src + '-en'][src]
                sim[tgt] = similarity[tgt + '-en'][tgt]
                sim[src + '_wo_lang_proj'] = similarity[src + '-en'][src + '_wo_lang_proj']
                sim[tgt + '_wo_lang_proj'] = similarity[tgt + '-en'][tgt + '_wo_lang_proj']
                sim['cos_sim'] = torch.cosine_similarity(sim[src], sim[tgt], dim=0).item()
                sim['cos_sim_wo_lang_proj'] = torch.cosine_similarity(sim[src + '_wo_lang_proj'], sim[tgt + '_wo_lang_proj'], dim=0).item()
                similarity[src+'-'+tgt] = deepcopy(sim)
    return similarity

def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
            not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
            cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

@torch.no_grad()
def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)
    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # 覆盖参数lang_pairs的输入，使用test_langs和train_langs构成
    test_langs = cfg.task.test_langs.split(',')
    train_langs = cfg.task.langs
    all_pairs = {}
    for s in test_langs + train_langs:
        for t in train_langs:
            if s in test_langs:
                all_pairs[s + '-' + t] = 'test'
            elif s != t and s != 'en':
                all_pairs[s + '-' + t] = 'train'
    cfg.task.lang_pairs = list(all_pairs.keys())

    # Load dataset splits
    # task = tasks.setup_task(cfg.task)
    task = tasks.setup_task(cfg)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # # 初始化分布式环境
    # dist.init_process_group(backend='nccl')

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # # 创建DistributedDataParallel包装器
    # models = [DistributedDataParallel(model) for model in models]

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight, "data_dir": cfg.task.data}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # dict caculate similarity
    similarity = {}
    # Load dataset (possibly sharded)
    # 为每一个语种建一个迭代器
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    langpair_type = ast.literal_eval(task.args.langpair_type)

    for pair in all_pairs:
        # pair = 'de-es'
        data_type = all_pairs[pair]
        cfg.dataset.gen_subset = pair
        task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task, data_type=data_type)

        itr = task.get_batch_iterator(
            dataset=task.dataset(cfg.dataset.gen_subset),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(), *[m.max_positions() for m in models]
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=cfg.distributed_training.distributed_world_size,
            shard_id=cfg.distributed_training.distributed_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
            # sampler=sampler,  # 使用DistributedSampler
        ).next_epoch_itr(shuffle=True)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        # Initialize generator
        gen_timer = StopwatchMeter()

        num_sentences = 0
        total_src_encoder_out = 0
        total_tgt_encoder_out = 0
        total_src_encoder_out_wo_lang_proj = 0
        total_tgt_encoder_out_wo_lang_proj = 0
        wps_meter = TimeMeter()
        encoder_out_list = []
        sim = {}
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if cfg.generation.prefix_size > 0:
                prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

            constraints = None
            if "constraints" in sample:
                constraints = sample["constraints"]

            gen_timer.start()
            encoder_out = task.encoder_output_step(
                generator,
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )
            total_src_encoder_out += encoder_out[0]['encoder_out'][0].mean(dim=0).sum(dim=0)
            total_tgt_encoder_out += encoder_out[0]['tgt_encoder_out'][0].mean(dim=0).sum(dim=0)
            total_src_encoder_out_wo_lang_proj += encoder_out[0]['encoder_out_wo_lang_proj'][0].mean(dim=0).sum(dim=0)
            total_tgt_encoder_out_wo_lang_proj += encoder_out[0]['tgt_encoder_out_wo_lang_proj'][0].mean(dim=0).sum(dim=0)

            # move to cpu for similarity calculation
            # encoder_out = utils.move_to_cpu(encoder_out)
            # encoder_out_list.extend(encoder_out)
            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )
            wps_meter.update(num_sentences)
            progress.log({"wps": round(wps_meter.avg)})
            # 在这里删除不再需要的变量
            del sample
            del encoder_out

            # 释放未使用的显存
            if use_cuda:
                torch.cuda.empty_cache()

            # 数据集本身已经进行了采样，减少了训练集的样本数量，这里就不需要限制了
            # if num_sentences >= task.args.num_samples:
            #     break
        # Calculate the average of all 'encoder_out' representations in encoder_out_list
        # The output shape is sqe_len * batch_size * hidden_dim
        # avg_src_encoder_out = torch.cat(
        #     [item['encoder_out'][0].mean(dim=0) for item in encoder_out_list], dim=0).mean(
        #     dim=0)
        # avg_tgt_encoder_out = torch.cat(
        #     [item['tgt_encoder_out'][0].mean(dim=0) for item in encoder_out_list], dim=0).mean(
        #     dim=0)
        # avg_src_encoder_out_wo_lang_proj = torch.cat(
        #     [item['encoder_out_wo_lang_proj'][0].mean(dim=0) for item in encoder_out_list], dim=0).mean(
        #     dim=0)
        # avg_tgt_encoder_out_wo_lang_proj = torch.cat(
        #     [item['tgt_encoder_out_wo_lang_proj'][0].mean(dim=0) for item in encoder_out_list], dim=0).mean(
        #     dim=0)
        # 计算平均的encoder输出，tensor除法

        avg_src_encoder_out = total_src_encoder_out / num_sentences
        avg_tgt_encoder_out = total_tgt_encoder_out / num_sentences
        avg_src_encoder_out_wo_lang_proj = total_src_encoder_out_wo_lang_proj / num_sentences
        avg_tgt_encoder_out_wo_lang_proj = total_tgt_encoder_out_wo_lang_proj / num_sentences

        # Calculate the similarity between source and target
        src = pair.split('-')[0]
        tgt = pair.split('-')[1]
        sim[src] = avg_src_encoder_out
        sim[tgt] = avg_tgt_encoder_out
        sim[src+'_wo_lang_proj'] = avg_src_encoder_out_wo_lang_proj
        sim[tgt+'_wo_lang_proj'] = avg_tgt_encoder_out_wo_lang_proj
        sim['cos_sim'] = torch.cosine_similarity(avg_src_encoder_out, avg_tgt_encoder_out, dim=0).item()
        sim['cos_sim_wo_lang_proj'] = torch.cosine_similarity(avg_src_encoder_out_wo_lang_proj, avg_tgt_encoder_out_wo_lang_proj, dim=0).item()
        similarity[pair] = deepcopy(sim)

        del task.datasets[pair]
        # 在这里删除不再需要的变量
        # del encoder_out_list
        del sim

        del avg_src_encoder_out
        del avg_tgt_encoder_out
        del avg_src_encoder_out_wo_lang_proj
        del avg_tgt_encoder_out_wo_lang_proj
        # 释放未使用的显存
        if use_cuda:
            torch.cuda.empty_cache()

    # 没有双语预料的利用翻译模型进行翻译后再计算，不再复用单边的语料，消除语义的bias，以下部分先注释了
    # task.langs.remove('en')
    # test_src = []
    # for key in langpair_type.keys():
    #     if langpair_type[key] == 'test':
    #         test_src.append(key.split('-')[0])
    # similarity = cal_similarity_wo_en(test_src, task.langs, similarity)
    # similarity = cal_similarity_wo_en(task.langs, task.langs, similarity)

    # torch.save(similarity, f'similarity_{task.args.num_samples}.pt')
    torch.save(similarity, f'similarity_genres.pt')
    # only print pair and similarity
    for pair, sim in similarity.items():
        print(pair + ": " + str(sim['cos_sim']))

def read_similarity(file):
    similarity = torch.load(file)
    return similarity

def construct_graph(similarity, target='en'):
    edges = {}
    for pair, sim in similarity.items():
        src = pair.split('-')[0]
        tgt = pair.split('-')[1]
        edges.setdefault(src, {})[tgt] = sim['distance']
    # set en to other languages similarity
    exits_lang = deepcopy(list(edges.keys()))
    for lang in exits_lang:
        edges.setdefault('en', {})[lang] = edges[lang]['en']
    # set edges of target to infinity
    for lang in edges[target].keys():
        edges[target][lang] = float('infinity')
    return edges

def similarity_to_distance(similarity):
    for pair, sim in similarity.items():
        src = pair.split('-')[0]
        tgt = pair.split('-')[1]
        src_pre = sim[src] # 1024
        tgt_pre = sim[tgt] # 1024
        # calculate distance of 1024-dim vector
        # sim['distance'] = np.exp(torch.dist(src_pre, tgt_pre, p=2).item()) + np.exp(1 / (sim['cos_sim'] + 0.01))
        sim['distance'] = np.exp(torch.dist(src_pre, tgt_pre, p=2).item() + 1 / (sim['cos_sim'] + 0.01))
    torch.save(similarity, 'similarity_distance_2000.pt')

        # sim['distance'] = np.exp(1 / (sim['cos_sim'] + 0.01))
    return similarity

def translation_step_cost():
    return 0.5

def dijkstra(graph, start):
    queue = [(0, start, [])]
    shortest_paths = {start: (0, [])}
    while queue:
        (dist, current_vertex, path) = heapq.heappop(queue)
        path = path + [current_vertex]
        for neighbor, neighbor_dist in graph[current_vertex].items():
            old_dist, old_path = shortest_paths.get(neighbor, (float('inf'), []))
            new_dist = dist + neighbor_dist
            if new_dist < old_dist:
                shortest_paths[neighbor] = (new_dist, path)
                heapq.heappush(queue, (new_dist, neighbor, path))
    return shortest_paths

def yen_k_shortest_paths(graph, start, end, K=1):
    A = [dijkstra(graph, start)[end]]
    B = []

    for k in range(1, K):
        for dist, path in A:
            for i in range(len(path) - 1):
                spur_node = path[i]
                root_path = path[:i]

                edges_removed = []
                for _, p in A:
                    if p[:i] == root_path and spur_node in graph[p[i-1]]:
                        edges_removed.append((p[i-1], spur_node, graph[p[i-1]][spur_node]))
                        del graph[p[i-1]][spur_node]

                spur_path = dijkstra(graph, spur_node)

                if end in spur_path:
                    total_path = root_path + spur_path[end][1]
                    total_dist = spur_path[end][0]
                    potential_k = (total_dist, total_path)
                    if potential_k not in B and potential_k not in A:
                        B.append(potential_k)

                for edge in edges_removed:
                    if edge[0] not in graph:
                        graph[edge[0]] = {}
                    graph[edge[0]][edge[1]] = edge[2]

        if B:
            B = sorted(B, key=lambda element: element[0])
            min_path = B.pop(0)
            A.append(min_path)
        else:
            break

    return A

def dijkstra_shortest_paths(start, target, vertices, edges, n):
    queue = [(0, [start])]
    distances = {vertex: float('infinity') for vertex in vertices}
    distances[start] = 0
    paths = []

    while queue and len(paths) < n:
        (distance, path) = heapq.heappop(queue)
        current_vertex = path[-1]

        if current_vertex == target:
            paths.append((distance, path))
        # keep n shortest paths
        for neighbor, edge_distance in edges[current_vertex].items():
            new_distance = distances[current_vertex] + edge_distance
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, path + [neighbor]))

    return paths

def compute_shortest_translation_paths(train_languages, test_languages, edges, n=1):
    results = {}
    for test_lang in test_languages:
        shortest = dijkstra_shortest_paths(test_lang, 'en', train_languages + test_languages + ['en'], edges, n)
        # shortest = yen_k_shortest_paths(edges, test_lang, 'en', n)
        results[test_lang] = shortest

    return results

def get_shortest_path(sim_pt):
    similarity = read_similarity(sim_pt)
    similarity = similarity_to_distance(similarity)
    edges = construct_graph(similarity, 'en')
    train_languages = ['de', 'es', 'fi', 'ru', 'hi', 'zh']
    test_languages = ['et', 'gu', 'kk', 'lv', 'si', 'ne']
    results = compute_shortest_translation_paths(train_languages, test_languages, edges, n=1)
    for test_lang in test_languages:
        print(test_lang)
        for path in results[test_lang]:
            print(path)

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--encoder-out-only', action='store_true', help='only need encoder output to calculate similarity')
    parser.add_argument('--langpair-type', type=str, default=None, help='data type of language pair')
    parser.add_argument('--num-samples', type=int, default=20000, help='number of samples to calculate similarity')
    parser.add_argument('--test-langs', type=str, default=None, help='test languages')
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == "__main__":
    # cli_main()

    # similarity = read_similarity('similarity_2000.pt')
    # for pair, sim in similarity.items():
    #     print(pair + ": " + str(sim['cos_sim']))

    get_shortest_path('./similarity_genres.pt')
