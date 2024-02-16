import json
import sentencepiece as spm
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import os

# 加载SentencePiece模型
sp = spm.SentencePieceProcessor()
model_path = '/mnt/e/unmt/acl22-sixtp/models/xlmrL_base/sentencepiece.bpe.model'  # 请替换为实际路径
sp.load(model_path)

# 分词函数
def tokenize_texts(text):
    return sp.EncodeAsPieces(text)
    # return sp.encode_as_pieces(text)

# 词频统计函数
def count_tokens(token_lists):
    return Counter(token_lists)

# 计算JSD散度的函数
def calculate_jsd(counter1, counter2):
    all_tokens = list(set(counter1.keys()) | set(counter2.keys()))
    freq1 = [counter1.get(token, 0) for token in all_tokens]
    freq2 = [counter2.get(token, 0) for token in all_tokens]
    prob1 = np.array(freq1) / np.sum(freq1)
    prob2 = np.array(freq2) / np.sum(freq2)
    return jensenshannon(prob1, prob2)

# 文件读取与处理
def read_file(file_path, is_json=False):
    if is_json:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

# 创建词汇到数值的映射
def create_vocab_mapping(*counters_list):
    unique_tokens = set()
    for counters in counters_list:
        for counter in counters:
            unique_tokens.update(counter.keys())
    return {token: i for i, token in enumerate(unique_tokens)}

# 将词频计数转换为数值分布
def convert_counters_to_numeric(counters, mapping):
    numeric_distributions = []
    for counter in counters:
        # 检查counter是否为Counter对象的列表
        if isinstance(counter, list):
            # 对于包含Counter对象的列表（如gen_counters），将每个Counter对象转换为数值
            # distribution = []
            # for sub_counter in counter:
            #     distribution.extend([mapping[token]] * freq for token, freq in sub_counter.items())
            numeric_distribution = [mapping[token] for sub_counter in counter for token, freq in sub_counter.items() for _ in range(freq)]
            numeric_distributions.append(numeric_distribution)
        else:
            # 对于单个Counter对象（如refine_counters和ref_counters）
            numeric_distribution = [mapping[token] for token, freq in counter.items() for _ in range(freq)]
            numeric_distributions.append(numeric_distribution)
    return numeric_distributions


# 绘制KDE图
def plot_kde_distributions(numeric_distributions, labels):
    plt.figure(figsize=(10, 7))
    for distribution, label in zip(numeric_distributions, labels):
        sns.kdeplot(distribution, bw_adjust=0.5, label=label)
    plt.legend()
    plt.title('KDE of Token Distributions')
    # plt.xlabel('Token Numeric Mapping')
    plt.xlabel('Token frequency')
    plt.xticks([])
    plt.ylabel('Density')
    plt.yticks([])
    plt.show()

# 将数据转换为数值并绘制分布图
def plot_distributions_with_kde_numeric(counters, labels):
    fig, axs = plt.subplots(len(counters), 1, figsize=(10, 7 * len(counters)), sharex=True)
    if len(counters) == 1:
        axs = [axs]
    for ax, counter, label in zip(axs, counters, labels):
        sns.kdeplot(list(counter.elements()), ax=ax, bw_adjust=0.5, label=label)
        ax.legend()
        ax.set_title(f'KDE for {label}')
    plt.tight_layout()
    plt.show()

# 主流程
def main(samples_dir, refine_path, ref_path):
    # 获取所有样本文件路径
    # sample_files = [os.path.join(samples_dir, f"{i}") for i in range(len(os.listdir(samples_dir))) if os.path.isfile(os.path.join(samples_dir, f"{i}")]
    sample_files = []
    for i in range(len(os.listdir(samples_dir))):
        file_path = os.path.join(samples_dir, f"{i}")
        if os.path.isfile(file_path):
            sample_files.append(file_path)

    # 读取和处理gen数据
    gen_counters = []
    for file_path in sample_files:
        data = read_file(file_path, is_json=True)
        for text in data['all_gens'].values():
            tokens = tokenize_texts(text)
            counter = count_tokens(tokens)
            gen_counters.append(counter)

    # 读取和处理refine与ref数据
    refine_texts = read_file(refine_path)
    ref_texts = read_file(ref_path)
    refine_tokenized = [tokenize_texts(text) for text in refine_texts]
    ref_tokenized = [tokenize_texts(text) for text in ref_texts]
    refine_counters = count_tokens([token for sublist in refine_tokenized for token in sublist])
    ref_counters = count_tokens([token for sublist in ref_tokenized for token in sublist])

    # 计算JSD散度
    gen_jsds = [calculate_jsd(counter, ref_counters) for counter in gen_counters]
    refine_jsd = calculate_jsd(refine_counters, ref_counters)
    avg_gen_jsd = np.mean(gen_jsds)

    with open(os.path.join(samples_dir, 'jsd.json'), 'w') as f:
        json.dump({'avg_gen_jsd': avg_gen_jsd, 'refine_jsd': refine_jsd}, f, indent=4)
    print(f"Average JSD between Gen and Ref: {avg_gen_jsd}")
    print(f"JSD between Refine and Ref: {refine_jsd}")


    # 创建词汇到数值的映射
    vocab_mapping = create_vocab_mapping(gen_counters, [refine_counters], [ref_counters])

    # 将词频计数转换为数值分布
    numeric_distributions = convert_counters_to_numeric([gen_counters, refine_counters, ref_counters], vocab_mapping)

    # 绘制KDE图
    plot_kde_distributions(numeric_distributions, ['1-auxiliary', 'POMP', 'Reference'])
    # save high quality figure
    plt.savefig(os.path.join(samples_dir, 'kde.png'), dpi=300, bbox_inches='tight')

    # 可视化
    # plot_distributions_with_kde_numeric([gen_counters, refine_counters, ref_counters], ['Gen', 'Refine', 'Ref'])

langs = ["gu", "kk", "ne", "si"]
for lang in langs:
    samples_dir = f'/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/{lang}2en'
    refine_path = f'/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/{lang}2en/test_got_refine'
    ref_path = f'/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/{lang}2en/ref'
    main(samples_dir, refine_path, ref_path)
# # 示例目录和文件路径，根据实际情况替换
# samples_dir = '/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/gu2en'
# refine_path = '/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/gu2en/test_got_refine'
# ref_path = '/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/gu2en/ref'
# main(samples_dir, refine_path, ref_path)