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
model_path = '/mnt/e/unmt/acl22-sixtp/models/xlmrL_base/sentencepiece.bpe.model'  # 实际路径
sp.load(model_path)


# 分词函数
def tokenize_texts(text):
    return sp.EncodeAsPieces(text)


# 词频统计函数，这里需要将原始token转换为ID
def count_tokens(token_lists, token_to_id):
    # 检查并扁平化嵌套的列表
    if any(isinstance(t, list) for t in token_lists):
        # 如果token_lists包含嵌套列表，则扁平化这些列表
        tokens = [item for sublist in token_lists for item in sublist]
    else:
        tokens = token_lists
    # 将tokens转换为对应的token ID列表
    token_ids = [token_to_id.get(token, -1) for token in tokens]  # 使用get以处理未知token
    return Counter(token_ids)



# 创建词汇到数值的映射
def create_vocab_mapping(*texts):
    # 将所有文本的tokens合并，创建一个大的token集合
    all_tokens = set()
    for text_group in texts:
        for text in text_group:
            all_tokens.update(text)
    # 为每个唯一token分配一个唯一ID
    return {token: i for i, token in enumerate(all_tokens)}


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


# 将词频计数转换为数值分布
def convert_counters_to_numeric(counters):
    numeric_distributions = []
    for counter in counters:
        numeric_distribution = [item for sublist in [([token] * freq) for token, freq in counter.items()] for item in
                                sublist]
        numeric_distributions.append(numeric_distribution)
    return numeric_distributions


# 绘制KDE图
def plot_kde_distributions(numeric_distributions, labels, save_path):
    plt.figure(figsize=(10, 7))
    for distribution, label in zip(numeric_distributions, labels):
        sns.kdeplot(distribution, bw_adjust=0.5, label=label)
    plt.legend()
    plt.title('KDE of Token Distributions')
    plt.xlabel('Token ID')
    plt.ylabel('Density')
    plt.xticks([])
    plt.yticks([])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 主流程
def main(samples_dir, refine_path, ref_path):
    # 读取和预处理数据
    refine_texts = tokenize_texts(read_file(refine_path))
    ref_texts = tokenize_texts(read_file(ref_path))
    sample_files = []
    for i in range(len(os.listdir(samples_dir))):
        file_path = os.path.join(samples_dir, f"{i}")
        if os.path.isfile(file_path):
            sample_files.append(file_path)

    # 创建词汇到ID的映射
    token_to_id = create_vocab_mapping(refine_texts, ref_texts,
                                       *[read_file(f, is_json=True)['all_gens'].values() for f in sample_files])

    # 处理gen数据
    gen_texts = []
    for file_path in sample_files:
        data = read_file(file_path, is_json=True)
        for text in data['all_gens'].values():
            gen_texts.extend(tokenize_texts(text))
    gen_counters = [count_tokens(text, token_to_id) for text in gen_texts]

    # 处理refine和ref数据
    refine_counters = count_tokens(refine_texts, token_to_id)
    ref_counters = count_tokens(ref_texts, token_to_id)

    # 计算JSD散度
    gen_jsds = [calculate_jsd(counter, ref_counters) for counter in gen_counters]
    refine_jsd = calculate_jsd(refine_counters, ref_counters)
    avg_gen_jsd = np.mean(gen_jsds)

    # 保存JSD计算结果
    jsd_results_path = os.path.join(samples_dir, 'jsd_results_id.json')
    with open(jsd_results_path, 'w') as f:
        json.dump({'avg_gen_jsd': avg_gen_jsd, 'refine_jsd': refine_jsd}, f, indent=4)

    # 绘制并保存KDE图
    numeric_distributions = convert_counters_to_numeric([gen_counters, [refine_counters], [ref_counters]])
    kde_save_path = os.path.join(samples_dir, 'kde_distribution.png')
    plot_kde_distributions(numeric_distributions, ['Gen', 'Refine', 'Ref'], kde_save_path)


# 执行主流程
langs = ["gu", "kk", "ne", "si"]
for lang in langs:
    samples_dir = f'/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/results/jsd_only_gen/{lang}2en'
    refine_path = f'{samples_dir}/test_got_refine'
    ref_path = f'{samples_dir}/ref'
    main(samples_dir, refine_path, ref_path)
