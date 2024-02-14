import json

import sentencepiece as spm
import numpy as np
import seaborn as sns
from collections import Counter
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import os

# 加载sentencepiece模型
sp = spm.SentencePieceProcessor()
sp.load('/mnt/e/unmt/acl22-sixtp/models/xlmrL_base/sentencepiece.bpe.model')

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

# 可视化词频分布
def plot_distributions(counters, labels):
    fig, ax = plt.subplots(figsize=(10, 7))
    for counter, label in zip(counters, labels):
        tokens = list(counter.keys())
        frequencies = list(counter.values())
        indices = np.arange(len(tokens))
        ax.bar(indices, frequencies, label=label)
        ax.set_xticks(indices)
        ax.set_xticklabels(tokens, rotation='vertical')
        ax.set_ylabel('Frequency')
        ax.set_title('Word Distribution')
    ax.legend()
    plt.show()

def create_mapping(counters):
    unique_tokens = set(token for counter in counters for token in counter.keys())
    return {token: i for i, token in enumerate(unique_tokens)}


# 修改可视化函数以处理数值数据
def plot_distributions_with_kde_numeric(counters, labels, mapping):
    fig, axs = plt.subplots(len(counters), 1, figsize=(10, 7 * len(counters)), sharex=False)

    if len(counters) == 1:
        axs = [axs]  # 保证axs是列表，即使只有一个计数器

    for ax, counter, label in zip(axs, counters, labels):
        # 将数据转换为数值
        numeric_values = convert_to_numeric(counter, mapping)

        # 绘制KDE曲线
        sns.kdeplot(numeric_values, ax=ax, bw_adjust=0.5, label=f"{label} KDE")

        # 设置x轴标签
        ax.set_xticks(list(mapping.values()))
        ax.set_xticklabels(list(mapping.keys()), rotation='vertical')
        ax.legend()
        ax.set_title(f'KDE for {label}')

    plt.tight_layout()
    plt.show()


# 将所有数据转换为数值并合并
def convert_all_to_numeric(counters, mapping):
    numeric_values_list = []
    for counter in counters:
        numeric_values = []
        for token, freq in counter.items():
            numeric_values.extend([mapping[token]] * freq)
        numeric_values_list.append(numeric_values)
    return numeric_values_list

# 更新转换数据的函数，只包括那些最大密度非零的词
def convert_to_numeric(counter, mapping):
    numeric_values = []
    for token in mapping:
        numeric_values.extend([mapping[token]] * counter.get(token, 0))
    return numeric_values

# 绘制所有分布在一个图中的函数
def plot_combined_distributions_with_kde(numeric_values_list, labels, reduced_mapping):
    plt.figure(figsize=(12, 8))

    # 绘制每个数据集的KDE曲线
    for numeric_values, label in zip(numeric_values_list, labels):
        sns.kdeplot(numeric_values, bw_adjust=0.5, label=label)

    # 设置图例和标题
    plt.legend(title='Legend')
    plt.title('Combined KDE of Translations and Reference')

    # 设置x轴标签为词汇
    # plt.xticks(list(reduced_mapping.values()), list(reduced_mapping.keys()), rotation='vertical')
    plt.xlabel('Vocabulary')
    plt.xticks([])
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()


# 示例文本
texts = ["According to the latest data from the FBI hate crime reporting program, 100 hate crimes were documented in Missouri in 2015, placing the state 16th across the state in terms of the number of similar violations.",
          "According to the latest data from the FBI's hate crime reporting program, 100 hate crimes were documented in Missouri in 2015, ranking the state 16th in the country in terms of the number of similar violations.",
          "Missouri recorded 100 hate crimes in 2015, according to the latest figures from the FBI's hate crime reporting program, ranking the state at 16th in the country in terms of the number of such violations."]
tokenized_texts = [tokenize_texts(text) for text in texts]



# 统计词频
counters = [count_tokens(tokens) for tokens in tokenized_texts]
# 计算所有文本中每个词的最大密度
max_density_per_token = Counter()
for counter in counters:
    total = sum(counter.values())
    for token, count in counter.items():
        max_density_per_token[token] = max(max_density_per_token[token], count / total)

# 仅为那些最大密度非零的词创建映射
tokens_with_nonzero_density = [token for token, density in max_density_per_token.items() if density > 0]
mapping = {token: i for i, token in enumerate(tokens_with_nonzero_density)}

# 转换所有数据为数值
numeric_values_list = [convert_to_numeric(counter, mapping) for counter in counters]

# 计算JSD散度
jsd12 = calculate_jsd(counters[0], counters[1])
jsd1r = calculate_jsd(counters[0], counters[2])
jsd2r = calculate_jsd(counters[1], counters[2])
print(f"JSD between Translation 1 and Translation 2: {jsd12}")
print(f"JSD between Translation 1 and Reference: {jsd1r}")
print(f"JSD between Translation 2 and Reference: {jsd2r}")

# 可视化词频分布
# 使用改进的函数
# plot_distributions_with_kde_numeric(counters, texts, mapping)
# 使用改进的函数绘制合并的KDE图
plot_combined_distributions_with_kde(numeric_values_list, ["single_aux", "POMP", "reference"], mapping)