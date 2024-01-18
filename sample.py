import numpy as np


def sample_from_file(filenames, num_samples):
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()

    total_lines = len(lines)
    mean = total_lines / 2  # 中心位置
    std_dev = total_lines / 6  # 标准差，可以根据需要进行调整

    samples = np.random.normal(mean, std_dev, num_samples).astype(int)
    # 保证行号在有效范围内
    valid_samples = [lines[s] for s in samples if 0 <= s < total_lines]

    return valid_samples


if __name__ == "__main__":
    pairs = ['de-en', 'es-en', 'fi-en', 'ru-en', 'hi-en', 'zh-en']
    for pair in pairs:
        print(pair)
        src = pair.split('-')[0]
        tgt = pair.split('-')[1]
        split = 'train'
        src_file = f"bpe/{split}.{src}-{tgt}.{src}"
        tgt_file = f"bpe/{split}.{src}-{tgt}.{tgt}"
        src_output = f"bpe_temp/{split}.{src}-{tgt}.{src}"
        tgt_output = f"bpe_temp/{split}.{src}-{tgt}.{tgt}"
        num_samples = 2000  # 替换为您希望抽取的样本数量
        print('read' + src)
        with open(src_file, 'r') as f:
            src_lines = f.readlines()

        total_lines = len(src_lines)
        mean = total_lines / 2  # 中心位置
        std_dev = total_lines / 6  # 标准差，可以根据需要进行调整

        # 在0-total_lines之间随机抽取num_samples个整数


        samples = np.random.normal(mean, std_dev, num_samples).astype(int)
        print('write' + src)
        with open(src_output, 'w') as f:
            f.writelines([src_lines[s] for s in samples if 0 <= s < total_lines])
        del src_lines
        print('read' + tgt)
        with open(tgt_file, 'r') as f:
            tgt_lines = f.readlines()
        print('write' + tgt)
        with open(tgt_output, 'w') as f:
            f.writelines([tgt_lines[s] for s in samples if 0 <= s < total_lines])

        # 删除src_lines 和 tgt_lines

        del tgt_lines


