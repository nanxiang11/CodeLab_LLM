"""
text = (
    "在自然语言处理中，语言模型是一种用于计算句子概率的模型。"
    "它可以被用来生成文本、翻译语言、回答问题。"
    "为了让模型能够理解更复杂的上下文，我们通常需要将长文本切分为较短的片段。"
)


chunk_len = 50
overlap = 10

chunks = []
strat = 0
step = chunk_len - overlap

while strat < len(text):
    end = min(strat + chunk_len, len(text))
    chunk = text[strat: end]
    chunks.append(chunk)
    if end == len(text):
        break
    strat += step

for i, c in enumerate(chunks):
    print(f"--- chunk {i} ({len(c)}字) ---")
    print(c)
    print()
"""
import json

from tqdm import tqdm


def split_text(text, chunk_size=256, overlap=32):
    """
    将文本按指定长度切分成块，支持滑动窗口，保证相邻块有 overlap 个字符重叠。

    参数：
        text (str): 待切分文本
        chunk_size (int): 每块最大字符数
        overlap (int): 相邻块重叠字符数

    返回：
        List[str]: 切分后的文本块列表
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    chunks = []
    step = chunk_size - overlap
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step  # 滑动窗口
    return chunks


pretrain_data = "/root/StudyLLM/NX_LLM/第二章 动手实现/Dataset/mobvoi_seq_monkey_general_open_corpus_min.jsonl"
output_pretrain_data = "/root/StudyLLM/NX_LLM/第二章 动手实现/Dataset/pretrain_data.jsonl"

with open(output_pretrain_data, 'a', encoding='utf-8', errors='ignore') as pretrain:
    # 注意这里加了 errors='ignore'，避免非 utf-8 字节报错
    with open(pretrain_data, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc=f"Processing lines in {pretrain_data}", leave=False):
            try:
                line = json.loads(line)   # 尝试解析 JSON
                text = line.get('text', '')  # 保险：如果没有 text 字段返回空字符串
                chunks = split_text(text)
                for chunk in chunks:
                    pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
            except Exception as e:
                # 如果某行解析失败（坏行 / 编码问题），跳过
                print(f"跳过坏行: {e}")