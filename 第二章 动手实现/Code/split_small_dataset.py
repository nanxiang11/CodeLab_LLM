from tqdm import tqdm


file_path = "/root/autodl-tmp/Dataset/mobvoi_seq_monkey_general_open_corpus.jsonl"

count = 0
with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
    for _ in f:
        count += 1

subset_lines = count // 10

output_file = "/root/StudyLLM/NX_LLM/第二章 动手实现/Dataset/mobvoi_seq_monkey_general_open_corpus_min.jsonl"

with open(file_path, "r", encoding="utf-8", errors="ignore") as fin, \
        open(output_file, "w", encoding="utf-8") as fout, \
        tqdm(total=subset_lines, desc="抽取进度", unit="行") as pbar:
    for i, line in enumerate(fin):
        if i >= subset_lines:
            break
        fout.write(line)
        pbar.update(1)

print(f"已完成抽取，结果保存到：{output_file}")
