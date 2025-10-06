"""
预览超大json，文本，通用方法
"""
import json

# 我们定义一个文件路径
file_path = "/root/StudyLLM/NX_LLM/第二章 动手实现/Dataset/pretrain_data.jsonl"


k = 1000


with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= k:
            break
        try:
            data = json.loads(line.strip())
            print(data)
        except json.JSONDecodeError:
            print("解析失败")