import os
import json
from typing import List
import torch
from torch.utils.data import Dataset, get_worker_info

class PretrainMapDataset(Dataset):
    """
    Map-style 数据集（按行 jsonl），不会一次性加载整个文件。
    返回：
      tuple: (input_ids, labels, attention_mask)
      - input_ids: (L,) torch.long, 包含 BOS token
      - labels: (L,) torch.long, padding -> -100
      - attention_mask: (L,) torch.long, 1/0
    """
    def __init__(self, path: str, tokenizer, max_length: int = 256):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

        # 强制使用 pad_id=0，BOS/EOS 用 tokenizer 对应 id
        self.pad_id = 0
        self.bos_id = getattr(self.tokenizer, "bos_token_id", None)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)

        # 构建文件行偏移表
        self._build_offsets()
        self._fp = None

    def _build_offsets(self):
        self.offsets: List[int] = []
        cur = 0
        with open(self.path, "rb") as f:
            for line in f:
                self.offsets.append(cur)
                cur += len(line)
        self._len = len(self.offsets)

    def __len__(self):
        return self._len

    def _ensure_file_open(self):
        if self._fp is None:
            self._fp = open(self.path, "r", encoding="utf-8")

    def _read_line_by_index(self, idx: int) -> str:
        self._ensure_file_open()
        self._fp.seek(self.offsets[idx])
        line = self._fp.readline()
        return line.rstrip("\n")

    def _tokenize_to_ids(self, text: str) -> List[int]:
        if text is None:
            text = ""
        # 获取 token ids
        ids = self.tokenizer(text).data['input_ids'][:self.max_length]
        ids = [int(i) for i in ids]
        # 添加 BOS token
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        # 截断
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        return ids

    def __getitem__(self, index: int):
        line = self._read_line_by_index(index)
        try:
            obj = json.loads(line)
            text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
        except Exception:
            text = line

        ids = self._tokenize_to_ids(text)

        # padding 到 max_length
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
        elif pad_len < 0:
            ids = ids[:self.max_length]

        seq = torch.tensor(ids, dtype=torch.long)
        input_ids = seq[:-1]
        labels = seq[1:].clone()
        attention_mask = (seq != self.pad_id).long()[:-1]

        return input_ids, labels, attention_mask

    def __del__(self):
        try:
            if self._fp is not None:
                self._fp.close()
        except Exception:
            pass

# ====== 测试例子 ======
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 创建临时 jsonl 文件
    test_file = "test.jsonl"
    texts = [
        {"text": "我爱南巷的花猫"},
        {"text": "今天天气很好"},
        {"text": "机器学习真有趣"}
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/StudyLLM/NX_LLM/第二章 动手实现/tokenizer_k"
    )

    dataset = PretrainMapDataset(path=test_file, tokenizer=tokenizer, max_length=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch in dataloader:
        X, Y, mask = batch
        print("input_ids:\n", X)
        print("labels:\n", Y)
        print("attention_mask:\n", mask)
        break

    os.remove(test_file)
