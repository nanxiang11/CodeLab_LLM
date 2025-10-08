import json
import torch
from torch.utils.data import Dataset

class SFTMapDataset(Dataset):
    """
    高效 map-style SFT Dataset
    支持 role/content 格式数据，X/Y/loss_mask 对齐
    """
    def __init__(self, data_path, tokenizer, max_length=256, padding=0):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

        # 构建文件行偏移表
        self.offsets = []
        cur = 0
        with open(data_path, "rb") as f:
            for line in f:
                self.offsets.append(cur)
                cur += len(line)
        self._len = len(self.offsets)
        self._fp = None

    def __len__(self):
        return self._len

    def _ensure_file_open(self):
        if self._fp is None:
            self._fp = open(self.data_path, "r", encoding="utf-8")

    def _read_line(self, index):
        self._ensure_file_open()
        self._fp.seek(self.offsets[index])
        return self._fp.readline().rstrip("\n")

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0

        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找第一个 4 (eos_token_id)
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index):
        line = self._read_line(index)
        sample = json.loads(line)
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(text).data['input_ids'][:self.max_length]

        # padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.padding] * pad_len

        loss_mask = self.generate_loss_mask(input_ids)

        # X/Y/loss_mask 对齐
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask

    def __del__(self):
        if hasattr(self, "_fp") and self._fp is not None:
            try:
                self._fp.close()
            except:
                pass


# ===== 测试 =====
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import os

    test_file = "test_sft.jsonl"
    sample_data = [[
        {'role': 'system', 'content': '你是一个AI助手'},
        {'role': 'user', 'content': '根据以下文本，对此事件进行分类：中国队在足球比赛中赢得了冠军。'},
        {'role': 'assistant', 'content': '这个事件可以被分类为体育比赛。具体地，中国队在足球比赛中致胜并赢得了冠军。'}
    ]]

    with open(test_file, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained("/root/StudyLLM/NX_LLM/第二章 动手实现/tokenizer_k")

    dataset = SFTMapDataset(test_file, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for X, Y, loss_mask in dataloader:
        print("input_ids:\n", X)
        print("labels:\n", Y)
        print("loss_mask:\n", loss_mask)
        break

    os.remove(test_file)
