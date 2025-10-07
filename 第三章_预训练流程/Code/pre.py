# -*- coding: utf-8 -*-
"""
简化版文本生成器示例
功能：
- 加载预训练模型
- 自动选择 CPU/GPU
- 支持混合精度推理
- 提供简单的生成接口
"""

from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from ModelConfig import ModelConfig
from LLaMA2 import Transformer


class TextGenerator:
    def __init__(self,
                 checkpoint_path,          # 模型检查点路径
                 tokenizer_path,           # 分词器路径
                 device=None,              # 推理设备，默认使用 GPU
                 dtype="bfloat16",         # 浮点精度
                 seed=42):                 # 随机种子，保证生成可复现
        """
        初始化文本生成器
        """
        # ---------------- 设备与随机种子 ----------------
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.dtype = dtype

        torch.manual_seed(seed)
        if 'cuda' in self.device:
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ---------------- 自动混合精度上下文 ----------------
        dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=dtype_map[self.dtype])

        # ---------------- 模型加载 ----------------
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = Transformer(ModelConfig(dim=512, n_layers=8))

        # 处理 checkpoint 中可能多余的前缀
        prefix = '_orig_mod.'
        for k in list(checkpoint.keys()):
            if k.startswith(prefix):
                checkpoint[k[len(prefix):]] = checkpoint.pop(k)

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()            # 推理模式
        self.model.to(self.device)   # 移动到 GPU/CPU

        # 模型参数量
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型加载完成，共 {total_params/1e6:.2f} M 参数。")

        # ---------------- 分词器加载 ----------------
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def generate_text(self, prompt, max_new_tokens=128, temperature=0.7, top_k=50, num_samples=1):
        """
        根据输入 prompt 生成文本

        参数：
        - prompt: str, 生成的起始文本
        - max_new_tokens: 每条生成文本的最大 token 数
        - temperature: 生成随机性，越大越随机
        - top_k: 采样时保留概率最高的 top_k token
        - num_samples: 生成样本数量

        返回：
        - List[str]，生成的文本列表
        """
        # 编码文本为 token ID
        input_ids = torch.tensor(self.tokenizer(prompt).data['input_ids'], device=self.device).unsqueeze(0)

        outputs = []
        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    generated_ids = self.model.generate(input_ids,
                                                        max_new_tokens=max_new_tokens,
                                                        temperature=temperature,
                                                        top_k=top_k)
                    text = self.tokenizer.decode(generated_ids[0].tolist())
                    outputs.append(text)
        return outputs


# --------------------- 使用示例 ---------------------
if __name__ == "__main__":
    # 模型路径
    checkpoint_path = '/root/StudyLLM/NX_LLM/第三章_预训练流程/ModelZoom/base-CodeLab-26M/pretrain_512_8_6144_step9000.pth'
    tokenizer_path = '/root/StudyLLM/NX_LLM/第三章_预训练流程/tokenizer_k'

    # 初始化生成器
    generator = TextGenerator(checkpoint_path, tokenizer_path)

    # 测试生成
    pretrain_prompt_datas = [
        '<|im_start|>教学',
        '<|im_start|>建设',
    ]

    for i, prompt in enumerate(pretrain_prompt_datas):
        samples = generator.generate_text(prompt, max_new_tokens=120, temperature=0.75, num_samples=1)
        print(f"\nSample {i + 1}:\n{samples[0]}\n{'-'*30}")
