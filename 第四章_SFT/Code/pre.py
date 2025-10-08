from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from ModelConfig import ModelConfig
from LLaMA2 import Transformer


class TextGenerator:
    def __init__(self,
                 base_model_path='',  # 原始基础模型路径，可选
                 lora_checkpoint='',  # LoRA 微调权重路径
                 tokenizer_model_path='/root/StudyLLM/NX_LLM/第三章_预训练流程/tokenizer_k',
                 seed=42,
                 device=None,
                 dtype="bfloat16"):
        """
        初始化 TextGenerator:
        - 加载基础模型 + LoRA 权重
        - 设置设备和自动混合精度
        """
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.dtype = dtype
        self.seed = seed

        torch.manual_seed(seed)
        if 'cuda' in self.device:
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=ptdtype
        )

        # -----------------------------
        # 加载基础模型
        # -----------------------------
        print("加载基础模型...")
        self.model = Transformer(ModelConfig(dim=512, n_layers=16))  # 按你的基础模型参数
        if base_model_path:
            checkpoint_dict = torch.load(base_model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint_dict, strict=False)

        # -----------------------------
        # 加载 LoRA 权重
        # -----------------------------
        if lora_checkpoint:
            print("加载 LoRA 权重...")
            # from_pretrained 会返回一个包装了 LoRA 的 PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint, device_map={'': self.device})

        # -----------------------------
        # 推理模式 & 设备
        # -----------------------------
        self.model.eval()
        self.model.to(self.device)

        # -----------------------------
        # 分词器
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

        # -----------------------------
        # 打印可训练参数量
        # -----------------------------
        num_params = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad)
        num_lora_params = sum(p.numel() for n, p in self.model.named_parameters() if 'lora' in n)
        print(f"总可训练参数量: {num_params / 1e6:.6f} M")
        print(f"LoRA 参数量: {num_lora_params / 1e6:.6f} M")

    # -----------------------------
    # 简单 chat 模板
    # -----------------------------
    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是一个AI助手，你的名字叫小明。"},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # -----------------------------
    # 生成文本
    # -----------------------------
    def sft_sample(self, start="Hello!", num_samples=3, max_new_tokens=256, temperature=0.7, top_k=50):
        start = self.chat_template(start)
        start_ids = self.tokenizer(start).data['input_ids']
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        generated_texts = []
        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    y = self.model.generate(
                        x,
                        stop_id=self.tokenizer.eos_token_id,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))
        return generated_texts


if __name__ == "__main__":
    prompts = [
        "你好呀",
        "你是谁？",
        "今天天气怎么样？"
    ]

    generator = TextGenerator(
        base_model_path='/root/StudyLLM/NX_LLM/第三章_预训练流程/ModelZoom/base-CodeLab-50M/pretrain_512_16_6144_step67200.pth',
        lora_checkpoint='/root/StudyLLM/NX_LLM/第四章_SFT/ModelZoom/SFT-CodeLab-50M-LORA/pretrain_512_16_6144_step22800',
        tokenizer_model_path='/root/StudyLLM/NX_LLM/第三章_预训练流程/tokenizer_k'
    )

    for i, prompt in enumerate(prompts):
        samples = generator.sft_sample(start=prompt, num_samples=1, max_new_tokens=128, temperature=0.3)
        print(f"\nSample {i + 1}:\nQuestion: {prompt}\nAI answer: {samples[0]}\n{'-'*20}")
