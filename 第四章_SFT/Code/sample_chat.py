from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from ModelConfig import ModelConfig
from LLaMA2 import Transformer
import torch.nn.functional as F



class TextGenerator:
    def __init__(self,
                 base_model_path='',   # 基础模型 checkpoint，可选
                 lora_checkpoint='',   # LoRA checkpoint，可选
                 tokenizer_model_path='/root/StudyLLM/NX_LLM/第三章_预训练流程/tokenizer_k',
                 seed=42,
                 device=None,
                 dtype="bfloat16"):
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
        self.model = Transformer(ModelConfig(dim=512, n_layers=16))
        if base_model_path:
            checkpoint_dict = torch.load(base_model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint_dict, strict=False)

        # -----------------------------
        # 加载 LoRA 权重
        # -----------------------------
        if lora_checkpoint:
            print("加载 LoRA 权重...")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint, device_map={'': self.device})

        self.model.eval()
        self.model.to(self.device)

        # -----------------------------
        # 分词器
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

        # -----------------------------
        # 打印参数量
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
    # 逐步生成函数
    # -----------------------------
    def sft_sample(self, start="Hello!", num_samples=1, max_new_tokens=128,
                   temperature=0.7, top_k=50, stop_token_id=None):
        """
        基于模型 forward 逐步生成 token
        """
        start_text = self.chat_template(start)
        start_ids = self.tokenizer(start_text).data['input_ids']
        idx = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]  # [1, seq_len]

        stop_token_id = stop_token_id or self.tokenizer.eos_token_id
        outputs = []

        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    cur_idx = idx.clone()
                    generated = []

                    for _ in range(max_new_tokens):
                        logits = self.model(cur_idx).logits[:, -1, :]  # [batch, vocab]
                        logits = logits / temperature
                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

                        if next_token.item() == stop_token_id:
                            break

                        generated.append(next_token.item())
                        cur_idx = torch.cat([cur_idx, next_token], dim=1)

                    outputs.append(self.tokenizer.decode(generated))

        return outputs

    # -----------------------------
    # 控制台聊天函数
    # -----------------------------
    def chat_console(self):
        print("输入 'exit' 退出对话。")
        while True:
            user_input = input("你: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            answers = self.sft_sample(start=user_input, num_samples=1, max_new_tokens=128,
                                      temperature=0.8, top_k=50)
            print(f"小明: {answers[0]}")


# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    generator = TextGenerator(
        base_model_path='/root/StudyLLM/NX_LLM/第三章_预训练流程/ModelZoom/base-CodeLab-50M/pretrain_512_16_6144_step67200.pth',
        lora_checkpoint='/root/StudyLLM/NX_LLM/第四章_SFT/ModelZoom/SFT-CodeLab-50M-LORA/pretrain_512_16_6144_step4800',
        tokenizer_model_path='/root/StudyLLM/NX_LLM/第三章_预训练流程/tokenizer_k'
    )

    generator.chat_console()


