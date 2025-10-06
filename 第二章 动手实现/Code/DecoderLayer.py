import torch.nn.functional as F
import math
import torch
from torch import nn
from ModelConfig import ModelConfig
from Attention import Attention
from MLP import LlamaMLP
from RMSNorm import RMSNorm
from ROPE import *

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 下面我们按照框架图一步一步来就行
        # 定义输入特征维度
        self.dim = args.dim
        # 定义有几个多头注意力头
        self.n_heads = args.n_heads
        # 定义每个头的维度，等于输入维度除以头数,这个要千万注意
        self.head_dim = args.dim // args.n_heads
        # 定义注意力机制
        self.attention = Attention(args)
        # 定义FFN
        self.FFN = LlamaMLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        # 设置层id
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        单层解码器前向（pre-norm 风格）
        Args:
            x: [batch_size, seq_len, dim] 输入张量（残差输入）
            freqs_cos, freqs_sin: RoPE 预计算的 cos / sin，shape 与 Attention 要求一致
        Returns:
            out: [batch_size, seq_len, dim] 经过本层处理后的输出（已加残差）
        """
        # 先对输入做 RMSNorm（归一化），再传入 Attention 计算
        # 注意：attention 接受的是原始 x 的归一化版本（x_norm）
        # Shapes:
        #   x              -> [B, L, D]
        #   x_norm         -> [B, L, D]
        x_norm = self.attention_norm(x)

        # attention 返回 [B, L, D]
        attn_out = self.attention(x_norm, freqs_cos, freqs_sin)

        # 残差连接：将 attention 的输出加回到原始输入
        x = x + attn_out

        # 对残差后的 x 做 RMSNorm，再输入到 FFN（LlamaMLP）
        # Shapes:
        #   x             -> [B, L, D]
        #   x_ffn_norm    -> [B, L, D]
        x_ffn_norm = self.ffn_norm(x)

        # FFN 返回 [B, L, D]
        ffn_out = self.FFN(x_ffn_norm)

        # 残差连接：将 FFN 的输出加回
        x = x + ffn_out

        # 返回本层最终输出
        return x




if __name__ == "__main__":
    args = ModelConfig()
    decoderlayer = DecoderLayer(0, args)
    # 模拟输入数据
    dim = args.dim
    seq_len = 50

    x = torch.randn(1, seq_len, dim)  # [bs, seq_len, dim]

    freqs_cos, freqs_sin = precompute_rotary_freqs(dim // args.n_heads, seq_len)

    out = decoderlayer(x, freqs_cos, freqs_sin)

    print(out.shape)  # 形状和输入的x一样 [batch_size, seq_len, dim]