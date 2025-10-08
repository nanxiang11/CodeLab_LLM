import torch
import torch.nn.functional as F
from torch import nn
from ModelConfig import ModelConfig


class LlamaMLP(nn.Module):
    """
    输入维度：dim
    中间维度：hidden_dim（若未指定，则自动计算为 4/3 * dim，并取 multiple_of 的倍数）

    结构：
        x -> W1 -> SiLU() 激活
        x -> W3 -> gating 通道
        二者逐元素相乘 (SwiGLU)
        -> W2 -> Dropout -> 输出
    等价于：output = Dropout(W2(SiLU(W1(x)) * W3(x)))
    """

    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()

        # 若未指定 hidden_dim，则：
        #   1. 取输入维度 dim 的 4 倍；
        #   2. 再缩小为 2/3；
        #   3. 调整为 multiple_of 的整数倍（例如 256 的倍数）。
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # --------------------------------------------
        # 定义线性层
        # --------------------------------------------
        # W1：主通道（经过 SiLU 激活）
        self.fc_gate = nn.Linear(dim, hidden_dim, bias=False)
        # W3：门控通道（直接相乘）
        self.fc_up = nn.Linear(dim, hidden_dim, bias=False)
        # W2：输出投影层（回到原始维度）
        self.fc_down = nn.Linear(hidden_dim, dim, bias=False)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            output: [batch_size, seq_len, dim]
        """
        # 主通道经过 SiLU 激活
        gated = F.silu(self.fc_gate(x))  # [B, L, hidden_dim]

        # 门控通道（线性变换）
        up = self.fc_up(x)  # [B, L, hidden_dim]

        # SwiGLU 核心：逐元素相乘
        hidden = gated * up  # [B, L, hidden_dim]

        # 输出投影 + Dropout
        output = self.fc_down(hidden)
        output = self.dropout(output)

        return output


# -------------------------------------------------
# 测试示例
# -------------------------------------------------
if __name__ == "__main__":
    args = ModelConfig()

    # 创建 MLP 模块实例
    mlp = LlamaMLP(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        multiple_of=args.multiple_of,
        dropout=args.dropout
    )

    # 随机输入 [batch=1, seq_len=50, dim]
    x = torch.randn(1, 50, args.dim)

    # 前向计算
    output = mlp(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
