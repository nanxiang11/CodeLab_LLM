import torch
from torch import nn


"""
种归一化有助于通过确保权重的规模不会变得过大或过小来稳定学习过程，这在具有许多层的深度学习模型中特别有用。
"""


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    归一化方法：
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    与 LayerNorm 的区别：
        - 不减去均值
        - 更轻量，计算效率高
        - 常用于大型 Transformer 模型，如 LLaMA
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Args:
            dim (int): 输入特征维度
            eps (float): 避免除以零的微小常数
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 γ，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        核心归一化操作：按最后一个维度计算 RMS
        Args:
            x (torch.Tensor): 输入张量，形状 (..., dim)
        Returns:
            torch.Tensor: 归一化后的张量
        """
        # 计算均方根
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 将输入除以 RMS，实现归一化
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量，形状 (..., dim)
        Returns:
            torch.Tensor: 归一化并缩放后的输出
        """
        # 转为 float 避免低精度问题
        x_norm = self._rms_norm(x.float())
        # 转回原始数据类型，并乘以可学习权重 γ
        return x_norm.type_as(x) * self.weight




if __name__ == "__main__":
    norm = RMSNorm(768, 0.001)
    x = torch.randn(1, 60, 768)
    output = norm(x)
    print(output.shape)

