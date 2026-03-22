"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    创建正弦时间步嵌入。
    :param timesteps: 一个一维张量，包含 N 个时间步索引（可为小数），通常代表批次中的每个样本。
    :param dim: 输出嵌入的维度，控制嵌入的大小。
    :param max_period: 控制嵌入的最小频率，影响正弦波的周期。
    :return: 一个形状为 [N x dim] 的位置嵌入张量。
    """
    half = dim // 2 # 将输出维度 dim 除以 2，得到 half，用于后续计算频率
    # th.arange(start=0, end=half, dtype=th.float32)：生成从 0 到 half - 1 的张量，表示不同的频率
    # math.log(max_period)：计算最大周期的自然对数
    # th.exp(...)：使用指数函数计算频率，生成一个频率张量 freqs，其形状为 [half]
    # .to(device=timesteps.device)：确保频率张量与输入 timesteps 在同一设备上（如 CPU 或 GPU）
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    # timesteps[:, None]：将 timesteps 转换为列向量，以便与频率张量进行广播
    # float()：确保 timesteps 为浮点类型
    # freqs[None]：将频率张量转换为行向量，便于与 timesteps 进行乘法运算
    # args 将是一个形状为 [N x half] 的张量，存储每个时间步与频率的乘积
    args = timesteps[:, None].float() * freqs[None]
    # th.cos(args) 和 th.sin(args)：分别计算 args 的余弦和正弦值
    # th.cat(..., dim=-1)：将余弦和正弦值在最后一个维度上进行拼接，生成一个形状为 [N x dim] 的嵌入张量
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    # 如果 dim 是奇数，则在嵌入中添加一个全零的列，以确保输出张量的维度为偶数
    # th.zeros_like(embedding[:, :1]) 作用是创建一个与嵌入第一列形状相同的全零张量
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    # 返回形状为 [N x dim] 的时间步嵌入张量
    return embedding