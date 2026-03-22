"""
Helpers to train with 16-bit precision.
"""

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


def make_master_params(model_params):
    """
    将模型参数复制到全精度参数的（不同形状的）列表中.
    """
    # 使用列表推导式将每个参数转换为全精度格式 float()
    # 并调用 detach() 以确保在计算图中不再跟踪这些参数的梯度
    # _flatten_dense_tensors 函数将这些参数扁平化为一个单一的张量
    master_params = _flatten_dense_tensors([param.detach().float() for param in model_params])
    # 将扁平化后的张量转换为 nn.Parameter 类型，以便在训练过程中可以更新
    master_params = nn.Parameter(master_params)
    # 设置 requires_grad 为 True，确保在训练中计算梯度
    master_params.requires_grad = True
    # 返回一个包含全精度参数的列表
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def zero_grad(model_params):
    for param in model_params:
        """
        将模型参数的梯度清零
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        """
        if param.grad is not None:
            # 调用 detach_() 方法来断开当前梯度与计算图的连接
            # 这意味着在后续操作中不会计算该梯度的反向传播
            param.grad.detach_()
            # 调用 zero_() 方法将当前参数的梯度值清零
            param.grad.zero_()
