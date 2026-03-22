import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, GPT2TokenizerFast
import sys, yaml, os
import json

import numpy as np

def get_knn(model_emb, text_emb, dist='cos'):
    if dist == 'cos':
        adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
            model_emb.size(0), -1, -1)
        adjacency = -torch.norm(adjacency, dim=-1)
    topk_out = torch.topk(adjacency, k=6, dim=0)
    return topk_out.values, topk_out.indices


def rounding_func(text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    decoded_out_lst = []
    
    model_emb = model.weight
    down_proj_emb2 = None

    dist = 'l2'
    
    for text_emb in text_emb_lst:
        import torch
        text_emb = torch.tensor(text_emb)
        if len(text_emb.shape) > 2:
            text_emb = text_emb.view(-1, text_emb.size(-1))
        else:
            text_emb = text_emb
        val, indices = get_knn((down_proj_emb2 if dist == 'cos' else model_emb),
                                text_emb.to(model_emb.device), dist=dist)
    
        decoded_out_lst.append(tokenizer.decode_token(indices[0]))

    return decoded_out_lst

def compute_logp(args, model, x, input_ids):
    word_emb = model.weight
    sigma = 0.1
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)
    word_emb_flat = word_emb.unsqueeze(1) 
    diff = (x_flat - word_emb_flat) ** 2 

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  
    logp_expanded = logp_expanded.permute((1, 0))

    ce = torch.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)

    return loss

def get_weights(model, args):
    if hasattr(model, 'transformer'):
        input_embs = model.transformer.wte
        down_proj = model.down_proj
        model_emb = down_proj(input_embs.weight)
        print(model_emb.shape)
        model = torch.nn.Embedding(model_emb.size(0), model_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = model_emb * args.emb_scale_factor

    elif hasattr(model, 'weight'):
        pass
    else:
        assert NotImplementedError
        
    model.weight.requires_grad = False
    return model


def get_efficient_knn(model_emb, text_emb):
    """
    计算模型嵌入 (model_emb) 和文本嵌入 (text_emb) 之间的 K 最近邻 (KNN) 距离
    """
    # 使用 L2 范数计算模型嵌入的平方和
    # 对 model_emb 中的每个元素取平方, 对最后一个维度（特征维度）求和, 得到每个嵌入的平方和,
    # 将结果转换为形状为 (N, 1) 的张量，其中 N 是模型嵌入的数量
    emb_norm = (model_emb**2).sum(-1).view(-1, 1)
    # 转置文本嵌入
    # 如果 text_emb 的形状大于 2, 展平为二维张量,
    # 转置张量，使得原本的行变为列，方便后续的矩阵乘法
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
    # 计算文本嵌入的范数
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
    # 计算每个文本嵌入的 L2 范数并转换为形状 (M, 1), M 为文本嵌入的数量
    # torch.mm(model_emb, text_emb_t)：进行矩阵乘法，计算模型嵌入与文本嵌入的点积，结果形状为 (N, M)
    # 整个表达式计算出每对嵌入之间的欧几里得距离
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)
    # 确保所有距离值非负
    dist = torch.clamp(dist, 0.0, np.inf)
    # 获取 K 最近邻:
    # -dist：取负值，因为 torch.topk 默认返回最大值，而我们需要最小距离
    # k=1：获取最近的一个邻居
    # dim=0：在第一个维度上进行操作，得到每个文本嵌入的最近邻
    topk_out = torch.topk(-dist, k=1, dim=0)
    # 返回最近邻的距离值和索引
    return topk_out.values, topk_out.indices


def denoised_fn_round(args, model, text_emb, t):
    """
    根据 KNN 结果更新文本嵌入
    """
    # 从模型中提取嵌入权重，通常是模型的参数
    model_emb = model.weight
    # 保存 text_emb 的原始形状与所在的设备，以便后续恢复
    old_shape = text_emb.shape
    old_device = text_emb.device
    # 展平文本嵌入：
    # 如果 text_emb 的维度大于 2，展平为二维张量
    # 这样可以将多维输入转换为适合 KNN 计算的格式
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # 计算 KNN：
    # 将 text_emb 移动到与 model_emb 相同的设备上，然后调用 get_efficient_knn 函数，计算最近邻的索引和距离
    val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    # 从返回的索引中提取最近邻的索引 (每个文本嵌入的最近邻)
    rounded_tokens = indices[0]
    # 使用最近邻的索引从模型中获取新的嵌入, 恢复为原始形状并将新的嵌入移动回原始设备
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    # 返回更新后的嵌入
    return new_embeds
