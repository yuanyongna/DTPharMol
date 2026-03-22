import os
import torch
import torch.distributed as dist
from diffumol.rounding import denoised_fn_round
import pandas as pd
import time
from diffumol.utils import dist_util, logger
from functools import partial


def smiles_sample(
    args,
    world_size,
    index,
    cond,
    model,
    diffusion,
    model_emb,
    tokenizer,
    general_path,
    start_t,
):
    smiles = []
    # 如果 cond 是一个空字典, 循环 world_size 次,
    # 调用分布式的屏障同步, 等待所有进程到达此屏障
    # 如果有进程未到达, 其他进程将被阻塞,确保所有进程都到达此点
    if not cond:
        for i in range(world_size):
            dist.barrier()
        # continue
    # 从 cond 中弹出 input_ids 键对应的值, 将其转换为浮点类型, 并将其移动到指定的设备
    input_ids_x = cond.pop("input_ids").to(torch.float).to(dist_util.dev())
    # 从 cond 中弹出 input_mask 键对应的值, 不需要进行类型转换或设备移动
    input_ids_mask = cond.pop("input_mask")

    if args.num_props and args.ppgraph_len:
        props = input_ids_x[:, : args.num_props + args.ppgraph_len].clone()
        props = model.get_props(props)
        x_start = torch.cat(
            [
                props,
                model.get_embeds(input_ids_x[:, args.num_props + args.ppgraph_len :]),
            ],
            1,
        )
        input_ids_mask = input_ids_mask[
            :, args.num_props + args.ppgraph_len - 1 :
        ].contiguous()

    elif args.num_props:
        props = input_ids_x[:, : args.num_props].clone()
        props = model.get_props(props)
        mol = input_ids_x[:, args.num_props :]
        x_start = torch.cat([props, model.get_embeds(mol)], 1)
        input_ids_mask = input_ids_mask[:, args.num_props - 1 :].contiguous()

    elif args.ppgraph_len:
        props = input_ids_x[:, : args.ppgraph_len].clone()
        props = model.get_props(props)
        x_start = torch.cat(
            [props, model.get_embeds(input_ids_x[:, args.ppgraph_len :])], 1
        )
        input_ids_mask = input_ids_mask[:, args.ppgraph_len - 1 :].contiguous()
        # logger.log(f"\n### 仅药效团")

    # 没有属性条件时
    else:
        x_start = model.get_embeds(input_ids_x)
        # logger.log(f"\n### 没有属性条件时, 直接使用 model.get_embeds 对 input_ids_x 进行嵌入得到 x_start: {x_start.shape}")

    input_ids_mask_ori = input_ids_mask
    # logger.log(f"\n### 保存一个掩码 input_ids_mask 的副本 input_ids_mask_ori: {input_ids_mask_ori.shape}")
    # 创建一个与 x_start 形状相同的张量, 其中的每个元素都是从标准正态分布中随机采样的
    noise = torch.randn_like(x_start)
    # logger.log(f"\n### 生成与 x_start 形状相同的随机噪声张量 noise: {noise.shape}")
    # input_ids_mask.unsqueeze(dim=-1) 在最后一个维度增加一个维度, 使其形状变为 [32, 103, 1]
    # torch.broadcast_to(..., x_start.shape): 将扩展后的 input_ids_mask 广播到 x_start 的形状 [32, 106, 128]
    # broadcast_to 不会复制数据，而是返回一个视图，使得原始数据可以在新形状下使用
    input_ids_mask = torch.broadcast_to(
        input_ids_mask.unsqueeze(dim=-1), x_start.shape
    ).to(dist_util.dev())
    # logger.log(f"\n### 使用 torch.broadcast_to 方法将 input_ids_mask 扩展到与 x_start 相同的形状, input_ids_mask: {input_ids_mask.shape}")
    # 根据掩码决定保留原始数据还是用噪声替换
    # 如果 input_ids_mask 中的元素为 0, 则保留 x_start 中的对应元素, 否则用 noise 中的对应元素替换 x_start 中的元素
    # 即保留性质与骨架信息, 将格式化的 SMILES 部分替换为随机噪声
    x_noised = torch.where(input_ids_mask == 0, x_start, noise)
    # logger.log(f"\n### 根据掩码的值, 将 x_start 中格式化的 SMILES 部分及之后的填充部分用噪声替换, 性质与骨架部分保留原始值, 得到 x_noised: {x_noised.shape}")
    # 初始化扩散参数默认值
    # logger.log(f"\n### 训练过程扩散步数 diffusion_steps={args.diffusion_steps}, 生成过程扩散步数 step={args.step}")
    if args.step == args.diffusion_steps:
        args.use_ddim = False
        step_gap = 1
        # logger.log("生成过程与训练过程的扩散步数相同, 则初始化 use_ddim=False, step_gap=1, 表示不使用 DDIM 方法且在每一步中使用每个扩散步骤")
    if args.step < args.diffusion_steps:
        args.use_ddim = True
        step_gap = args.diffusion_steps // args.step
        # logger.log(f"生成过程与训练过程的扩散步数不同, 则初始化 use_ddim=True, step_gap=diffusion_steps//step={step_gap}, 表示使用 DDIM 方法且在每一步中会跳过 {step_gap} 个扩散步骤")
    if args.step > args.diffusion_steps:
        args.use_ddim = True
        step_gap = args.diffusion_steps // args.step
        # logger.log(f"step_gap 不应该为 0, 但此处计算为: step={step_gap}, 请设置合理的参数 step={args.step} 值")
    # 一般情况下都会采用 use_ddim=False, 即使用 diffusion.p_sample_loop
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    # logger.log(f"\n### 根据 use_ddim 参数的取值, 初始化扩散采样模型 sample_fn: {sample_fn.__name__}")
    sample_shape = (
        x_start.shape[0],
        args.seq_len - args.num_props + 1,
        args.hidden_dim,
    )
    # logger.log(f"\n### 设置参数 sample_shape: {sample_shape}")
    # logger.log(f"参数 sample_shape 的组成包括: 批量大小 batch_size=x_start.shape[0], 属性部分之外的序列长度 seq_len-num_props+1, 每个样本的特征维度 hidden dimension")
    """
    执行采样过程
    """
    # logger.log("\n\n", "#"*50, "\n### 执行采样过程", "\n")
    samples = sample_fn(
        model,  # 调用 create_model_and_diffusion 方法创建 Transformer 模型, 并加载了之前训练过程最终得到的参数权重文件
        sample_shape,  # sample_shape = (x_start.shape[0], args.seq_len-args.num_props+1, args.hidden_dim)
        noise=x_noised,  # 将 x_start 中格式化的 SMILES 部分及之后的填充部分用噪声替换, 性质与骨架部分保留原始值, 得到 x_noised
        clip_denoised=args.clip_denoised,  # 默认设置为 False
        # 嵌入层 model_emb 初始化自 torch.nn.Embedding, 并将训练过程得到的的嵌入层权重 model.word_embedding.weight 克隆至 model_emb
        # denoised_fn_round 函数定义在 round.py 文件中, 就是该扩散模型去噪过程所需要遵循的方法
        # 该函数通过计算模型嵌入 (model_emb) 和文本嵌入 (text_emb) 之间的 K 最近邻 (KNN), 从而选择最相似的嵌入
        # 返回更新后的嵌入, 以便在生成或去噪过程中使用
        # partial 函数: 来自 functools 模块的一个函数, 用于固定某些参数并返回一个新的函数
        # 在这里, denoised_fn 被定义为调用 denoised_fn_round 函数的一个固定版本, 其中 args 和 model_emb 被预先填充
        # 这使得在调用 sample_fn 时可以直接使用该函数，而无需每次都传递这些参数
        denoised_fn=partial(denoised_fn_round, args, model_emb),
        model_kwargs={},
        top_p=args.top_p,  # 默认为 5
        clamp_step=args.clamp_step,  # 默认为 0
        clamp_first=True,
        mask=input_ids_mask,  # 与 x_noised 对应的掩码, 0 对应的保留, 1 对应的替换为高斯噪声
        x_start=x_start,  # 没有被使用随机噪声替换的原始数据
        gap=step_gap,  # 默认为 1
    )
    # logger.log(f"采样结果 samples: {type(samples)}")
    """
    采样后对嵌入格式的张量进行解码处理
    """
    # logger.log("\n\n", "#"*50, "\n### 采样后对嵌入格式的张量进行解码处理", "\n")
    sample = samples[-1]
    # logger.log(f"\n### 采样结果 samples 中表示分子序列的部分 sample=samples[-1]: {type(sample)}, {sample.shape}")
    # 存在属性条件时, 对除属性值之外的值进行处理, 并更新掩码
    # model.get_logits 方法: return self.lm_head(hidden_repr)
    # 其中, self.lm_head = nn.Linear(self.input_dims, vocab_size)
    if args.num_props:
        # logger.log("\n### 存在属性条件时, 切掉属性部分 (sample[:,1:])")
        # logger.log("之后使用 model.get_logits 方法通过线性层 nn.Linear(self.input_dims, vocab_size) 生成 logits")
        # logger.log("并对应地裁剪掩码 input_ids_mask_ori=input_ids_mask_ori[:,1:]")
        logits = model.get_logits(sample[:, 1:])
        input_ids_mask_ori = input_ids_mask_ori[:, 1:]
    else:
        # logger.log("\n### 不存在属性条件时, 直接使用 model.get_logits 方法通过线性层 nn.Linear(self.input_dims, vocab_size) 生成 logits")
        logits = model.get_logits(sample)
    # logger.log(f"logits:\n{type(logits)}, {logits.shape}\ninput_ids_mask_ori:\n{type(input_ids_mask_ori)}, {input_ids_mask_ori.shape}")
    cands = torch.topk(logits, k=1, dim=-1)
    # logger.log(f"\n### cands:\n{type(cands)}")
    word_lst_recover = []
    for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
        len_x = len(input_mask) - sum(input_mask).tolist()
        # 正常情况下, 使用下方这一行代码即可
        tokens = tokenizer.decode_token(seq[len_x:])
        # 当仅输入多种属性 num_prop 作为约束时, 使用下方三行代码, 在序列前加入一个 0, 作为序列的开始标志, 否则无法生成合法分子
        # if args.complexity == 0:
        #     new_seq = seq[len_x - 1 :]
        #     new_seq[0] = torch.tensor(0, dtype=torch.long, device="cuda")
        #     tokens = tokenizer.decode_token(new_seq)
        #     # logger.log(type(tokens), tokens)
        # else:
        #     new_seq = seq[len_x + 1 :]
        #     new_seq[0] = torch.tensor(0, dtype=torch.long, device="cuda")
        #     tokens = tokenizer.decode_token(new_seq)
        word_lst_recover.append(tokens)
    smiles += word_lst_recover

    df = pd.DataFrame(smiles, columns=["smiles"])
    if os.path.exists(general_path):
        df.to_csv(general_path, mode="a", header=False, index=False)
    else:
        df.to_csv(general_path, mode="w", index=False)

    logger.log(f"生成分子保存至: {general_path}")
    logger.log("\n本批次耗时: {:.2f}s".format(time.time() - start_t))
