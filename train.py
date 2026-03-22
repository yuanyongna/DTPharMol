import os
import re
import json
import wandb
import time
import psutil
import argparse
import setproctitle
import pandas as pd
from diffumol.utils import dist_util, logger
from diffumol.text_datasets import load_data_text
from diffumol.step_sample import create_named_schedule_sampler
from diffumol.train_util import TrainLoop
from transformers import set_seed
from evaluate.basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "5286dc1a63fbde135489755cc7407102d649be44"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def create_argparser():
    """
    从文件 config_file 中读取并加载参数到 defaults,
    创建一个命令行的参数的解析器 parser,
    将文件参数进行相关处理后加入到参数解析器中
    返回:
        parser: 参数解析器
    """
    print("*" * 100)
    print(f"加载训练参数配置文件: {train_config_file}")
    defaults = dict()
    defaults.update(load_defaults_config(train_config_file))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    logger.configure()
    """
    解析命令行参数为 args, 设置随机种子, 设置分布式进程组,
    配置日志记录的设置, 包括日志目录、格式、通信对象等
    """
    print("*" * 100)
    setproctitle.setproctitle("shw_train")
    current_process = psutil.Process(os.getpid())
    print(f"当前进程 ID: {current_process.pid}")
    print(f"当前进程名称: {current_process.name()}")
    print(f"父进程 ID: {current_process.ppid()}")
    print(f"进程状态: {current_process.status()}")
    print(f"进程创建时间: {time.ctime(current_process.create_time())}")
    print(f"内存信息: {current_process.memory_info()}")

    args = create_argparser().parse_args()
    set_seed(args.seed)
    args.num_props = len(args.props)
    if args.complexity:
        args.num_props += 1
    dist_util.setup_dist()

    """
    初始化数据加载器 tokenizer, 初始化嵌入模型 model_weight
    """
    tokenizer = load_tokenizer(args)
    model_weight = load_model_emb(args, tokenizer.vocab_size)

    """
    从 data_path 读取数据集 data, 并处理缺失值、重置索引, 以及将列名变为小写字母, 
    如果 data 来自于 Moses 数据集: 
        选择 split=="train" 的行以构建训练集 train_data, 并重置索引; 
        选择 split=="test" 的行以构建验证集 val_data, 并重置索引
    如果 data 来自于 Guacamol 数据集: 
        选择 split=="train" 的行以构建训练集 train_data, 并重置索引; 
        选择 split=="val" 的行以构建验证集 val_data, 并重置索引
    然后提取训练集与验证集的 SMILES 字符串以及对应分子骨架, 
        训练集提取至 smiles 与 scaffold; 
        验证集提取至 vsmiles 与 vscaffold
    """
    print("*" * 100)
    print(f"加载数据集: {args.data_path}")
    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    if "Moses" in args.data_name:
        train_data = data[data["split"] == "train"].reset_index(drop=True)
        val_data = data[data["split"] == "test"].reset_index(drop=True)
    if "Guacamol" in args.data_name:
        train_data = data[data["source"] == "train"].reset_index(drop=True)
        val_data = data[data["source"] == "val"].reset_index(drop=True)
    print(f"训练集大小: {train_data.shape[0]} rows, {train_data.shape[1]} columns")
    print(f"验证集大小: {val_data.shape[0]} rows, {val_data.shape[1]} columns")
    smiles = train_data["smiles"]
    scaffold = train_data["scaffold_smiles"]
    vsmiles = val_data["smiles"]
    vscaffold = val_data["scaffold_smiles"]
    if args.ppgraph:
        # 获取第一个样本的 ppgraph 长度, 所有样本长度相同
        args.ppgraph_len = len(train_data["ppgraph"][0])
    else:
        args.ppgraph_len = 0

    """
    定义模式并编译为正则表达式, 
    计算所有的 SMILES 字符串 (训练集与验证集加在一起) 能够使用正则表达式匹配到的最大元素数量 max_len; 
    计算所有的骨架字符串能够使用正则表达式匹配到的最大元素数量 scaffold_max_len
    complexity 也视为 num_props 之一
    如果同时提供了 num_props 和 scaffold 信息: 
        args.seq_len = max_len + scaffold_max_len + args.num_props + 3
    如果仅提供了 scaffold 信息: 
        args.seq_len = max_len + scaffold_max_len + 5
    如果仅提供了 num_props 信息: 
        args.seq_len = max_len + args.num_props + 3
    如果未提供任何额外信息: 
        args.seq_len = max_len + 6
    常数是预留值, 以容纳分子表示中的特殊符号或分隔符, 确保模型的输入格式一致
    """
    print("*" * 100)
    print(f"计算参数 seq_len")
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    lens = [
        len(regex.findall(i.strip()))
        for i in (list(smiles.values) + list(vsmiles.values))
    ]
    max_len = max(lens)
    print(f"单个 SMILES token 数量的最大值 max_len: {max_len}")
    lens = [
        len(regex.findall(i.strip()))
        for i in (list(scaffold.values) + list(vscaffold.values))
    ]
    scaffold_max_len = max(lens)
    print(f"单个分子骨架 token 数量的最大值: {scaffold_max_len}")
    if args.num_props and args.scaffold:
        args.seq_len = max_len + scaffold_max_len + args.num_props + 3
    elif args.num_props:
        args.seq_len = max_len + args.num_props + 3
    elif args.scaffold:
        args.seq_len = max_len + scaffold_max_len + 5
    else:
        args.seq_len = max_len + 6
    if args.ppgraph_len:
        args.seq_len = args.seq_len + args.ppgraph_len
    print(
        f"seq_len: {args.seq_len} (props={args.props}, num_props={args.num_props}, complexity={args.complexity}, "
        f"scaffold={args.scaffold}, ppgraph_len={args.ppgraph_len})"
    )

    """
    构建训练数据迭代器 data 和验证数据迭代器 data_valid
    再次声明参数: 
        batch_size: 每次迭代的样本数量
        seq_len: 刚才计算的参数, 序列的最大长度
        data: 使用的具体数据集
        data_args: 数据集相关的参数对象
        loaded_vocab: 实例化的分词器
        model_emb: 实例化的嵌入模型
        split: 数据类型, 默认为 train
        deterministic: 是否打乱数据顺序, 默认为 False 即需要打乱顺序
    """
    print("*" * 100)
    print("构建训练数据迭代器 data")
    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data=train_data,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=model_weight,
        split="train",
        deterministic=False,
    )
    print("*" * 100)
    print("构建验证数据迭代器 data_valid")
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data=val_data,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=model_weight,
        split="valid",
        deterministic=True,
    )

    """
    创建模型 DIFFUMOL
    """
    print("*" * 100)
    print("创建模型 DIFFUMOL")
    model_kwargs = args_to_dict(args, load_defaults_config(train_config_file).keys())
    model_kwargs["num_props"] = len(args.props)
    model_kwargs["ppgraph_len"] = args.ppgraph_len
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.to(dist_util.dev())
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"model 包含的参数总数为: {pytorch_total_params}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print(f"将超参数保存到文件: {args.checkpoint_path}/training_args.json")
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DIFFUMOL"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    """
    执行训练过程
    """
    print("*" * 100)
    print("开始训练")
    TrainLoop(
        model=model,  # 初始化的 TransformerNetModel 类
        diffusion=diffusion,  # 初始化的 SpacedDiffusion 类, 继承自 GaussianDiffusion
        data=data,  # 训练数据迭代器
        batch_size=args.batch_size,  # 每次迭代的样本数量
        microbatch=args.microbatch,  # 每个批次再划分的小批次样本数量
        lr=args.lr,  # 学习率
        ema_rate=args.ema_rate,  # 指数滑动平均
        resume_checkpoint=args.resume_checkpoint,  # 断点接续训练, 默认为 none
        use_fp16=args.use_fp16,  # 使用混合精度, 默认为 false
        fp16_scale_growth=args.fp16_scale_growth,  # 默认为 0.001
        schedule_sampler=schedule_sampler,  # 刚才定义的 lossaware 采样器
        weight_decay=args.weight_decay,  # 权重衰减, 默认为 0.0
        learning_steps=args.learning_steps,  # 迭代次数
        checkpoint_path=args.checkpoint_path,  # 模型保存路径
        gradient_clipping=args.gradient_clipping,  # 梯度裁剪, 默认为 -1.0
        eval_data=data_valid,  # 验证数据迭代器
        log_interval=args.log_interval,  # 日志记录间隔步骤数
        save_interval=args.save_interval,  # 模型保存间隔步骤数
        eval_interval=args.eval_interval,  # 模型验证间隔步骤数
        tip_interval=args.tip_interval,  # 提示运行过程间隔步数
    ).run_loop()
    print("\n训练结束")


"""
运行指令: nohup python -u train.py >Train.log 2>&1 &
监控资源占用: nvitop
监控资源占用: watch -n 2 -d nvidia-smi
"""
if __name__ == "__main__":
    train_config_file = "train_config.json"
    main()
