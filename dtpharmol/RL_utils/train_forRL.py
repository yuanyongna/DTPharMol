import re
import os
import sys
import json
import torch
import wandb
import psutil
import pandas
import argparse
from rdkit import Chem
from rdkit import RDLogger
from transformers import set_seed
from diffumol.RL_utils import dist_util
from diffumol.train_util import TrainLoop
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Crippen, Descriptors, Lipinski
from diffumol.RL_utils.my_sascorer import my_calculateScore
from evaluate.basic_utils import create_model_and_diffusion
from diffumol.RL_utils.Tokenizer_forRL import Tokenizer_forRL
from diffumol.step_sample import create_named_schedule_sampler
from diffumol.RL_utils.create_dataloader import create_dataloader
from diffumol.RL_utils.create_embedding_model import create_embedding_model

RDLogger.DisableLog("rdApp.*")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "5286dc1a63fbde135489755cc7407102d649be44"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def data_calculate(smiles, scaffold=True, props=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * (
            len(props) + (1 if scaffold else 0)
        )  # 返回 None 值以表示无法计算
    results = []
    # 根据 props 列表计算对应的属性值
    for prop in props:
        if prop == "logp":
            results.append(Crippen.MolLogP(mol))
        elif prop == "qed":
            results.append(Descriptors.qed(mol))
        elif prop == "sas":
            results.append(my_calculateScore(mol))
        elif prop == "tpsa":
            results.append(Descriptors.TPSA(mol))
        elif prop == "mw":
            results.append(Descriptors.MolWt(mol))
        elif prop == "hba":
            results.append(Lipinski.NumHBA(mol))
        elif prop == "hbd":
            results.append(Lipinski.NumHBD(mol))
        elif prop == "rob":
            results.append(Lipinski.RingCount(mol))
        elif prop == "chiral_centers":
            chiral_centers_count = len(Chem.ChiralCenters(mol, includeUnspecified=True))
            results.append(chiral_centers_count)
    if scaffold:
        scaffold_value = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        results.append(scaffold_value)
    return results


def load_defaults_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def train_forRL(
    docking_score_path, epoch_now, model_ckpt, nums, config_file, model_save_dir
):
    print("*" * 100)
    print("进入训练程序")
    current_process = psutil.Process(os.getpid())
    print(f"当前进程 ID: {current_process.pid}")
    print(f"当前进程名称: {current_process.name()}")

    with open(config_file, "r") as f:
        defaults = json.load(f)
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v)
    parser.add_argument("--docking_score_path", default=docking_score_path)
    parser.add_argument("--epoch_now", default=epoch_now, type=int)
    parser.add_argument("--model_ckpt", default=model_ckpt)
    args = parser.parse_args([])  # 传入空列表以避免从命令行读取
    args.checkpoint_path = model_save_dir
    print(type(args), args)
    set_seed(args.seed)
    # args.num_props = len(args.props)
    # if args.complexity:
    #     args.num_props += 1
    # print(f"num_props = {args.num_props}, complexity = {args.complexity}")
    dist_util.setup_dist()

    # 数据读取与初步计算
    # print("*" * 70)
    # print("数据读取与初步计算")
    tokenizer = Tokenizer_forRL(args)
    embedding_model = create_embedding_model(args, tokenizer.vocab_size)
    data = pandas.read_csv(args.docking_score_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data = data[:nums]
    print(f"epoch{epoch_now} 截取的的数据集大小: {len(data)}")

    results = data["smiles"].apply(
        lambda smile: data_calculate(smile, scaffold=args.scaffold, props=args.props)
    )
    # for idx, result in enumerate(results):
    #     print(f"Result {idx}: {result}, Length: {len(result)}")
    property_columns = args.props + (["scaffold_smiles"] if args.scaffold else [])
    results_df = pandas.DataFrame(results.tolist(), columns=property_columns)
    data = pandas.concat([data, results_df], axis=1)
    # print(data)

    # 初步处理数据
    # print("*" * 70)
    # print("数据划分与序列最大长度计算")
    train_size = int(0.8 * len(data))
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_data = data[:train_size]
    val_data = data[train_size:]
    # 获取最大序列长度值
    train_smiles = train_data["smiles"]
    val_smiles = val_data["smiles"]
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    lens = [
        len(regex.findall(i.strip()))
        for i in (list(train_smiles.values) + list(val_smiles.values))
    ]
    max_len = max(lens)
    if args.scaffold:
        train_scaffold = train_data["scaffold_smiles"]
        val_scaffold = val_data["scaffold_smiles"]
        lens = [
            len(regex.findall(i.strip()))
            for i in (list(train_scaffold.values) + list(val_scaffold.values))
        ]
        scaffold_max_len = max(lens)
    else:
        # print("未指定分子骨架约束")
        scaffold_max_len = 0

    if args.scaffold and args.num_props:
        args.seq_len = max_len + scaffold_max_len + args.num_props + 3
        # print(f"同时存在骨架和属性条件时计算 seq_len: {args.seq_len}")
    elif args.scaffold:
        args.seq_len = max_len + scaffold_max_len + 5
        # print(f"仅存在骨架条件时计算 seq_len: {args.seq_len}")
    elif args.num_props:
        args.seq_len = max_len + args.num_props + 3
        # print(f"仅存在属性条件时计算 seq_len: {args.seq_len}")
    else:
        args.seq_len = max_len + 6
        # print(f"骨架和属性条件时都不存在时计算 seq_len: {args.seq_len}")
    print(
        f"训练数据集大小: {len(train_smiles)}, 验证数据集大小: {len(val_smiles)}, 最大序列长度: {args.seq_len}"
    )

    # 构建数据加载器
    # print("*" * 70)
    print("构建数据加载器")
    data_train = create_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data=train_data,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=embedding_model,
    )
    # print(data_train)
    # print("\n", "#"*50, "\n构建验证数据迭代器 data_valid", "\n")
    data_valid = create_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data=val_data,
        data_args=args,
        split="valid",
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=embedding_model,
    )

    # print("*"*70)
    print(f"初始化模型与采样器, 加载模型参数文件 {args.model_ckpt}")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config(config_file).keys())
    )
    # 加载模型参数文件
    ckpt = torch.load(args.model_ckpt, map_location="cuda:0")
    model.load_state_dict(ckpt)
    model.to(dist_util.dev())
    # 构建采样器
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print(f"\n初始化过程的超参数保存至: {args.checkpoint_path}/training_args.json")
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DIFFUMOL"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    # print("*"*70)
    print("\n开始训练过程")
    #  model_savepath = os.path.join(args.checkpoint_path, f"epoch_{args.epoch_now}")
    model_savepath = args.checkpoint_path
    print("model_savepath: ", model_savepath)
    model_path = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_train,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=model_savepath,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        tip_interval=args.tip_interval,
    ).run_loop()
    # print("\n", "#"*50, "\n###全部训练过程结束", "\n")
    return model_path


# 测试
"""
init_docking_score_path = "./RL/ligand_data/init/docking_result.csv"
epoch = 1
initial_model_ckpt_path = "./RL/model_ckpt/ema_0.9999_100000.pt"
train_forRL(init_docking_score_path, epoch, initial_model_ckpt_path)
"""
