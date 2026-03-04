import os
import gc
import re
import dgl
import json
import torch
import time
import psutil
import argparse
import numpy as np
import pandas as pd
import setproctitle
from rdkit import Chem
import torch.nn as nn
import dgl.nn as dglnn
from rdkit import RDLogger
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import torch.distributed as dist
from transformers import set_seed
from dgl import function as dglfn
import diffumol.sascorer as sascorer
from torch.nn import functional as F
from diffumol.utils import dist_util, logger
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from diffumol.smiles_sample import smiles_sample
from diffumol.text_datasets import load_data_text
from diffumol.utils.file_utils import load_phar_file
from data.my_ppgraph import GGCNEncoderBlock, TransformerEncoder
from evaluate.basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "5286dc1a63fbde135489755cc7407102d649be44"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
RDLogger.DisableLog("rdApp.*")


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):
        """
        初始化多层感知机 MLP
        参数:
            input_dim (int): 输入特征的维度
            output_dim (int): 输出特征的维度
            L (int): 隐藏层的数量, 默认值为 2
        """
        super().__init__()
        # 创建全连接层列表，输入维度逐层减半
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        # 添加最后一层，将最后的隐藏层映射到输出维度
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        # 将所有层存储为 nn.ModuleList，便于管理
        self.FC_layers = nn.ModuleList(list_FC_layers)
        # 保存隐藏层的数量
        self.L = L

    def forward(self, x):
        """
        前向传播方法
        参数:
            x (Tensor): 输入特征张量
        返回:
            y (Tensor): 输出特征张量
        """
        # 将输入赋值给 y
        y = x
        # 循环遍历所有隐藏层
        for l in range(self.L):
            # 通过当前的全连接层
            y = self.FC_layers[l](y)
            # 应用 ReLU 激活函数
            y = F.relu(y)
        # 最后一层不应用激活函数，直接输出
        y = self.FC_layers[self.L](y)
        # 返回输出特征
        return y


class GGCNEncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        out_dim,
        n_layers,
        dropout,
        readout_pooling,
        batch_norm,
        residual,
    ):
        """
        初始化 GGCNEncoderBlock 类
        参数:
            hidden_dim (int): 隐藏层特征的维度
            out_dim (int): 输出特征的维度
            n_layers (int): GatedGCNLayer 的数量 (层数), 默认为 4
            dropout (float): Dropout 的比率, 用于防止过拟合
            readout_pooling (str): 选择读出池化方式, 支持 "sum", "mean", "max"
            batch_norm (bool): 是否使用批量归一化
            residual (bool): 是否使用残差连接
        """
        super().__init__()
        # 创建多个 GatedGCNLayer
        self.layers = nn.ModuleList(
            [
                GatedGCNLayer(hidden_dim, hidden_dim, dropout, batch_norm, residual)
                for _ in range(n_layers)
            ]
        )
        # 根据选择的池化方式初始化池化层
        self.pool = {
            "sum": dglnn.SumPooling(),
            "mean": dglnn.AvgPooling(),
            "max": dglnn.MaxPooling(),
        }[readout_pooling]
        # 创建用于映射到输出维度的 MLP 层
        self.MLP_layer = MLP(hidden_dim, out_dim)

    def forward(self, g, h, e):
        """
        前向传播方法
        参数:
            g (DGLGraph): 输入的图对象, 包含节点和边的数据
            h (Tensor): 节点特征张量
            e (Tensor): 边特征张量
        返回:
            hg (Tensor): 经过读出层的输出特征
        """
        # 特征更新
        h, e = self.forward_feature(g, h, e)
        # 读出特征
        hg = self.readout(g, h)
        return hg

    def forward_feature(self, g: dgl.DGLGraph, h, e):
        """
        特征更新方法，通过多个 GatedGCNLayer 更新节点和边特征
        参数:
            g (DGLGraph): 输入的图对象
            h (Tensor): 节点特征张量
            e (Tensor): 边特征张量
        返回:
            h (Tensor): 更新后的节点特征
            e (Tensor): 更新后的边特征
        """
        # 逐层更新特征
        for conv in self.layers:
            h, e = conv(g, h, e)
        return h, e

    def readout(self, g, h):
        """
        读出方法, 通过池化和 MLP 层获取最终输出
        参数:
            g (DGLGraph): 输入的图对象
            h (Tensor): 更新后的节点特征
        返回:
            hg (Tensor): 经过池化和 MLP 层的最终输出特征
        """
        # 通过池化层进行读出
        hg = self.pool(g, h)
        # 通过 MLP 层映射到输出维度
        return self.MLP_layer(hg)


class GatedGCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        """
        初始化 GatedGCNLayer 类
        参数:
            input_dim (int): 输入特征的维度
            output_dim (int): 输出特征的维度
            dropout (float): Dropout 的比率, 用于防止过拟合
            batch_norm (bool): 是否使用批量归一化
            residual (bool): 是否使用残差连接
        """
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        # # 如果输入和输出维度不一致，禁用残差连接
        if input_dim != output_dim:
            self.residual = False
        # 创建多个全连接线性层 (A, B, C, D, E), 用于不同的特征变换
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        # 创建用于节点特征和边特征的批量归一化层
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):
        """
        前向传播方法
        参数:
            g (DGLGraph): 输入的图对象, 包含节点和边的数据
            h (Tensor): 节点特征张量
            e (Tensor): 边特征张量
        返回:
            h (Tensor): 更新后的节点特征
            e (Tensor): 更新后的边特征
        """
        # 在 dgl 中, 有时仅是为了计算值并不想改变原始图,
        # 使用local_scope() 范围时, 任何对节点或边的修改在脱离这个局部范围后将不会影响图中的原始特征值
        with g.local_scope():
            # 保存输入特征以便后续的残差连接
            h_in = h
            e_in = e
            # 将节点特征和通过线性层变换后的特征存储在图的节点数据中
            g.ndata["h"] = h
            g.ndata["Ah"] = self.A(h)
            g.ndata["Bh"] = self.B(h)
            g.ndata["Dh"] = self.D(h)
            g.ndata["Eh"] = self.E(h)
            # 将边特征和通过线性层变换后的边特征存储在图的边数据中
            g.edata["e"] = e
            g.edata["Ce"] = self.C(e)
            # g.apply_edges(...): 这个方法用于应用自定义的边操作
            # dglfn.u_add_v("Dh", "Eh", "DEh"): 计算每条边的源节点特征 Dh 和目标节点特征 Eh 的和, 并将结果存储在边的数据字典中命名为 "DEh"
            g.apply_edges(dglfn.u_add_v("Dh", "Eh", "DEh"))
            # 经过测试, 这里是将 DEh 保存为边特征 edata
            # 将边特征 "DEh" 和 "Ce" 相加, 更新边特征 "e"
            g.edata["e"] = g.edata["DEh"] + g.edata["Ce"]
            # 使用 sigmoid 函数计算边特征的激活值, 并存储在边特征中
            g.edata["sigma"] = torch.sigmoid(g.edata["e"])
            # u_mul_e("Bh", "sigma", "m"): 将节点特征 Bh 与边特征 sigma 相乘, 并将结果存储在边的消息 m 中
            # sum("m", "sum_sigma_h"): 对每个节点接收到的消息进行求和, 结果存储在节点特征 sum_sigma_h 中
            g.update_all(
                dglfn.u_mul_e("Bh", "sigma", "m"), dglfn.sum("m", "sum_sigma_h")
            )
            # copy_e("sigma", "m"): 将边特征 sigma 复制到边消息 m 中
            # sum("m", "sum_sigma"): 对每个节点的边消息进行求和, 结果存储在节点特征 sum_sigma 中
            g.update_all(dglfn.copy_e("sigma", "m"), dglfn.sum("m", "sum_sigma"))
            # 将节点特征 Ah 与接收到的消息 sum_sigma_h 进行结合, 得到新的节点特征 h
            # 为了避免除以零，添加了一个小常数 1e-6
            g.ndata["h"] = g.ndata["Ah"] + g.ndata["sum_sigma_h"] / (
                g.ndata["sum_sigma"] + 1e-6
            )
            # 提取图卷积后的节点特征和边特征
            h = g.ndata["h"]
            e = g.edata["e"]
            # 如果启用批量归一化, 对节点特征和边特征进行归一化处理
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
            # 使用 ReLU 激活函数对节点特征和边特征进行非线性变换
            h = F.relu(h)
            e = F.relu(e)
            # 如果启用残差连接, 将输入特征与输出特征相加, 保留输入信息
            if self.residual:
                h = h_in + h
                e = e_in + e
            # 对输出特征应用 dropout，以防止过拟合
            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)
        # 返回更新后的节点特征和边特征
        return h, e

    # Python 中的一个特殊方法, 用于定义类的“官方”字符串表示,即 repr() 返回的字符串, 这个表示通常用于调试或记录目的
    def __repr__(self):
        """
        返回类的字符串表示, 用于调试和记录
        返回:
            str: 表示类的字符串, 包括输入和输出通道数
        """
        return "{}(in_channels={}, out_channels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


def create_argparser():
    """
    从文件 general_config_file 和 train_config_file 中读取并加载参数
    返回:
        parser: 参数解析器
    """
    print("*" * 100)
    print(f"加载生成参数配置文件: {general_config_file}")
    with open(general_config_file, "r") as f:
        general_config = json.load(f)
    train_config_file = general_config.get("train_config_file")
    print(f"加载训练参数配置文件: {train_config_file}")
    with open(train_config_file, "r") as f:
        train_config = json.load(f)
    # 注意: general_config 中的设置会覆盖 train_config 中的同名设置
    defaults = train_config.copy()
    defaults.update(general_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def check_novelty(gen_smiles, train_smiles):
    """
    评估生成的 SMILES 是否新颖, 比较生成的 SMILES 列表与训练 SMILES 列表之间的重合情况
    """
    if len(gen_smiles) == 0:
        novel_ratio = 0.0
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel * 100.0 / len(gen_smiles)
    return novel_ratio


def canonic_smiles(smiles_or_mol):
    """
    将输入的 SMILES 字符串或分子对象转换为规范的 SMILES 表示
    参数:
        smiles_or_mol: str 或 RDKit 分子对象
    返回:
        str: 规范的 SMILES 表示, 如果输入无效, 则返回 None
    """
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_mol(
    smiles_or_mol, mask_token="[MASK]", candidates=["C", "O", "N", "F", "P", "Cl"]
):
    """
    将 SMILES 字符串或分子对象加载到 RDKit 的分子对象中
    若 SMILES 含有 [MASK]，则尝试使用候选原子替换，返回第一个合法分子
    参数:
        smiles_or_mol: str 或 RDKit 分子对象
    返回:
        Mol: RDKit 分子对象, 如果输入无效, 则返回 None
    """
    if isinstance(smiles_or_mol, Chem.Mol):
        return smiles_or_mol
    if not isinstance(smiles_or_mol, str) or len(smiles_or_mol.strip()) == 0:
        return None
    if mask_token not in smiles_or_mol:
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)  # 对分子进行检查和处理, 以确保其符合化学规范
            return mol
        except:
            return None
    # 含有 [MASK]，尝试候选原子替换
    for replacement in candidates:
        test_smiles = smiles_or_mol.replace(mask_token, replacement)
        mol = Chem.MolFromSmiles(test_smiles)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                return mol
            except:
                continue  # 尝试下一个替代原子
    return None  # 所有候选原子都失败


@torch.no_grad()
def main():
    logger.configure()
    """
    解析命令行参数为 args, 设置随机种子, 设置分布式进程组,
    配置日志记录的设置, 包括日志目录、格式、通信对象等
    """
    print("*" * 100)
    setproctitle.setproctitle("shw_general")
    current_process = psutil.Process(os.getpid())
    print(f"当前进程 ID: {current_process.pid}")
    print(f"当前进程名称: {current_process.name()}")
    print(f"父进程 ID: {current_process.ppid()}")
    print(f"进程状态: {current_process.status()}")
    print(f"进程创建时间: {time.ctime(current_process.create_time())}")
    print(f"内存信息: {current_process.memory_info()}")
    args = create_argparser().parse_args()
    if not args.phar_path:
        args.ppgraph_len = 0

    dist_util.setup_dist()
    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0
    set_seed(args.seed)

    """
    创建模型 model、diffusion 和 model_emb, 以及一个 Tokenizer 类
    其中 model 继承自 Transformer, 加载了 model_path 文件中的权重参数
    diffusion 继承自 SpacedDiffusion, 仅初始化
    model_emb 使用 torch.nn.Embedding 进行初始化, 并加载 model.word_embedding.weight 权重参数
    """
    print("*" * 100)
    print("创建模型 model 并加载权重参数")
    model_kwargs = args_to_dict(
        args, load_defaults_config(args.train_config_file).keys()
    )
    model_kwargs["num_props"] = len(args.props)
    model_kwargs["ppgraph_len"] = args.ppgraph_len
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    # 加载权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval().requires_grad_(False).to(dist_util.dev())
    tokenizer = load_tokenizer(args)
    # 将训练过程得到的的嵌入层权重克隆至 model_emb"
    model_emb = (
        torch.nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=args.hidden_dim,
            _weight=model.word_embedding.weight.clone().cpu(),
        )
        .eval()
        .requires_grad_(False)
    )
    print(f"model_emb 模型的参数: {model_emb.state_dict()['weight'].shape}")

    """
    处理药效团约束
    """
    if args.phar_path:
        ppgraphs = load_phar_file(args.phar_path)
        pp_v_init = nn.Linear(args.pp_v_dim, args.hidden_dim)
        pp_e_init = nn.Linear(args.pp_e_dim, args.hidden_dim)
        pp_seg_encoding = nn.Parameter(torch.randn(args.hidden_dim))
        pp_encoder = GGCNEncoderBlock(
            args.hidden_dim,
            args.hidden_dim,
            n_layers=4,
            dropout=0,
            readout_pooling="max",
            batch_norm=True,
            residual=True,
        )
        v = pp_v_init(ppgraphs.ndata["h"])
        e = pp_e_init(ppgraphs.edata["h"])
        v, e = pp_encoder.forward_feature(ppgraphs, v, e)
        vv = v.new_ones((args.MAX_NUM_PP_GRAPHS, v.shape[1])) * -999
        vv[: v.shape[0], :] = v
        # v = vv
        vvs = vv + pp_seg_encoding
        transformer_model = TransformerEncoder(output_dim=args.ppgraph_len)
        vvs_new = transformer_model(vvs).tolist()
        vvs_new = [str(i) for i in vvs_new[0]]
        print(f"药效团表征 vvs (len={len(vvs_new)}): {vvs_new}")
        args.vvs = vvs_new
    else:
        print("未指定药效团约束")

    """
    初始化格式数据集 all_test_data, 将 model_emb 移至 GPU, 
    设置分布式进程以及 all_test_data 的数据迭代器 iterator
    """
    print("*" * 100)
    print("创建格式化数据对象 data_valid, 不传入任何现有数据")
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),
        loop=False,
    )
    all_test_data = []
    idx = 0
    # world_size 是进程的总数量, rank 是当前进程的当前进程的唯一标识符 (排名)
    # 根据当前进程的排名 (rank) 和索引 (idx), 决定是否将数据添加到 all_test_data 列表中
    try:
        while True:
            _, cond = next(data_valid)
            if idx % world_size == rank:
                all_test_data.append(cond)
            idx += 1
    except StopIteration:
        print("迭代读取数据对象 data_valid 中的数据序列完成")

    print(f"将嵌入模型 model_emb 移动至设备 {dist_util.dev()}")
    model_emb.to(dist_util.dev())

    print("设置分布式进程以及 all_test_data 的数据迭代器 iterator:")
    print(
        f"序号 idx: {idx}, 总进程数 world_size: {world_size}, 当前进程的排名 rank: {rank}"
    )
    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})
    iterator = iter(all_test_data)

    """
    构建嵌入后的数据 x_start, 更新掩码 input_ids_mask_ori,
    构建去噪过程的初始数据 x_noised (保留属性与骨架信息, 将 SMILES 与之后的填充部分用随机噪声替换), 
    构建去噪扩散模型 sample_fn 与样本形状参数 sample_shape
    """
    print("*" * 100)
    print("构建数据和去噪扩散模型")
    if args.complexity:
        args.num_props -= 1
    print(f"生成分子将会保存至: {args.output_path}")
    start_t = time.time()
    # 对 iterator 中的每个元素进行迭代
    # 一般来将, all_test_data 列表中的每一个字典都会构成一个 iterator
    # 也就是说, 这里 cond 就是列表中的那唯一的字典, cond 中包含了两个列表 input_ids 和 input_mask
    for index, cond in enumerate(iterator):
        torch.cuda.empty_cache()
        gc.collect()  # 进行显式垃圾回收
        print(f"序号: {index}, 批次大小 = {len(cond['input_ids'])}")
        smiles_sample(
            args,
            world_size,
            index,
            cond,
            model,
            diffusion,
            model_emb,
            tokenizer,
            args.general_path,
            start_t,
        )
    print("采样过程共耗时: {:.2f}s".format(time.time() - start_t))
    df = pd.read_csv(args.general_path)
    smiles = df["smiles"].tolist()
    smiles = list(set(smiles))
    print("采样不重复的分子数量: ", len(smiles))

    """
    检验样本的合理性, 并添加到生成文件
    """
    print("*" * 100)
    print("检验样本的合理性")
    pattern = "\[START\](.*?)\[END\]"
    regex = re.compile(pattern)
    molecules = []
    for smile in smiles:
        temp = regex.findall(smile)
        if len(temp) != 1:
            continue
        completion = temp[0].replace(" ", "")
        mol = get_mol(completion)
        if mol:
            molecules.append(mol)
    print(f"SMILES 的总数: {len(smiles)}")
    print(f"通过校验的分子总数: {len(molecules)}")

    all_dfs = []
    mol_dict = []
    for i in molecules:
        mol_dict.append({"molecule": i, "smiles": Chem.MolToSmiles(i)})
    results = pd.DataFrame(mol_dict)
    # 校验分子指标
    canon_smiles = [canonic_smiles(s) for s in results["smiles"]]
    unique_smiles = list(set(canon_smiles))
    data = pd.read_csv(args.data_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    # 新颖性
    if "Moses" in args.data_name:
        novel_ratio = check_novelty(
            unique_smiles, set(data[data["split"] == "train"]["smiles"])
        )
    else:
        novel_ratio = check_novelty(
            unique_smiles, set(data[data["source"] == "train"]["smiles"])
        )
    # 分子性质
    results["qed"] = results["molecule"].apply(lambda x: QED.qed(x))
    results["sas"] = results["molecule"].apply(lambda x: sascorer.calculateScore(x))
    results["logp"] = results["molecule"].apply(lambda x: Crippen.MolLogP(x))
    results["tpsa"] = results["molecule"].apply(lambda x: CalcTPSA(x))
    # 整体指标
    results["validity"] = np.round(len(results) / (args.sample), 3)
    results["unique"] = np.round(len(unique_smiles) / len(results), 3)
    results["novelty"] = np.round(novel_ratio / 100, 3)
    # 保存
    all_dfs.append(results)
    results = pd.concat(all_dfs)
    # results.to_csv(csv_path, index=False)
    if os.path.exists(args.output_path):
        results.to_csv(args.output_path, mode="a", header=False, index=False)
    else:
        results.to_csv(args.output_path, mode="w", index=False)
    print(f"生成分子保存至: {args.output_path}")
    print("validity: ", np.round(len(results) / (args.sample), 3))
    print("unique: ", np.round(len(unique_smiles) / len(results), 3))
    print("novelty: ", np.round(novel_ratio / 100, 3))


"""
nohup python -u generate.py >Generate.log 2>&1 &
"""
if __name__ == "__main__":
    general_config_file = "general_config.json"
    main()
