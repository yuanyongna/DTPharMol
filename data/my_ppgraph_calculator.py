import os
import gc
import dgl
import time
import torch
import random
import psutil
import numpy as np
import pandas as pd
import setproctitle
from torch import nn
from tqdm import tqdm
from rdkit import Chem
from typing import Dict
from rdkit import RDConfig
from torch.nn import functional
from rdkit.Chem import ChemicalFeatures


class TransformerEncoder(nn.Module):
    """
    自创的降维模块
    """

    def __init__(self, input_dim=384, output_dim=6):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)  # 嵌入层，降维到128
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        self.fc = nn.Linear(128, output_dim)  # 输出层，最终降维到6

    def forward(self, x):
        x = self.embedding(x)  # 输入形状为 [6, 384]，输出形状为 [6, 128]
        x = x.unsqueeze(0)  # 添加序列维，变为 [1, 6, 128]
        x = self.transformer_encoder(x)  # 输出形状为 [1, 6, 128]
        x = x.mean(dim=1)  # 对序列维取均值，输出形状为 [1, 128]
        x = self.fc(x)  # 输出形状为 [1, 6]
        return x


def cal_dist(mol, start_atom, end_atom):
    """
    cal_dist 函数计算分子中两个原子之间的距离
    它通过广度优先搜索 (BFS) 算法找到起始原子到目标原子的最短路径, 并根据路径上的键类型累加相应的距离值
    """
    list_ = [start_atom]
    seen = set()
    seen.add(start_atom)
    parent = {start_atom: None}
    nei_atom = []
    bond_num = mol.GetNumBonds()  # 获取分子中的键的数量
    while len(list_) > 0:
        # 从 list_ 列表中取出第一个元素, 赋值给变量 vertex, 同时将该元素从 list_ 中删除
        vertex = list_[0]
        del list_[0]
        # 获取原子 vertex 的邻居原子对象, 并使用列表推导式将邻居原子的索引提取出来, 保存在 nei_atom 列表中
        nei_atom = [n.GetIdx() for n in mol.GetAtomWithIdx(vertex).GetNeighbors()]
        # 遍历 nei_atom 列表中的每个原子 w:
        for w in nei_atom:
            if w not in seen:  # 如果 w 不在 seen 集合中, 表示该原子尚未被访问过
                list_.append(
                    w
                )  # 将 w 添加到 list_ 列表中, 表示将其作为下一轮循环的待处理原子
                seen.add(w)  # 将 w 添加到 seen 集合中, 表示标记该原子已被访问
                parent[w] = (
                    vertex  # 在 parent 字典中, 以 w 为键 vertex 为值, 表示 w 的父原子是 vertex
                )
    path_atom = []  # 保存从终点原子 end_atom 到起始原子 start_atom 的路径
    while end_atom is not None:
        path_atom.append(end_atom)
        end_atom = parent[end_atom]  # 将 end_atom 更新为它的父原子
    nei_bond = []  # 保存分子中每个键的信息
    for i in range(bond_num):
        # 将每一个键的类型、起始原子索引和结束原子索引作为一个元组添加到 nei_bond 列表中
        nei_bond.append(
            (
                mol.GetBondWithIdx(i).GetBondType().name,
                mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                mol.GetBondWithIdx(i).GetEndAtomIdx(),
            )
        )
    # 下面这段代码的作用是判断两个键是否共享相同的起始和结束原子
    # 并将符合条件的键信息添加到 bond_collection 列表中，避免重复添加相同的键信息
    bond_collection = []  # 保存路径上的键信息
    for idx in range(len(path_atom) - 1):
        bond_start = path_atom[idx]
        bond_end = path_atom[idx + 1]
        for bond_type in nei_bond:
            # 查询之前使用 mol.GetBondWithIdx(i) 获取到的分子的所有键信息的任意一条键中,
            # 与此处路径上的一个键 (bond_start - bond_end),
            # 是否有共享相同的起始原子和结束原子的情况 (交集的长度等于 2)
            # 其实, 这种情况下就说明这两个键就是同一个键
            # 这一步的本质就是, 在所有的键 nei_bond 中, 找出路径上出现的那些键 bond_collection
            # 目的是确保加入的化学键是符合化学规范的
            if (
                len(
                    list(
                        {bond_type[1], bond_type[2]}.intersection(
                            {bond_start, bond_end}
                        )
                    )
                )
                == 2
            ):
                # 将键信息添加到 bond_collection 列表中
                if [bond_type[0], bond_type[1], bond_type[2]] not in bond_collection:
                    bond_collection.append([bond_type[0], bond_type[1], bond_type[2]])
    dist = 0  # 累积键的距离值
    # 根据键的类型，累加相应的距离值到 dist 中
    for elment in bond_collection:
        if elment[0] == "SINGLE":
            dist = dist + 1
        elif elment[0] == "DOUBLE":
            dist = dist + 0.87
        elif elment[0] == "AROMATIC":
            dist = dist + 0.91
        else:
            dist = dist + 0.78
    return dist


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
        # 如果输入和输出维度不一致，禁用残差连接
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
        # 在 dgl 中, 有时仅是为了计算值并不想改变原始图, 使用 local_scope() 范围时,
        # 任何对节点或边的修改在脱离这个局部范围后将不会影响图中的原始特征值
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
            g.apply_edges(dgl.function.u_add_v("Dh", "Eh", "DEh"))
            # 经过测试, 这里是将 DEh 保存为边特征 edata
            # 将边特征 "DEh" 和 "Ce" 相加, 更新边特征 "e"
            g.edata["e"] = g.edata["DEh"] + g.edata["Ce"]
            # 使用 sigmoid 函数计算边特征的激活值, 并存储在边特征中
            g.edata["sigma"] = torch.sigmoid(g.edata["e"])
            # u_mul_e("Bh", "sigma", "m"): 将节点特征 Bh 与边特征 sigma 相乘, 并将结果存储在边的消息 m 中
            # sum("m", "sum_sigma_h"): 对每个节点接收到的消息进行求和, 结果存储在节点特征 sum_sigma_h 中
            g.update_all(
                dgl.function.u_mul_e("Bh", "sigma", "m"),
                dgl.function.sum("m", "sum_sigma_h"),
            )
            # copy_e("sigma", "m"): 将边特征 sigma 复制到边消息 m 中
            # sum("m", "sum_sigma"): 对每个节点的边消息进行求和, 结果存储在节点特征 sum_sigma 中
            g.update_all(
                dgl.function.copy_e("sigma", "m"), dgl.function.sum("m", "sum_sigma")
            )
            # 将节点特征 Ah 与接收到的消息 sum_sigma_h 进行结合, 得到新的节点特征 h, 为了避免除以零，添加了一个小常数 1e-6
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
            h = functional.relu(h)
            e = functional.relu(e)
            # 如果启用残差连接, 将输入特征与输出特征相加, 保留输入信息
            if self.residual:
                h = h_in + h
                e = e_in + e
            # 对输出特征应用 dropout，以防止过拟合
            h = functional.dropout(h, self.dropout, training=self.training)
            e = functional.dropout(e, self.dropout, training=self.training)
        # 返回更新后的节点特征和边特征
        return h, e


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
        y = x
        for l in range(self.L):  # 循环遍历所有隐藏层
            y = self.FC_layers[l](y)  # 通过当前的全连接层
            y = functional.relu(y)  # 应用 ReLU 激活函数
        y = self.FC_layers[self.L](y)  # 最后一层不应用激活函数，直接输出
        return y  # 返回输出特征


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
            n_layers (int): GatedGCNLayer 的层数, 默认为 4
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
            "sum": dgl.nn.SumPooling(),
            "mean": dgl.nn.AvgPooling(),
            "max": dgl.nn.MaxPooling(),
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
        h, e = self.forward_feature(g, h, e)
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
        hg = self.pool(g, h)
        return self.MLP_layer(hg)


def get_vvs(
    mol,
    MAX_NUM_PP_GRAPHS=6,
    hidden_dim=384,
    v_dim=8,
    e_dim=1,
    remove_dis=False,
    encoder_n_layer=4,
):
    """
    计算单分子药效团的核心代码
    """
    # 对输入的 SMILES 字符串进行处理
    # 清除分子中各原子的同位素信息, 并确保其与分子对象 mol 的表示形式一致
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)

    # 通过 BuildFeatureFactory 构建特征工厂, 并获取分子中的药效团特征列表
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    features = factory.GetFeaturesForMol(mol)

    # 随机提取药效团特征
    all_pharmocophore_features = []
    mapping = {
        "Aromatic": 1,
        "Hydrophobe": 2,
        "PosIonizable": 3,
        "Acceptor": 4,
        "Donor": 5,
        "LumpedHydrophobe": 6,
    }
    for feat in features:
        phar = feat.GetFamily()  # 获取药效团特征类型
        atom_index = tuple(sorted(feat.GetAtomIds()))  # 获取构成该药效团的原子索引列表
        phar_index = mapping.setdefault(
            phar, 7
        )  # 将药效团特征类型映射为 mapping 中定义的索引, 未定义的映射为默认值 7
        all_pharmocophore_features.append([phar_index, atom_index])
    random.shuffle(all_pharmocophore_features)

    # 采用轮盘赌选择 (Roulette Wheel Selection) 或概率选择 (Fitness Proportionate Selection) 方法
    """
    注意, 这段代码有一定的问题:
    一旦选取了 7 个药效团特征, 就会与最终的 vv = v.new_ones((MAX_NUM_PP_GRAPHS, v.shape[1])) * -999 产生冲突
    且这种随机数量的选择意义并不是非常明确
    """
    # 选择随机数量的药效团特征构建药效团
    # num_list = [3, 4, 5, 6, 7]
    # p_list = [0.086, 0.0864, 0.389, 0.495, 0.0273]
    num_list = [4, 5, 6]
    p_list = [0.15, 0.35, 0.5]
    index = int(
        random.random() * len(p_list)
    )  # 随机生成一个初始索引 index，范围在 0 到 n-1 之间
    beta = random.random() * 2.0 * max(p_list)
    while beta > p_list[index]:
        beta -= p_list[index]
        index = (index + 1) % len(p_list)
    num = num_list[index]  # 选择药效团特征的数量
    if len(all_pharmocophore_features) >= num:  # 选择部分药效团特征作为构建药效团的基础
        mol_phco = all_pharmocophore_features[:num]
    else:
        mol_phco = all_pharmocophore_features

    # 对药效团特征的 "多对一" 非法结构进行处理, 药效团特征 ID 不同, 但构成原子完全相同, 即多对一
    for phar_i in range(len(mol_phco)):
        merged_pharmocophore = []
        for phar_j in range(phar_i, len(mol_phco)):
            # 这里对比的条件是: 药效团特征 ID 不同, 但构成原子完全相同, 即多对一
            if (
                mol_phco[phar_i][0] != mol_phco[phar_j][0]
                and mol_phco[phar_i][1] == mol_phco[phar_j][1]
            ):
                if isinstance(mol_phco[phar_i][0], list):
                    merged_pharmocophore = merged_pharmocophore + mol_phco[phar_i][0]
                    merged_pharmocophore.append(mol_phco[phar_j][0])
                else:
                    merged_pharmocophore.append(mol_phco[phar_i][0])
                    merged_pharmocophore.append(mol_phco[phar_j][0])
            else:
                pass
            merged_pharmocophore = list(set(merged_pharmocophore))
            atoms_list = mol_phco[phar_i][1]
        if len(merged_pharmocophore) != 0:
            for x in range(len(mol_phco)):
                if mol_phco[x][1] == atoms_list:
                    mol_phco[x] = [merged_pharmocophore, atoms_list]

    # 将刚才处理后的 mol_phco 去重
    unique_mol_phcos = []
    for phco in mol_phco:
        if phco not in unique_mol_phcos:
            unique_mol_phcos.append(phco)

    # 将药效团特征按照构成原子的原子序数均值进行排序, 规则为从小到大
    sort_index_list = []
    for phco in unique_mol_phcos:
        sort_index = sum(phco[1]) / len(phco[1])
        sort_index_list.append(sort_index)
    sorted_id = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])
    sort_mol_phcos = []
    for index_id in sorted_id:
        sort_mol_phcos.append(unique_mol_phcos[index_id])
    """
    sort_mol_phcos 最终形如:
    [2, (6,)]
    [2, (18,)]
    [4, (23,)]
    [5, (29,)]
    [6, (25, 26, 27, 28, 39, 43,)]
    [1, (33, 34, 35, 36, 37, 38,)]
    前者是药效团类型, 后者是组成该药效团的原子列表
    """

    # 根据药效团特征信息生成一个位置矩阵 position_matrix 用于描述药效团特征之间的相对位置关系
    phar_type = (
        []
    )  # 记录构成药效团的药效团特征的类型, 是一个嵌套列表, 每一个子列表是一个 one-hot 编码列表
    atoms_list = (
        []
    )  # 记录构成药效团的每一个药效团特征的组成原子列表, 是一个嵌套列表, 每一个子列表包含构成对应索引的药效团特征的原子
    atoms_num = (
        []
    )  # 记录构成药效团的每一个药效团特征的组成原子数量, 是一个列表, 每个值表示构成对应索引的药效团特征的原子数量
    position_matrix = np.zeros(
        (len(sort_mol_phcos), len(sort_mol_phcos))
    )  # 初始化位置矩阵
    for phco_i in range(len(sort_mol_phcos)):
        # 获取当前药效团特征的构成原子列表
        phco_i_atoms = list(sort_mol_phcos[phco_i][1])
        atoms_list.append(phco_i_atoms)
        atoms_num.append(len(phco_i_atoms))
        # 对药效团特征的类型索引进行编码
        atoms = sort_mol_phcos[phco_i][0]
        if type(atoms) is not list:
            atoms = [atoms]
        phco_i_type = [0, 0, 0, 0, 0, 0, 0, 0]
        for atom in atoms:
            phco_i_type[atom] = 1
        phar_type.append(torch.HalfTensor(phco_i_type[1:]))
        """
        phar_type: [tensor([0., 0., 0., 1., 0., 0., 0.], dtype=torch.float16),
                    tensor([0., 0., 0., 0., 1., 0., 0.], dtype=torch.float16),
                    tensor([1., 0., 0., 0., 0., 0., 0.], dtype=torch.float16),
                    tensor([0., 0., 0., 1., 0., 0., 0.], dtype=torch.float16),
                    tensor([0., 0., 0., 0., 1., 0., 0.], dtype=torch.float16),
                    tensor([0., 1., 0., 0., 0., 0., 0.], dtype=torch.float16)]
        atoms_list:  [[3], [9], [11, 12, 13, 14, 15, 20], [20], [23], [26]]
        atoms_num:  [1, 1, 6, 1, 1, 1]
        """

        # 计算当前药效团特征 phco_i 与其他药效团特征 phco_j 之间的位置关系
        for phco_j in range(len(sort_mol_phcos)):
            phco_j_atoms = list(sort_mol_phcos[phco_j][1])
            # 两个药效团特征的构成原子完全相同, 则位置关系默认为 0
            if phco_i_atoms == phco_j_atoms:
                position_matrix[phco_i, phco_j] = 0
            # 两个药效团特征的原子索引列表没有交集, 则位置关系需要通过计算原子之间的距离来确定
            elif str(set(phco_i_atoms).intersection(set(phco_j_atoms))) == "set()":
                dist_set = []  # 存储两个药效团特征中所有原子之间的距离
                # 通过双层遍历, 计算两个药效团特征之间每一对原子组合之间的距离
                for atom_i in phco_i_atoms:
                    for atom_j in phco_j_atoms:
                        # cal_dist 函数通过广度优先搜索 (BFS) 算法找到起始原子到目标原子的最短路径
                        # 并根据路径上的键类型累加相应的距离值
                        dist = cal_dist(mol, atom_i, atom_j)
                        dist_set.append(dist)
                min_dist = min(dist_set)  # 获取两个药效团特征之间距离的最小值
                # 根据两个药效团特征的大小 (原子数量) 来确定位置关系
                # 位置关系指的就是 position_matrix[phco_i, phco_j] 矩阵中对应位置应该记录的值
                # 如果两个药效团特征的构成原子数量都为 1, 那么位置关系就是最小距离
                # 否则, 位置关系是最小距离加上两个药效团特征大小中的最大值乘以 0.2
                if max(len(phco_i_atoms), len(phco_j_atoms)) == 1:
                    position_matrix[phco_i, phco_j] = min_dist
                else:
                    position_matrix[phco_i, phco_j] = (
                        min_dist + max(len(phco_i_atoms), len(phco_j_atoms)) * 0.2
                    )
            # 两个药效团特征的原子索引列表有交集但并非完全重合
            else:
                # 遍历两个药效团特征的原子索引列表
                for atom_i in phco_i_atoms:
                    for atom_j in phco_j_atoms:
                        # 一旦出现了相同的原子类型, 那么位置关系为两个药效团特征大小中的最大值乘以 0.2
                        # 如果没有检索到相同原子类型, 则位置关系默认为 0
                        if atom_i == atom_j:
                            position_matrix[phco_i, phco_j] = (
                                max(len(phco_i_atoms), len(phco_j_atoms)) * 0.2
                            )
    """
    position_matrix: [  [ 0.    7.73 11.6  15.46 12.93 20.66]
                        [ 7.73  0.    3.87  7.73  5.2  12.93]
                        [11.6   3.87  0.    5.6   3.07 10.8 ]
                        [15.46  7.73  5.6   0.    2.2   5.2 ]
                        [12.93  5.2   3.07  2.2   0.    6.2 ]
                        [20.66 12.93 10.8   5.2   6.2   0.  ]  ]
    """
    # 获取药效团特征之间边的权重数组, 以及每条边的起始与终点节点药效团特征索引列表
    weights = []  # 边的权重 (药效团之间的距离)
    u_list = []  # 起始节点 (药效团) 索引列表
    v_list = []  # 终点节点 (药效团) 索引列表
    for u in range(position_matrix.shape[0]):
        for v in range(position_matrix.shape[1]):
            if u != v:  # 排除自环 (即 u 和 v 相等的情况)
                u_list.append(u)
                v_list.append(v)
                # 节点 u 和节点 v 之间的权重就是位置矩阵中二者距离的较小值
                if position_matrix[u, v] >= position_matrix[v, u]:
                    weights.append(position_matrix[v, u])
                else:
                    weights.append(position_matrix[u, v])
    u_list_tensor = torch.tensor(u_list)
    v_list_tensor = torch.tensor(v_list)
    """
    weights: [7.73, 11.6, 15.46, 12.93, 20.66, 7.73, 3.87, 7.73, 5.2, 12.93, 11.6, 3.87, 
            5.6000000000000005, 3.0700000000000003, 10.8, 15.46, 7.73, 5.6000000000000005, 
            2.2, 5.2, 12.93, 5.2, 3.0700000000000003, 2.2, 6.2, 20.66, 12.93, 10.8, 5.2, 6.2]
    u_list_tensor: tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 
                            4, 4, 4, 4, 5, 5, 5, 5, 5])
    v_list_tensor: tensor([1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 5, 0, 
                            1, 2, 3, 5, 0, 1, 2, 3, 4])
    """

    # 构建药效团完全图
    g = dgl.graph((u_list_tensor, v_list_tensor))  # 初始化 DGL 图
    """
    g: Graph(num_nodes=6, num_edges=30,
            ndata_schemes={}
            edata_schemes={})
    """

    # 将半精度浮点数类型的边权重 (药效团特征之间的距离) 存储在图的边信息中, 键为 dist
    g.edata["dist"] = torch.HalfTensor(weights)
    """
    g: Graph(num_nodes=6, num_edges=30,
            ndata_schemes={}
            edata_schemes={'dist': Scheme(shape=(), dtype=torch.float16)})
    """

    # 将张量数据类型的节点类型 (药效团特征的类型索引列表) 存储在图的节点信息中, 键为 type
    # torch.stack(list, dim): 沿着一个新维度对输入张量序列进行连接, list 中的每个元素为返回的 tensor 中第 dim 维度的每个元素
    g.ndata["type"] = torch.stack(phar_type)
    """
    g: Graph(num_nodes=6, num_edges=30,
                ndata_schemes={'type': Scheme(shape=(7,), dtype=torch.float16)}
                edata_schemes={'dist': Scheme(shape=(), dtype=torch.float16)})
    """

    # 将半精度浮点数类型的节点大小 (药效团特征的组成原子数量) 存储在图的节点信息中, 键为 size
    g.ndata["size"] = torch.HalfTensor(atoms_num)
    """
    g: Graph(num_nodes=6, num_edges=30,
                ndata_schemes={
                    'type': Scheme(shape=(7,), dtype=torch.float16), 
                    'size': Scheme(shape=(), dtype=torch.float16)
                }
                edata_schemes={'dist': Scheme(shape=(), dtype=torch.float16)})
    其中边的权重信息 (药效团特征之间的距离) dist:
        tensor([ 7.7305, 11.6016, 15.4609, 12.9297, 20.6562,  7.7305,  3.8691,  7.7305,
                    5.1992, 12.9297, 11.6016,  3.8691,  5.6016,  3.0703, 10.7969, 15.4609,
                    7.7305,  5.6016,  2.1992,  5.1992, 12.9297,  5.1992,  3.0703,  2.1992,
                    6.1992, 20.6562, 12.9297, 10.7969,  5.1992,  6.1992], dtype=torch.float16)
    其中节点类型的信息 (药效团特征类型映射索引) type:
        tensor([[0., 1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0.],
                [1., 0., 0., 0., 0., 0., 0.]], dtype=torch.float16)
    其中节点大小的信息 (每个药效团特征的构成原子数量) size:
        tensor([1., 1., 1., 1., 6., 6.], dtype=torch.float16)
    """

    # 合并节点的类型和大小特征
    g.ndata["h"] = torch.cat(
        (g.ndata["type"], g.ndata["size"].reshape(-1, 1)), dim=1
    ).float()

    # 更新边的距离特征
    g.edata["h"] = g.edata["dist"].reshape(-1, 1).float()
    """
    g(type=<class 'dgl.heterograph.DGLGraph'>):
        Graph(
                num_nodes=6, 
                num_edges=30,
                ndata_schemes={
                    'type': Scheme(shape=(7,), dtype=torch.float16), 
                    'size': Scheme(shape=(), dtype=torch.float16), 
                    'h': Scheme(shape=(8,), dtype=torch.float32)
                }
                edata_schemes={
                    'dist': Scheme(shape=(), dtype=torch.float16), 
                    'h': Scheme(shape=(1,), dtype=torch.float32)
                }
        )
    其中边的权重信息 (药效团特征之间的距离) dist (shap=torch.Size([30])):
        tensor([ 7.7305, 11.6016, 15.4609, 12.9297, 20.6562,  7.7305,  3.8691,  7.7305,
                    5.1992, 12.9297, 11.6016,  3.8691,  5.6016,  3.0703, 10.7969, 15.4609,
                    7.7305,  5.6016,  2.1992,  5.1992, 12.9297,  5.1992,  3.0703,  2.1992,
                    6.1992, 20.6562, 12.9297, 10.7969,  5.1992,  6.1992], dtype=torch.float16)
    其中节点类型的信息 (药效团特征类型映射索引) type (shape=torch.Size([6, 7])):
        tensor([[0., 1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0.],
                [1., 0., 0., 0., 0., 0., 0.]], dtype=torch.float16)
    其中节点大小的信息 (每个药效团特征的构成原子数量) size: torch.Size([6])
        tensor([1., 1., 1., 1., 6., 6.], dtype=torch.float16)
    其中节点类型和大小的信息合并后得到 ndata['h'] (shape=torch.Size([6, 8])):
        tensor([[0., 1., 0., 0., 0., 0., 0., 1.],
                [0., 1., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 1., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1., 0., 6.],
                [1., 0., 0., 0., 0., 0., 0., 6.]])
    其中边的距离信息更新后得到 edata['h'] (shape=torch.Size([30, 1])):
        tensor([[ 7.7305, 11.6016, 15.4609, 12.9297, 20.6562,  7.7305,  3.8691,  7.7305,
                    5.1992, 12.9297, 11.6016,  3.8691,  5.6016,  3.0703, 10.7969, 15.4609,
                    7.7305,  5.6016,  2.1992,  5.1992, 12.9297,  5.1992,  3.0703,  2.1992,
                    6.1992, 20.6562, 12.9297, 10.7969,  5.1992,  6.1992]])
    """

    # 线性层将输入维度为 v_dim 的数据映射到 hidden_dim
    v_init = nn.Linear(v_dim, hidden_dim)
    e_init = nn.Linear(e_dim, hidden_dim)

    # 创建一个可学习的参数, 该参数的初始值为从标准正态分布中随机生成的张量, 形状为 (hidden_dim,)
    # 其实就相当于是一个额外的偏置, 帮助模型更好地适应数据, 提升模型的表示能力, 帮助模型更快地收敛
    pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))

    # 创建图编码器
    encoder = GGCNEncoderBlock(
        hidden_dim,
        hidden_dim,
        n_layers=encoder_n_layer,
        dropout=0,
        readout_pooling="max",
        batch_norm=True,
        residual=True,
    )
    v = v_init(g.ndata["h"])
    e = g.edata["h"]
    if remove_dis:
        e = torch.zeros_like(e)
    e = e_init(e)
    # 执行 4 层图神经网络层
    v, e = encoder.forward_feature(g, v, e)
    """
    奇怪的转换:
    PGMG 中由于将一个 batch_size 的所有分子的药效团图融合成一个大图传到此处
    之后执行了分割和填充操作得到该分子自己的药效团图 vv:
    v_list = torch.split(v, pp_graphs.batch_num_nodes().tolist())
    vv = pad_sequence(v_list, batch_first=False, padding_value=-999)
    之后再进行一系列转换
    """

    # 由于药效团特征数量 (v 矩阵的行数) 是随机的, 因此下面需要统一矩阵格式 (强制要求行数为 6, 即最多包含 6 个药效团特征)
    # PGMG 的源代码为 MAX_NUM_PP_GRAPHS=6, 该变量还参与计算药效团图的映射矩阵 mapping, 但此处不需要这个内容
    vv = v.new_ones((MAX_NUM_PP_GRAPHS, v.shape[1])) * -999
    vv[: v.shape[0], :] = v
    vvs = vv + pp_seg_encoding
    """
    在 PGMG 中关于药效团信息的融入:
    SMILES 经过词嵌入层 word_embed=nn.Embedding(vocab_size, hidden_dim) 转换为词向量 x
    然后调整 x 的维度顺序为 (seq, batch, feat)
    之后经过 pos_encoding = PositionalEncoding(hidden_dim, max_len=params['max_len']) 在序列中添加位置信息得到 xt
    最后将药效团信息 vvs 和位置编码后的输入 xt 沿第一个维度 (序列维度) 进行拼接 (torch.cat((pp_mask, input_mask), dim=1))
    """
    return vvs


def calculate_molecular_pharmacophore(
    input_csv,
    output_csv,
    chunksize=50000,
    pp_graph_len=6,
    MAX_NUM_PP_GRAPHS=6,
    hidden_dim=384,
    v_dim=8,
    e_dim=1,
    remove_dis=False,
    encoder_n_layer=4,
    update_freq=100,
) -> Dict[str, str]:
    print("*" * 100)
    setproctitle.setproctitle("shw_ppgraph")
    current_process = psutil.Process(os.getpid())
    print(f"当前进程 ID: {current_process.pid}")
    print(f"当前进程名称: {current_process.name()}")
    print(f"父进程 ID: {current_process.ppid()}")
    print(f"进程状态: {current_process.status()}")
    print(f"进程创建时间: {time.ctime(current_process.create_time())}")
    print(f"内存信息: {current_process.memory_info()}")

    stats = {"total": 0, "success": 0, "failed": 0}
    start_time = time.time()

    # 分块读取文件
    reader = pd.read_csv(input_csv, chunksize=chunksize)
    for chunk_idx, df_chunk in enumerate(reader):
        # 创建原始列的副本
        result_df = df_chunk.copy()
        result_df["flag"] = False
        result_df["ppgraph"] = None  # 初始化为None，失败时设为空列表

        # 添加mol列用于处理
        df_chunk["mol"] = df_chunk["smiles"].apply(
            lambda s: Chem.MolFromSmiles(str(s)) if pd.notnull(s) else None
        )

        total_mols = len(df_chunk)
        pbar = tqdm(
            total=total_mols,
            desc=f"Processing chunk {chunk_idx + 1}",
            unit="mol",
            miniters=update_freq,
            mininterval=1.0,
        )

        # 初始化计数器
        processed_count = 0

        for idx, row in df_chunk.iterrows():
            stats["total"] += 1
            processed_count += 1

            try:
                mol = row["mol"]
                if not mol or mol.GetNumAtoms() == 0:
                    raise ValueError("Invalid molecule object")

                # 计算药效团特征
                vvs = get_vvs(
                    mol,
                    MAX_NUM_PP_GRAPHS=MAX_NUM_PP_GRAPHS,
                    hidden_dim=hidden_dim,
                    v_dim=v_dim,
                    e_dim=e_dim,
                    remove_dis=remove_dis,
                    encoder_n_layer=encoder_n_layer,
                )

                # 降维处理
                transformer_model = TransformerEncoder(output_dim=pp_graph_len)
                ppgraph = transformer_model(vvs).tolist()[0]

                # 更新结果
                result_df.at[idx, "flag"] = True
                result_df.at[idx, "ppgraph"] = ppgraph
                stats["success"] += 1

            except Exception as e:
                # 计算失败时设置空列表
                result_df.at[idx, "ppgraph"] = []
                stats["failed"] += 1
                # 仅记录前5个错误避免日志过大
                if stats["failed"] <= 5:
                    print(f"\nError for SMILES {row['smiles']}: {str(e)[:100]}")

            # 定期更新进度条并释放内存
            if processed_count % update_freq == 0 or processed_count == total_mols:
                pbar.update(processed_count - pbar.n)
                pbar.set_postfix_str(
                    f"Success: {stats['success']}, Failed: {stats['failed']}",
                    refresh=False,
                )
                # 强制垃圾回收
                gc.collect()

        pbar.close()

        # 保存分块结果
        write_header = chunk_idx == 0 or not os.path.exists(output_csv)
        result_df.to_csv(output_csv, mode="a", header=write_header, index=False)

        # 分块处理完成后释放内存
        del df_chunk, result_df
        gc.collect()

    # 生成统计报告
    total_time = time.time() - start_time
    stats["time"] = f"{total_time:.2f} seconds"
    stats["rate"] = (
        f"{stats['success'] / stats['total'] * 100:.1f}%" if stats["total"] else "N/A"
    )

    return stats


"""
运行指令: nohup python -u data/my_ppgraph_calculator.py >data/my_ppgraph_calculator.log 2>&1 &
监控资源占用: nvitop
监控资源占用: watch -n 2 -d nvidia-smi
"""
if __name__ == "__main__":
    stats = calculate_molecular_pharmacophore(
        input_csv="data/Moses.csv",
        output_csv="data/Moses_ppgraph.csv",
        chunksize=1000,
        MAX_NUM_PP_GRAPHS=8,
        hidden_dim=384,
        v_dim=8,
        e_dim=1,
        remove_dis=False,
        encoder_n_layer=4,
        pp_graph_len=16,
        update_freq=100,
    )

    print(
        f"Calculation Report:\n"
        f"Total Molecules: {stats['total']}\n"
        f"Success Rate: {stats['rate']}\n"
        f"Time Consumed: {stats['time']}"
    )
