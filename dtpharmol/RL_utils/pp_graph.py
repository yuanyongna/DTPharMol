import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import dgl
from dgl import function as dglfn
import dgl.nn as dglnn
from rdkit.Chem import ChemicalFeatures
from rdkit import Chem, RDConfig
import warnings

# 忽略特定的 UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*use of `x.T` on tensors of dimension other than 2.*",
)


MAX_NUM_PP_GRAPHS = 6


def sample_probability(elment_array, plist, N):
    """
    轮盘赌选择 (Roulette Wheel Selection) 或概率选择 (Fitness Proportionate Selection) 方法
    sample_probability 函数根据给定的概率列表 plist, 从元素数组 elment_array 中进行概率抽样, 生成长度为 N 的样本列表 Psample
    """
    Psample = []
    n = len(plist)
    # 随机生成一个初始索引 index，范围在 0 到 n-1 之间
    index = int(random.random() * n)
    # 计算概率列表中的最大值 mw
    mw = max(plist)
    beta = 0.0
    # 通过循环 N 次进行概率抽样
    for _ in range(N):
        # 在每次循环中，更新 beta 为当前值加上一个范围在 0 到 2*mw 之间的随机数
        beta = beta + random.random() * 2.0 * mw
        # 如果 beta 大于当前索引对应的概率值 num_p[index], 说明抽样结果不在当前索引位置:
        # 将 beta 减去 num_p[index], 并将索引向后移动一个位置, 继续寻找下一个位置;
        # 如果不是, 说明抽样结果为当前索引位置, 将该元素 num[index] 添加到 Psample 中, 并退出循环
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])
    return Psample


def six_encoding(atom):
    """
    根据传入的 atom 列表中的元素, 生成一个具有 6 个元素的编码列表 (torch.HalfTensor 类型)
    """
    # 实际上第 0 位不会参与编码，仅有 7 位事实编码
    orgin_phco = [0, 0, 0, 0, 0, 0, 0, 0]
    # 遍历传入的 atom 列表中的元素，将出现的元素的编码位置设为 1
    for j in atom:
        orgin_phco[j] = 1
    # 返回时也剔除了第 0 位无意义编码
    return torch.HalfTensor(orgin_phco[1:])


def cal_dist(mol, start_atom, end_tom):
    """
    cal_dist 函数计算分子中两个原子之间的距离
    它通过广度优先搜索 (BFS) 算法找到起始原子到目标原子的最短路径, 并根据路径上的键类型累加相应的距离值
    """
    list_ = []
    list_.append(start_atom)
    seen = set()
    seen.add(start_atom)
    parent = {start_atom: None}
    nei_atom = []
    # 使用 mol.GetNumBonds() 获取分子中的键的数量
    bond_num = mol.GetNumBonds()
    while len(list_) > 0:
        # 从 list_ 列表中取出第一个元素, 赋值给变量 vertex, 同时将该元素从 list_ 中删除
        vertex = list_[0]
        del list_[0]
        # 使用 mol.GetAtomWithIdx(vertex).GetNeighbors() 获取原子 vertex 的邻居原子对象
        # 并使用列表推导式将邻居原子的索引提取出来, 保存在 nei_atom 列表中
        nei_atom = [n.GetIdx() for n in mol.GetAtomWithIdx(vertex).GetNeighbors()]
        # 遍历 nei_atom 列表中的每个原子索引 w:
        for w in nei_atom:
            # 如果 w 不在 seen 集合中，表示该原子尚未被访问过
            if w not in seen:
                # 将 w 添加到 list_ 列表中，表示将其作为下一轮循环的待处理原子
                list_.append(w)
                # 将 w 添加到 seen 集合中，表示标记该原子已被访问
                seen.add(w)
                # 在 parent 字典中，以 w 为键，vertex 为值，表示 w 的父原子是 vertex
                parent[w] = vertex
    # 列表 path_atom 用于保存从目标原子 end_atom 到起始原子 start_atom 的路径
    path_atom = []
    while end_tom != None:
        # 将 end_atom 添加到 path_atom 列表中
        path_atom.append(end_tom)
        # 将 parent[end_atom] 赋值给 end_atom，即更新 end_atom 为其父原子
        end_tom = parent[end_tom]
    # 创建一个空列表 nei_bond，用于保存分子中每个键的信息
    nei_bond = []
    for i in range(bond_num):
        # 将键的类型、起始原子索引和结束原子索引作为一个元组添加到 nei_bond 列表中
        # mol.GetBondWithIdx(i) 获取第 i 个键的键对象,
        # 使用 .GetBondType().name 获取该键的类型 (如 "SINGLE"、"DOUBLE" 等)
        # 使用 .GetBeginAtomIdx() 和 .GetEndAtomIdx() 获取该键的起始原子和结束原子的索引
        nei_bond.append(
            (
                mol.GetBondWithIdx(i).GetBondType().name,
                mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                mol.GetBondWithIdx(i).GetEndAtomIdx(),
            )
        )
    # 下面这段代码的作用是判断两个键是否共享相同的起始和结束原子
    # 并将符合条件的键信息添加到 bond_collection 列表中，避免重复添加相同的键信息
    # 创建一个空列表 bond_collection，用于保存路径上的键信息
    bond_collection = []
    # 遍历从 0 到 len(path_atom)-2 的索引 idx
    for idx in range(len(path_atom) - 1):
        # bond_start 和 bond_end 分别表示当前处理的键的起始原子和结束原子
        bond_start = path_atom[idx]
        bond_end = path_atom[idx + 1]
        # 进入一个嵌套循环，遍历 nei_bond 列表中的每个键信息 bond_type
        for bond_type in nei_bond:
            # {bond_type[1], bond_type[2]} 集合包含了 bond_type 键的起始原子和结束原子，即 GetBeginAtomIdx() 与 GetEndAtomIdx()
            # {bond_start, bond_end} 包含 path_atom[idx] 到 path_atom[idx + 1] 原子间键的起始原子和结束原子
            # intersection 操作返回两个集合的交集, list 函数将交集转换为列表
            # 如果交集的长度等于 2，表示两个键共享相同的起始和结束原子，进入下面的代码块
            if (
                len(
                    list(
                        set([bond_type[1], bond_type[2]]).intersection(
                            set([bond_start, bond_end])
                        )
                    )
                )
                == 2
            ):
                # bond_type[0] 即键的类型 GetBondType().name
                # 创建一个列表，添加键的类型、起始原子和结束原子
                # 使用 not in 运算符判断该键信息是否已经存在于 bond_collection 列表中
                # 如果不存在，将键信息添加到 bond_collection 列表中
                if [bond_type[0], bond_type[1], bond_type[2]] not in bond_collection:
                    bond_collection.append([bond_type[0], bond_type[1], bond_type[2]])
    # 创建变量 dist 并初始化为 0，用于累积键的距离值
    dist = 0
    # 遍历 bond_collection 列表中的所有键信息 element
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
    # 返回最终两个原子之间的距离值 dist
    return dist


def smiles_code_(smiles, g, e_list):
    """
    根据给定的 SMILES 字符串、图数据和元素嵌套列表, 生成原子在不同元素药效团特征中的编码信息
    """
    smiles = smiles
    dgl = g
    e_elment = e_list
    # 将 SMILES 字符串转换为 RDKit 的分子对象 mol
    mol = Chem.MolFromSmiles(smiles)
    # 获取分子中的原子数 atom_num
    atom_num = mol.GetNumAtoms()
    # 初始化一个由 0 填充的二维 NumPy 数组，形状为 (atom_num, MAX_NUM_PP_GRAPHS)
    # 含义为: 由 atom_num 个原子组成, 每个原子参与了某个药效团特征的构成,
    # 这里的 "某个" 指的是这一行数组中元素值为 1 的索引对应的药效团特征
    smiles_code = np.zeros((atom_num, MAX_NUM_PP_GRAPHS))
    # 遍历 e_elment 列表中的子列表
    for elment_i in range(len(e_elment)):
        elment = e_elment[elment_i]
        # 遍历当前子列表中的每个元素
        for e_i in range(len(elment)):
            # 获取具体的原子索引
            e_index = elment[e_i]
            # 循环遍历分子中的每个原子, 定位这个原子 e_index 在分子 mol 中的索引
            for atom in mol.GetAtoms():
                # 检查当前原子的索引 e_index 是否与目标原子 atom 的索引相同
                if e_index == atom.GetIdx():
                    # 获取图数据 dgl 中对应子列表 elment_i 索引对应的的类型信息，并将其转换为列表
                    list_ = ((dgl.ndata["type"])[elment_i]).tolist()
                    # 循环遍历类型列表 list_
                    for list_i in range(len(list_)):
                        # 检查当前类型是否为 1
                        if list_[list_i] == 1:
                            # 将对应的 smiles_code 数组位置设置为 1.0，表示该原子在特定元素的药效团特征中
                            smiles_code[atom.GetIdx(), elment_i] = 1.0
    # 返回生成的 smiles_code 数组
    return smiles_code


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


def pp_graph(smiles: str):

    # print("正在计算药效团: ", smiles)

    # 对输入的 SMILES 字符串进行处理, 清除分子中各原子的同位素信息, 并确保其与分子对象 mol 的表示形式一致
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    # print(f"清除同位素信息后的 SMILES 是否与原始 SMILES 依然一致: {Chem.MolToSmiles(mol) == smiles}")
    # print(f"原子数量 atom_num: {mol.GetAtoms()}")

    pharmocophore_all = []
    # 通过 ChemicalFeatures.BuildFeatureFactory 函数构建特征工厂
    # 并通过 factory.GetFeaturesForMol(mol) 函数获取分子中的特征列表
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    # for feat in feats: print(feat.GetFamily())

    for f in feats:
        # 对于每个特征 f, 获取其药效团特征类型 phar 和包含的原子索引 atom_index
        phar = f.GetFamily()
        atom_index = f.GetAtomIds()
        # 将原子索引排序并转换为元组 atom_index
        # 以及使用字典 mapping 将药效团特征类型映射为相应的索引, 未定义的映射为默认值 7, 结果为 phar_index
        atom_index = tuple(sorted(atom_index))
        mapping = {
            "Aromatic": 1,
            "Hydrophobe": 2,
            "PosIonizable": 3,
            "Acceptor": 4,
            "Donor": 5,
            "LumpedHydrophobe": 6,
        }
        phar_index = mapping.setdefault(phar, 7)
        # 将药效团特征类型和原子索引以列表形式存储至 pharmocophore_all
        pharmocophore_ = [phar_index, atom_index]
        pharmocophore_all.append(pharmocophore_)
    # 对提取的药效团特征进行乱序操作
    random.shuffle(pharmocophore_all)
    # print(type(pharmocophore_all))
    # for phar in pharmocophore_all: print(phar)

    # 使用 sample_probability 函数从 num 中按照 num_p 的概率随机采样 N 个样本, 并将结果赋值给 num_
    # 这里指的是该分子的药效团包含 num_ 个药效团特征
    num = [3, 4, 5, 6, 7]
    num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]
    # num_ = sample_probability(num, num_p, 1)
    num_ = [6]
    # print(num_)

    # 如果药效团特征的数量大于限定的数量 num_[0], 那么就截取保留前 num_[0] 个药效团特征, 否则就全部保留
    if len(pharmocophore_all) >= int(num_[0]):
        mol_phco = pharmocophore_all[: int(num_[0])]
    else:
        mol_phco = pharmocophore_all
    # print(type(mol_phco), \n, mol_phco)

    # 对药效团特征的 "多对一" 非法结构进行处理
    for phar_i in range(len(mol_phco)):
        merged_pharmocophore = []
        for phar_j in range(phar_i, len(mol_phco)):
            # 这里对比的条件是：药效团特征 ID 不同, 但构成原子完全相同, 即多对一
            if (
                mol_phco[phar_i][0] != mol_phco[phar_j][0]
                and mol_phco[phar_i][1] == mol_phco[phar_j][1]
            ):
                # print("出现多对一情况的药效团特征:\n", mol_phco[phar_i], mol_phco[phar_j])
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
            # print("将多对一进行合并之后的药效团特征 ID:", merged_pharmocophore)
            # print("对应的构成原子:", atoms_list, "\n")
            for x in range(len(mol_phco)):
                if mol_phco[x][1] == atoms_list:
                    mol_phco[x] = [merged_pharmocophore, atoms_list]
    # print("处理后的药效团特征列表为:")
    # for i in mol_phco: print(i)

    # 将刚才处理后的 mol_phco 去重
    unique_mol_phcos = []
    for phco in mol_phco:
        if phco not in unique_mol_phcos:
            unique_mol_phcos.append(phco)
    # print("去重后的药效团特征列表为:")
    # for i in unique_mol_phcos: print(i)
    # 以下实现药效团特征按构成原子序数的平均值排序, 规则为从小到大
    sort_index_list = []
    # 对于某一个药效团特征对象，计算参与原子索引的均值并以此构建列表 sort_index_list
    for unique in unique_mol_phcos:
        sort_index = sum(unique[1]) / len(unique[1])
        sort_index_list.append(sort_index)
    # print("每个药效团特征的构成原子的原子序数均值 sort_index_list: ", sort_index_list)
    # range(len(sort_index_list)) 会生成一个包含 sort_index_list 长度范围内整数的列表 [0，1，2，···，len(sort_index_list)-1]
    # sorted 对该列表进行排序
    # key=lambda k: sort_index_list[k] 指定了排序的关键函数，
    # 这里使用了一个匿名函数 lambda，它接受一个参数 k，并返回 sort_index_list[k] 的值作为排序的依据
    # 这意味着排序将根据 sort_index_list 中对应索引位置的元素进行排序
    # 即从小到大，按照 sort_index_list 中元素的索引进行排序
    # 例如: sort_index_list: [35.0, 9.0, 19.5, 8.0]，则 sorted_id: [3, 1, 2, 0]
    sorted_id = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])
    # print("按照均值赋予对应的位置索引 sorted_id: ", sorted_id)
    # 将 unique_mol_phcos 中的元素按照原子索引均值从小到大依次添加到 unique_index_filter_sort 中
    sort_mol_phcos = []
    for index_id in sorted_id:
        sort_mol_phcos.append(unique_mol_phcos[index_id])
    # print("排序后的药效团特征列表 sort_mol_phcos:")
    # for i in sort_mol_phcos: print(i)

    # 下面这段代码的功能是根据 sort_mol_phcos 列表中的药效团特征信息，
    # 生成一个位置矩阵 position_matrix，用于描述药效团特征之间的相对位置关系
    type_list = []
    size_ = []
    e_list = []
    # 初始化位置矩阵
    position_matrix = np.zeros((len(sort_mol_phcos), len(sort_mol_phcos)))
    # 遍历列表中的每个药效团特征
    for mol_phco_i in range(len(sort_mol_phcos)):
        # 获取当前药效团特征 mol_phco_i 的构成原子索引列表 mol_phco_i_elment
        mol_phco_i_elment = list(sort_mol_phcos[mol_phco_i][1])
        # 调用 six_encoding() 函数对药效团特征索引进行编码，将结果添加至 type_list
        # six_encoding 函数根据传入列表中的元素，生成一个 0-1 编码列表 (torch.HalfTensor 类型)
        # 因为之前定义了 7 种药效团特征, 因此 type_list 长度为 7
        # mapping = {"Aromatic": 1, "Hydrophobe": 2, "PosIonizable": 3, "Acceptor": 4, "Donor": 5, "LumpedHydrophobe": 6}
        # phar_index = mapping.setdefault(phar, 7)
        if type(sort_mol_phcos[mol_phco_i][0]) == list:
            type_list.append(six_encoding(sort_mol_phcos[mol_phco_i][0]))
        else:
            type_list.append(six_encoding([sort_mol_phcos[mol_phco_i][0]]))
        # 将当前药效团特征的构成原子的数量添加至 size_
        size_.append(len(mol_phco_i_elment))
        # 将当前药效团特征的构成原子索引列表添加至 e_list
        e_list.append(mol_phco_i_elment)
        # 到此，得到例如：
        # type_list:
        # [tensor([0., 0., 0., 1., 0., 0., 0.], dtype=torch.float16),
        # tensor([0., 0., 0., 0., 1., 0., 0.], dtype=torch.float16),
        # tensor([1., 0., 0., 0., 0., 0., 0.], dtype=torch.float16),
        # tensor([0., 0., 0., 1., 0., 0., 0.], dtype=torch.float16),
        # tensor([0., 0., 0., 0., 1., 0., 0.], dtype=torch.float16),
        # tensor([0., 1., 0., 0., 0., 0., 0.], dtype=torch.float16)]
        # size_:  [1, 1, 6, 1, 1, 1]
        # e_list:  [[3], [9], [11, 12, 13, 14, 15, 20], [20], [23], [26]]
        # 接下来内层循环遍历 sort_mol_phcos 列表中的每个药效团特征，计算当前药效团特征与其他药效团特征之间的位置关系
        for mol_phco_j in range(len(sort_mol_phcos)):
            # 获取当前药效团特征 mol_phco_j 的构成原子索引列表 mol_phco_j_elment
            mol_phco_j_elment = list(sort_mol_phcos[mol_phco_j][1])
            # 如果两个药效团特征的原子索引列表完全相同，那么它们的位置关系 position_matrix[mol_phco_i, mol_phco_j] = 0
            if mol_phco_i_elment == mol_phco_j_elment:
                position_matrix[mol_phco_i, mol_phco_j] = 0
            # 如果两个药效团特征的原子索引列表没有交集，那么它们之间的位置关系需要通过计算原子之间的距离来确定
            elif (
                str(set(mol_phco_i_elment).intersection(set(mol_phco_j_elment)))
                == "set()"
            ):
                # 初始化空列表用于存储两个药效团中所有原子之间的距离
                dist_set = []
                for atom_i in mol_phco_i_elment:
                    for atom_j in mol_phco_j_elment:
                        # cal_dist 函数计算分子 mol 中两个原子 atom_i 和 atom_j 之间的距离
                        # 它通过广度优先搜索 (BFS) 算法找到起始原子到目标原子的最短路径，并根据路径上的键类型累加相应的距离值
                        dist = cal_dist(mol, atom_i, atom_j)
                        dist_set.append(dist)
                # min_dist 记录了 dist_set 中的最小距离
                min_dist = min(dist_set)
                # 然后根据两个药效团特征的大小 (原子数量) 来确定位置关系
                # 如果两个药效团特征的构成原子数量都为 1，那么位置关系为最小距离；
                # 否则，位置关系为最小距离加上两个药效团特征大小中的最大值乘以 0.2
                # 将这个位置关系值赋给 position_matrix[mol_phco_i, mol_phco_j]
                if max(len(mol_phco_i_elment), len(mol_phco_j_elment)) == 1:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist
                else:
                    position_matrix[mol_phco_i, mol_phco_j] = (
                        min_dist
                        + max(len(mol_phco_i_elment), len(mol_phco_j_elment)) * 0.2
                    )
            # 如果两个药效团特征的原子索引列表有交集但并非完全重合
            else:
                # 遍历两个药效团特征的原子索引列表
                for type_elment_i in mol_phco_i_elment:
                    for type_elment_j in mol_phco_j_elment:
                        # 一旦出现了相同的原子类型，那么位置关系为两个药效团特征大小中的最大值乘以 0.2
                        if type_elment_i == type_elment_j:
                            position_matrix[mol_phco_i, mol_phco_j] = (
                                max(len(mol_phco_i_elment), len(mol_phco_j_elment))
                                * 0.2
                            )

    # print(f"药效团特征之间的位置矩阵 position_matrix:\n{position_matrix}")
    # print(f"\n每个药效团特征的种类映射 type_list:")
    # for i in type_list: print(i)
    # print(f"\n每个药效团特征的大小 (即构成原子的数量) size_:\n{size_}")

    # 初始化 weights、u_list 和 v_list 用于存储边的权重、起始节点索引和目标节点索引
    weights = []
    u_list = []
    v_list = []
    # 使用嵌套循环遍历位置矩阵的每一个元素。u 表示行索引，v 表示列索引
    for u in range(position_matrix.shape[0]):
        for v in range(position_matrix.shape[1]):
            # 条件 if u != v 用于排除自环 (即 u 和 v 相等的情况)
            if u != v:
                # 将当前的 u 和 v 分别添加到起始节点列表 u_list 和目标节点列表 v_list
                u_list.append(u)
                v_list.append(v)
                # 根据位置矩阵中节点 u 和节点 v 之间的距离，选择较小的距离作为边的权重，并将其添加到 weights 列表中
                if position_matrix[u, v] >= position_matrix[v, u]:
                    weights.append(position_matrix[v, u])
                else:
                    weights.append(position_matrix[u, v])
    # print(f"药效团特征之间的边的权重 weights:\n{weights}")
    # 将 u_list 和 v_list 转换为 PyTorch 的张量（Tensor），以便于后续处理
    u_list_tensor = torch.tensor(u_list)
    # print(f"\n起始节点药效团特征列表 u_list_tensor:\n{u_list_tensor}")
    v_list_tensor = torch.tensor(v_list)
    # print(f"\n终点节点药效团特征列表 v_list_tensor:\n{v_list_tensor}")

    g = dgl.graph((u_list_tensor, v_list_tensor))
    # print(f"使用 u_list_tensor 和 v_list_tensor 初始化 DGL 图 g:\n{g}")
    # 将边的权重 (距离) 存储在图的边数据中，键为 "dist", 这里使用了 torch.HalfTensor，表示使用半精度浮点数
    g.edata["dist"] = torch.HalfTensor(weights)
    # print(f"\n将边的权重信息 (药效团特征之间的距离) 存储在图的边数据中，键为 'dist', 此时的图 g:\n{g}")
    # 将 type_list (节点类型的列表) 转换为张量，使用 torch.stack 函数将其堆叠成一个新的张量 type_list_tensor
    type_list_tensor = torch.stack(type_list)
    # 将节点类型存储在图的节点数据中，键为 "type"
    g.ndata["type"] = type_list_tensor
    # print(f"\n将节点类型信息 (药效团特征类型映射索引) 存储在图的节点数据中, 键为 'type', 此时的图 g:\n{g}")
    # 将节点大小信息存储在图的节点数据中，键为 "size"，同样使用半精度浮点数
    g.ndata["size"] = torch.HalfTensor(size_)
    # print(f"\n将节点大小信息 (每个药效团特征的构成原子数量) 存储在图的节点数据中, 键为 'size', 此时的图 g:\n{g}")
    # print(f"\n其中边的权重信息 (药效团特征之间的距离) dist:\n{g.edata['dist']}")
    # print(f"\n其中节点类型的信息 (药效团特征类型映射索引) type:\n{g.ndata['type']}")
    # print(f"\n其中节点大小的信息 (每个药效团特征的构成原子数量) size:\n{g.ndata['size']}")
    # smiles_code_ 函数根据给定的 SMILES 字符串、图数据和元素嵌套列表，生成原子在不同元素药效团特征中的编码信息 smiles_code_res
    smiles_code_res = smiles_code_(smiles, g, e_list)
    # print(f"\n每个原子参与构成的药效团特征的映射矩阵 (每一行代表一个原子, 每一列代表一种药效团特征, "
    #     f"8 列是因为使用了 MAX_NUM_PP_GRAPHS 参数) smiles_code_res (shape={smiles_code_res.shape}):\n{smiles_code_res}")

    pp_graph = g
    mapping = smiles_code_res

    # 合并节点的类型和大小特征
    pp_graph.ndata["h"] = torch.cat(
        (pp_graph.ndata["type"], pp_graph.ndata["size"].reshape(-1, 1)), dim=1
    ).float()
    # 更新边距离特征
    pp_graph.edata["h"] = pp_graph.edata["dist"].reshape(-1, 1).float()
    # print(f"随机药效团完全图 pp_graph (type={type(pp_graph)}):\n{pp_graph}")
    # print(f"\n其中边的权重信息 (药效团特征之间的距离) dist (shap={pp_graph.edata['dist'].shape}):\n{pp_graph.edata['dist']}")
    # print(f"\n其中节点类型的信息 (药效团特征类型映射索引) type (shape={pp_graph.ndata['type'].shape}):\n{pp_graph.ndata['type']}")
    # print(f"\n其中节点大小的信息 (每个药效团特征的构成原子数量) size: {pp_graph.ndata['size'].shape}\n{pp_graph.ndata['size']}")
    # print(f"\n其中节点类型和大小的信息合并后得到 ndata['h'] (shape={pp_graph.ndata['h'].shape}):\n{pp_graph.ndata['h']}")
    # print(f"\n其中边的距离信息更新后得到 edata['h'] (shape={pp_graph.edata['h'].shape}):\n{pp_graph.edata['h'].T}")

    # 将 mapping 转换为 FloatTensor
    # mapping = torch.FloatTensor(mapping)
    # 将映射矩阵中超过药效团特征数量的那几列全部设置为 -100
    # 默认情况下 Torch Cross Entropy Loss 计算时会忽略 -100
    # mapping[:,pp_graph.num_nodes():] = -100
    # print(f"\nmapping (shape={mapping.shape}):\n{mapping}")
    # 创建一个新的映射张量 mapping_, 默认值为 -100
    # mapping_ = torch.ones(target_seq.shape[0], MAX_NUM_PP_GRAPHS) * -100
    # 根据原子索引 atom_idx 将对应的映射值填入, target_seq 应该比 atom_num 更大
    # 本例中, target_seq 长度为 46, atom_num 为 27
    # atom_idx 其实就是 [1, 2, 3, ······, target_seq.shape[0]]
    # print(f"\natom_idx (len={len(atom_idx)}):\n{atom_idx}")
    # 得到的 mapping_ 其实是每个 token 参与构成的药效团特征的映射索引, 相对应的, mapping 是每个原子参与构成的药效团特征的映射索引
    # mapping_[atom_idx,:] = mapping
    # print(f"\n每个 token 参与构成的药效团特征的映射矩阵 mapping_ (type={type(mapping_)}, shape={mapping_.shape}):\n{mapping_}")
    # tokens = [tokenizer.vocabs[idx] for idx in target_seq.tolist()]
    # print(f"\n目标分子的 token 序列:\n{tokens}")
    # print(f"\n输入分子的张量表示 corrupted_input (type={type(corrupted_input)}, shape={corrupted_input.shape}):\n{corrupted_input}")
    # print(f"\n目标分子的张量表示 target_seq (type={type(target_seq)}, shape={target_seq.shape}):\n{target_seq}")

    """
    到此为止, 我们得到了:
    随机药效团完全图 pp_graph (DGLGraph 格式)
    每个 token 与药效团特征的映射关系矩阵 mapping_
    输入分子的张量表示 corrupted_input
    目标分子的张量表示 target_seq
    """

    # pad_token = Tokenizer.SPECIAL_TOKENS.index("<pad>")
    # print(f"填充标记 <pad> 对应的索引为: {pad_token}")
    # seq_len = 100

    # print(f"\n填充前的输入序列 corrupted_inputs (len={len(corrupted_input)}): {corrupted_input}")
    # if corrupted_input.size(0) < seq_len:
    #     padding_size = seq_len - corrupted_input.size(0)
    #     inputs = torch.cat([corrupted_input, torch.full((padding_size, *corrupted_input.size()[1:]), pad_token)])
    # else:
    #     inputs = corrupted_input[:seq_len]
    # print(f"\n填充后的输入序列 inputs (shape={inputs.shape}): {inputs}")
    # input_mask = (inputs==pad_token).bool()
    # print(f"\n填充后的掩码张量 input_mask (shape={input_mask.shape}): {input_mask}")

    MODEL_DEFAULT_SETTINGS = {
        "max_len": 128,  # 生成的 SMILES 的最大长度
        "pp_v_dim": 7 + 1,  # 药效团嵌入向量的维度
        "pp_e_dim": 1,  # 药效团图边嵌入向量的维度
        "pp_encoder_n_layer": 4,  # 药效团 GNN 层数
        "hidden_dim": 384,  # 隐藏维度
        "n_layers": 8,  # Transformer 编码器和解码器的层数
        "ff_dim": 1024,  # 用于 Transformer 块的 ff dim
        "n_head": 8,  # Transformer 块的注意力头数量
        "remove_pp_dis": False,  # boolean, 如果为 True，则忽略药效团图中的任何空间信息
        "non_vae": False,  # boolean, 如果设为 True，则禁用 VAE 框架
        "in": "rs",  # 是否使用随机输入 SMILES
        "out": "rs",  # 是否使用随机目标 SMILES
    }
    params = dict(MODEL_DEFAULT_SETTINGS)
    hidden_dim = params["hidden_dim"]
    remove_pp_dis = params.setdefault("remove_pp_dis", False)

    pp_encoder = GGCNEncoderBlock(
        hidden_dim,
        hidden_dim,  # 默认为 384
        n_layers=params["pp_encoder_n_layer"],  # 默认为 4
        dropout=0,
        readout_pooling="max",
        batch_norm=True,
        residual=True,
    )

    # 创建线性层, 用于将输入维度为 pp_v_dim/pp_e_dim 的数据映射到 hidden_dim
    pp_v_init = nn.Linear(params["pp_v_dim"], hidden_dim)
    pp_e_init = nn.Linear(params["pp_e_dim"], hidden_dim)
    # 创建一个可学习的参数, 该参数的初始值为从标准正态分布中随机生成的张量, 形状为 (hidden_dim,)
    pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))

    # pp_v_init 是一个线性层, 用于将输入维度 params["pp_v_dim"]=8 的节点特征映射到 hidden_dim 的输出特征
    v = pp_v_init(pp_graph.ndata["h"])
    # print(f"药效团图中的合并后的节点类型和大小信息 pp_graph.ndata['h'] (shape={pp_graph.ndata['h'].shape}):\n{pp_graph.ndata['h']}")
    # print(f"经过线性层变换后得到 v (shape={v.shape}):\n{v}")

    # pp_e_init 是一个线性层, 用于将输入维度 params["pp_e_dim"]=1 的边特征映射到 hidden_dim 的输出特征
    _e = pp_graph.edata["h"]
    if remove_pp_dis:  # 默认为 False
        _e = torch.zeros_like(_e)
    e = pp_e_init(_e)
    # print(f"\n药效团图中的更新后的边距离信息 pp_graph.edata['h'] (shape={_e.shape}):\n{_e}")
    # print(f"经过线性层变换后得到 e (shape={e.shape}):\n{e}")

    # 此处执行 4 层图神经网络层进行处理
    v, e = pp_encoder.forward_feature(pp_graph, v, e)
    # print(f"\n经过若干层 GatedGCNLayer 处理后得到:\nv (shape={v.shape}):\n{v}\n\ne (shape={e.shape}):\n{e}")

    # 将节点特征 v 进行分割并填充为一个固定形状的张量
    # print(f"\n分割时遵循的划分规则 pp_graph.batch_num_nodes().tolist():\n{pp_graph.batch_num_nodes().tolist()}")
    # vv = pad_sequence(torch.split(v, pp_graph.batch_num_nodes().tolist()), batch_first=False, padding_value=-999)
    # print(f"\n划分后得到 vv (shape={vv.shape}):\n{vv}")
    vv = v

    # 创建全为 -999 的张量
    # vv2 = vv.new_ones((MAX_NUM_PP_GRAPHS, pp_graph.batch_size, vv.shape[2])) * -999
    vv2 = vv.new_ones((MAX_NUM_PP_GRAPHS, vv.shape[1])) * -999
    # 将填充后的 vv 填入 vv2
    # vv2[:vv.shape[0], :, :] = vv
    vv2[: vv.shape[0], :] = vv
    # 更新 vv 为填充后的张量
    vv = vv2
    # print(f"\n填充后的 vv (shape={vv.shape}):\n{vv}")

    # 创建掩码, 标记填充位置
    # pp_mask = (vv[:, :, 0].T == -999).bool() # batch, seq
    pp_mask = (vv[:, 0].T == -999).bool()  # seq
    # print(f"\nvv 填充时对应的掩码 pp_mask (shape={pp_mask.shape}):\n{pp_mask}")

    # 将序列特征与位置编码相加
    vvs = vv + pp_seg_encoding  # seq, batch, feat
    # print(f"\n将序列特征与位置编码相加后的 vvs (shape={vvs.shape}):\n{vvs}")

    return vvs


class TransformerEncoder(nn.Module):

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
