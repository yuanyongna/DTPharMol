import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
import datasets
from datasets import Dataset as Dataset2
from diffumol.utils import logger
from data import my_ppgraph
from data import my_complexity_calculator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math


torch.set_printoptions(edgeitems=4)


def load_data_text(
    batch_size,
    seq_len,
    data=None,
    deterministic=False,
    data_args=None,
    model_emb=None,
    split="train",
    loaded_vocab=None,
    loop=True,
):
    """
    load_data_text 方法针对数据集, 创建无限迭代器或普通迭代器
    首先调用 get_corpus2 方法, 对数据集 data 进行处理, 得到 training_data,
        其中每一条数据都被执行了 token ID 编码化以及合并与填充,
        包含四部分信息: 属性骨架、SMILES、属性骨架与 SMILES 合并、对应的掩码
    之后使用 training_data 和 model_emb 实例化一个自定义的数据加载器:
        dataset = TextDataset(training_data, data_args, model_emb=model_emb)
        TextDataset 继承自 torch.utils.data.Dataset, 目的是为之后 DataLoader 的数据批量加载做准备
    然后判断数据 data 的类型, 即 split 是否为 test,
    如果不是 test, 说明是训练集或验证集:
        首先使用 DistributedSampler 方法将数据集 dataset 划分出子集, 允许之后每个进程只处理数据集的一部分,
        变量 sampler=DistributedSampler(dataset) 会得到一种 “抽样策略”, 用于在 DataLoader 中支持多进程分布式采样
        之后使用 DataLoader 将 dataset 划分为小批量数据集 data_loader, 此处就使用了刚才的 sampler
    如果是 test, 说明是测试集:
        直接使用 DataLoader 将 dataset 划分为小批量数据集 data_loader, 但加入了数据乱序功能,
        如果 deterministic 为 False, 则打乱数据顺序; 如果为 True, 则不打乱
    最终迭代返回小批量数据集 data_loader
    参数:
        batch_size: 划分小数据集时每个批次中包含样本的数量
        seq_len: 刚才计算的参数, 序列的最大长度
        data: 使用的具体数据集
        data_args: 数据集相关的参数对象
        loaded_vocab: 实例化的分词器
        model_emb: 实例化的嵌入模型
        split: 数据类型, 默认为 train
        deterministic: 是否打乱数据顺序, 默认为 False, 即需要打乱顺序
        loop: 是否循环获取批处理数据
    返回:
        infinite_loader(data_loader) 或 iter(data_loader), 是 data_loader 的无限迭代器或普通迭代器
        data_loader 是一个被划分为小批次的数据集, 包含 input_ids 的嵌入张量 arr、input_ids 自己、input_mask 自己
    """
    training_data = get_corpus2(data_args, seq_len, data, loaded_vocab=loaded_vocab)
    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb,
        vocab_size=loaded_vocab.vocab_size,
    )
    if split != "test":
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not deterministic,
            num_workers=0,
        )
    if loop:
        return infinite_loader(data_loader)
    else:
        return iter(data_loader)


def infinite_loader(data_loader):
    while True:
        yield from data_loader


def helper_tokenize(sentence_lst, vocab_dict, seq_len, data_args):
    """
    对 sentence_lst 中每一对 src 与 trg 进行 token 化编码、合并与掩码、填充
    首先将 sentence_lst 转换为 Dataset2 格式的 raw_datasets,
    然后调用 tokenize_function 子函数, 使用 vocab_dict 对 src 和 trg 进行 token 化编码,
        这其实就是调用了 tokenizer 的 encode_token 函数,
        得到了 ID 编码格式的数据集 tokenized_datasets,
        其中 src 对应 input_id_x, trg 对应 input_id_y
    然后调用 merge_and_mask 方法将 input_id_x 和 input_id_y 合并起来, 并初始化对应的掩码,
        合并后的序列键索引为 input_ids, 掩码键索引为 input_mask
    之后调用 pad_function 方法对 input_ids 和 input_mask 进行填充,
        并对掩码进行赋值, 其中 1 对应的位置表示有效信息位置,
        最大长度为 seq_len, 填充所用的 token ID 为 vocab_dict["[PAD]"]=3
    最后将得到的数据集 (包含 input_id_x、input_id_y、input_ids、input_mask)
        赋给 datasets.DatasetDict() 格式的数据对象 raw_datasets["train"],
        返回整个 raw_datasets
    参数:
        sentence_lst: 包含 src 和 trg 列表的字典
        vocab_dict: 实例化的 tokenizer
        seq_len: 最大序列长度
        data_args: 参数对象
    返回:
        raw_datasets: 一个包含训练数据集的 DatasetDict 对象
    """
    raw_datasets = Dataset2.from_dict(sentence_lst)

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(
            examples["src"],
            ppgraph_len=data_args.ppgraph_len,
            num_props=data_args.num_props,
            scaffold=data_args.scaffold,
        )
        input_id_y = vocab_dict.encode_token(examples["trg"])
        result_dict = {
            "input_id_x": input_id_x,
            "input_id_y": input_id_y,
            "smiles": examples["trg"],
        }
        return result_dict

    # map 方法能够对数据集中的每个元素应用指定的函数
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,  # 以批处理的方式处理数据
        num_proc=1,  # 使用进程的数量
        remove_columns=["src", "trg"],  # 移除原始的 src 和 trg 列
        load_from_cache_file=False,  # 尝试从缓存中加载之前处理的结果
        desc="tokenize",  # 在处理过程中显示进度条
    )

    print("\ntokenized_datasets: ")
    for input_id_x, input_id_y, smiles in list(
        zip(
            tokenized_datasets["input_id_x"],
            tokenized_datasets["input_id_y"],
            tokenized_datasets["smiles"],
        )
    )[:3]:
        print(
            f"input_id_x: {input_id_x} \n——> input_id_y: {input_id_y} \n——> smiles: {smiles}"
        )

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        if data_args.num_props:
            for i in range(len(group_lst["input_id_x"])):
                # 一般 input_id_y 最后的一个 token 编码是结束符
                end_token = group_lst["input_id_y"][i][-1]
                src = group_lst["input_id_x"][i]
                trg = group_lst["input_id_y"][i][:-1]
                # 使用 while 循环确保合并后的序列不超过最大长度 seq_len-2
                # 如果 input_id_x 和 input_id_y 的长度之和超过这个限制, 则依次从较长的句子中去掉最后的 token 以缩短长度
                while len(src) + len(trg) > seq_len - 2:
                    if len(src) > len(trg):
                        src.pop()
                    elif len(src) < len(trg):
                        trg.pop()
                    else:
                        src.pop()
                        trg.pop()
                trg.append(end_token)
                # 合并 input_id_x 和 input_id_y, 并在二者之间添加分隔符 [END] 对应的 ID
                # 一般 [END] 对应的 ID 是 1
                lst.append(src + [vocab_dict.sep_token_id] + trg)
                # 创建掩码, 长度与 len(input_id_x)+1 相同
                mask.append([0] * (len(src) + 1))
        else:
            for i in range(len(group_lst["input_id_x"])):
                end_token = group_lst["input_id_y"][i][-1]
                src = group_lst["input_id_x"][i][:-1]
                trg = group_lst["input_id_y"][i][:-1]
                while len(src) + len(trg) > seq_len - 3:
                    if len(src) > len(trg):
                        src.pop()
                    elif len(src) < len(trg):
                        trg.pop()
                    else:
                        src.pop()
                        trg.pop()
                src.append(end_token)
                trg.append(end_token)
                lst.append(src + [vocab_dict.sep_token_id] + trg)
                mask.append([0] * (len(src) + 1))
        # 将合并后的序列与掩码添加到 group_lst 字典中并返回
        group_lst["input_ids"] = lst
        group_lst["input_mask"] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge_and_mask",
    )
    print("\ntokenized_datasets:")
    for input_id_x, input_id_y, input_ids, input_mask, smiles in list(
        zip(
            tokenized_datasets["input_id_x"],
            tokenized_datasets["input_id_y"],
            tokenized_datasets["input_ids"],
            tokenized_datasets["input_mask"],
            tokenized_datasets["smiles"],
        )
    )[:3]:
        print(
            f"input_id_x (len={len(input_id_x)}): {input_id_x}"
            f"\n——> input_id_y (len={len(input_id_y)}): {input_id_y}"
            f"\n——> input_ids (len={len(input_ids)}): {input_ids}"
            f"\n——> input_mask (len={len(input_mask)}): {input_mask}"
            f"\n——> smiles (len={len(smiles)}): {smiles}"
        )

    def pad_function(group_lst):
        max_length = seq_len
        # _collate_batch_helper: 将所有数据归一化为 max_length 长度的数组, 空白位置用 vocab_dict["[PAD]"] 填充
        # vocab_dict["[PAD]"] 默认为 3
        group_lst["input_ids"] = _collate_batch_helper(
            group_lst["input_ids"], vocab_dict.pad_token_id, max_length
        )
        group_lst["input_mask"] = _collate_batch_helper(
            group_lst["input_mask"], 1, max_length
        )
        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )
    print("\ntokenized_datasets:")
    for input_id_x, input_id_y, input_ids, input_mask, smiles in list(
        zip(
            tokenized_datasets["input_id_x"],
            tokenized_datasets["input_id_y"],
            tokenized_datasets["input_ids"],
            tokenized_datasets["input_mask"],
            tokenized_datasets["smiles"],
        )
    )[:3]:
        print(
            f"input_id_x (len={len(input_id_x)}): {input_id_x}"
            f"\n——> input_id_y (len={len(input_id_y)}): {input_id_y}"
            f"\n——> input_ids (len={len(input_ids)}): {input_ids}"
            f"\n——> input_mask (len={len(input_mask)}): {input_mask}"
            f"\n——> smiles (len={len(smiles)}): {smiles}"
        )

    raw_datasets = datasets.DatasetDict()
    raw_datasets["train"] = lm_datasets
    return raw_datasets


class TextDataset(Dataset):
    """
    TextDataset 类是一个自定义的数据集类, 继承自 Dataset (通常是 PyTorch 中的 torch.utils.data.Dataset)
    主要作用是加载文本数据, 并为模型的输入准备合适的格式
    参数:
        text_datasets: 需要处理的数据集
        data_args: 参数对象
        model_emb: 嵌入模型 (torch.nn.Embedding)
    """

    def __init__(self, text_datasets, data_args, model_emb=None, vocab_size=0):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets["train"])
        self.data_args = data_args
        self.model_emb = model_emb
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    # 该方法用于根据索引 idx 返回数据集中的一个样本
    def __getitem__(self, idx):
        with torch.no_grad():  # 禁用梯度计算, 节省内存和计算资源
            # 读取 input_ids, 并将其转换为 PyTorch 张量
            input_ids = self.text_datasets["train"][idx]["input_ids"]
            # tmp = torch.tensor(input_ids, dtype=torch.int64)
            tmp = torch.tensor(
                [i if 0 <= i < self.vocab_size else 0 for i in input_ids],
                dtype=torch.int64,
            )

            # tmp[:5] = 0
            # print("tmp:\n", tmp)

            # 自定义的条件判断处理
            if self.data_args.ppgraph_len and self.data_args.num_props:
                tmp[: self.data_args.ppgraph_len + self.data_args.num_props] = 0
            elif self.data_args.ppgraph_len:
                tmp[: self.data_args.ppgraph_len] = 0
            elif self.data_args.ppgraph_len:
                tmp[: self.data_args.num_props] = 0

            # print("\nprocessed_tmp:\n", tmp)
            hidden_state = self.model_emb(
                tmp
            )  # 使用嵌入模型将输入转换为对应的隐藏状态表示
            arr = np.array(
                hidden_state, dtype=np.float32
            )  # 将嵌入结果转换为 NumPy 数组
            # out_kwargs 字典中包含输入 ID 和输入掩码, 都是 NumPy 数组格式
            out_kwargs = {}
            out_kwargs["input_ids"] = np.array(
                self.text_datasets["train"][idx]["input_ids"]
            )
            out_kwargs["input_mask"] = np.array(
                self.text_datasets["train"][idx]["input_mask"]
            )
            return arr, out_kwargs

    def log_info(self):
        TextDataset_info = vars(self)
        TextDataset_info_str = "\n".join(
            [f"{key}: {value}" for key, value in TextDataset_info.items()]
        )
        print(f"TextDataset 封装的参数一览:\n{TextDataset_info_str}")
        methods = [
            method
            for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        ] + ["__len__", "__getitem__"]
        print(f"以及实现的方法一览:\n{methods}\n")


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    """
    对列表 examples 中的每个序列进行填充, 生成输入序列和可选的掩码
    输入:
        examples: token 序列的列表
        pad_token_id: 填充token 对应的 ID
        max_length: 允许的最大长度
        return_mask: 布尔值, 指示是否返回掩码
    返回:
        result: 填充后的输入序列 result 和可选的掩码 mask_
    """
    # torch.full 方法能够创建一个形状为 [len(examples), max_length] 的张量
    # 其中所有元素初始化为 pad_token_id
    result = torch.full(
        [len(examples), max_length], pad_token_id, dtype=torch.int64
    ).tolist()
    mask_ = torch.full(
        [len(examples), max_length], pad_token_id, dtype=torch.int64
    ).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        # 替换有效值的同时将掩码 mask_ 中的对应位置的值设置为 1
        # 表示该位置是有效信息而非填充信息
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


# def get_corpus2(data_args, seq_len, data=None, loaded_vocab=None):
#     """
#     get_corpus2 方法对数据集中的每一条数据进行同样的处理:
#     如果提供了数据集 data:
#         如果同时包含属性和骨架信息:
#             将属性值列表和骨架字符串组合起来,
#             之后添加至列表 sentence_lst["src"]
#         如果只包含属性信息:
#             直接将属性值列表添加至列表 sentence_lst["src"]
#         如果只包含骨架信息:
#             直接将骨架字符串添加至列表 sentence_lst["src"]
#         如果不包含任何额外信息:
#             将 [UNCONDITION] 无条件标记添加至列表 sentence_lst["src"]
#         之后将对应的 SMILES 字符串表示添加到 sentence_lst["trg"]
#     如果未提供数据集 data:
#         则根据 num_props 和 scaffold 参数, 构建一些假数据:
#         对于 sentence_lst["src"] 部分的构建:
#             如果同时需要属性信息与骨架信息, 则将属性信息表示为 str(0.9),
#                 骨架信息表示为 C1=NCN=CO1;
#             如果仅需要属性信息, 则将属性信息表示为 [0.4,3.,2.];
#             如果仅需要骨架信息, 则将骨架信息表示为 data_args.scaffold,
#                 可以猜想骨架信息并不一定必须是布尔值, 也可以是字符串;
#             如果不需要任何额外信息, 将 src 数据初始化为 [UNCONDITION]
#         对于 sentence_lst["trg"] 部分, 统一使用一个 [PAD] 进行初始化
#             sentence_lst["trg"].append("[PAD]")
#     之后调用 helper_tokenize 方法对 src 与 trg 进行 token 化编码、合并、掩码、填充
#     最终得到 DatasetDict 格式的数据对象 train_dataset
#     参数:
#         data_args: 参数对象
#         seq_len: 最大序列长度, train.py 之前已经重新计算
#         data: 使用的具体数据集
#         loaded_vocab: 实例化的分词器
#     返回:
#         train_dataset: 一个训练数据集, 通常为 DatasetDict 对象, 包含处理后的数据
#         经过处理后, 训练集中每一个条目都是包含属性、骨架信息与 SMILES 信息的 token 编码向量
#     """
#     sentence_lst = {"src": [], "trg": []}
#     flag = 0
#     total = len(data)
#     if data is not None:
#         for _, row in data.iterrows():
#             flag = flag + 1

#             if data_args.num_props and data_args.scaffold and data_args.ppgraph_len:
#                 src = []
#                 smiles = row["smiles"].strip()
#                 try:
#                     vvs = pp_graph.pp_graph(smiles)
#                 except Exception as e:
#                     total = total - 1
#                     flag = flag - 1
#                     print(f"{row}\n处理时出错:\n{e}")
#                     continue
#                 transformer_model = pp_graph.TransformerEncoder(
#                     output_dim=data_args.ppgraph_len
#                 )
#                 vvs_new = transformer_model(vvs).tolist()
#                 vvs_new = [str(i) for i in vvs_new[0]]
#                 src.extend(vvs_new)
#                 prop = row[data_args.props].values.tolist()
#                 prop = [str(i) for i in prop]
#                 src.extend(prop)
#                 scaffold = row["scaffold_smiles"].strip()
#                 src.append(scaffold)
#                 sentence_lst["src"].append(src)

#             elif data_args.num_props and data_args.scaffold:
#                 src = []
#                 smiles = row["smiles"].strip()
#                 prop = row[data_args.props].values.tolist()
#                 prop = [str(i) for i in prop]
#                 src.extend(prop)
#                 scaffold = row["scaffold_smiles"].strip()
#                 src.append(scaffold)
#                 sentence_lst["src"].append(src)

#             elif data_args.num_props and data_args.ppgraph_len:
#                 src = []
#                 smiles = row["smiles"].strip()
#                 try:
#                     vvs = pp_graph.pp_graph(smiles)
#                 except Exception as e:
#                     total = total - 1
#                     flag = flag - 1
#                     print(f"{row}\n处理时出错:\n{e}")
#                     continue
#                 transformer_model = pp_graph.TransformerEncoder(
#                     output_dim=data_args.ppgraph_len
#                 )
#                 vvs_new = transformer_model(vvs).tolist()
#                 vvs_new = [str(i) for i in vvs_new[0]]
#                 src.extend(vvs_new)
#                 prop = row[data_args.props].values.tolist()
#                 prop = [str(i) for i in prop]
#                 src.extend(prop)
#                 sentence_lst["src"].append(src)

#             elif data_args.scaffold and data_args.ppgraph_len:
#                 src = []
#                 smiles = row["smiles"].strip()
#                 try:
#                     vvs = pp_graph.pp_graph(smiles)
#                 except Exception as e:
#                     total = total - 1
#                     flag = flag - 1
#                     print(f"{row}\n处理时出错:\n{e}")
#                     continue
#                 transformer_model = pp_graph.TransformerEncoder(
#                     output_dim=data_args.ppgraph_len
#                 )
#                 vvs_new = transformer_model(vvs).tolist()
#                 vvs_new = [str(i) for i in vvs_new[0]]
#                 src.extend(vvs_new)
#                 scaffold = row["scaffold_smiles"].strip()
#                 src.append(scaffold)
#                 sentence_lst["src"].append(src)

#             elif data_args.num_props:
#                 src = []
#                 smiles = row["smiles"].strip()
#                 prop = row[data_args.props].values.tolist()
#                 prop = [str(i) for i in prop]
#                 src.extend(prop)
#                 sentence_lst["src"].append(src)

#             elif data_args.scaffold:
#                 src = []
#                 scaffold = row["scaffold_smiles"].strip()
#                 src.append(scaffold)
#                 sentence_lst["src"].append(src)

#             elif data_args.ppgraph_len:
#                 src = []
#                 smiles = row["smiles"].strip()
#                 try:
#                     vvs = pp_graph.pp_graph(smiles)
#                 except Exception as e:
#                     total = total - 1
#                     flag = flag - 1
#                     print(f"{row}\n处理时出错:\n{e}")
#                     continue
#                 transformer_model = pp_graph.TransformerEncoder(
#                     output_dim=data_args.ppgraph_len
#                 )
#                 vvs_new = transformer_model(vvs).tolist()
#                 vvs_new = [str(i) for i in vvs_new[0]]
#                 src.extend(vvs_new)
#                 sentence_lst["src"].append(src)

#             else:
#                 sentence_lst["src"].append("[UNCONDITION]")

#             sentence_lst["trg"].append(row["smiles"].strip())

#             x = total / 3
#             if flag % x == 0:
#                 print(
#                     f"{flag}/{total}  src:{sentence_lst['src'][flag-1]}——>trg:{sentence_lst['trg'][flag-1]}"
#                 )

#     else:
#         for _ in range(data_args.sample):
#             flag = flag + 1

#             if data_args.num_props and data_args.scaffold and data_args.ppgraph_len:
#                 sentence_lst["src"].append(
#                     data_args.vvs
#                     + [str(prop) for prop in data_args.props_cond]
#                     + [data_args.scaffold_cond]
#                 )

#             elif data_args.num_props and data_args.scaffold:
#                 sentence_lst["src"].append(
#                     [str(prop) for prop in data_args.props_cond]
#                     + [data_args.scaffold_cond]
#                 )

#             elif data_args.num_props and data_args.ppgraph_len:
#                 sentence_lst["src"].append(
#                     data_args.vvs + [str(prop) for prop in data_args.props_cond]
#                 )

#             elif data_args.scaffold and data_args.ppgraph_len:
#                 sentence_lst["src"].append(data_args.vvs + data_args.scaffold_cond)

#             elif data_args.num_props:
#                 sentence_lst["src"].append([str(prop) for prop in data_args.props_cond])

#             elif data_args.scaffold:
#                 sentence_lst["src"].append(data_args.scaffold_cond)

#             elif data_args.ppgraph_len:
#                 sentence_lst["src"].append([data_args.vvs])

#             else:
#                 sentence_lst["src"].append("[UNCONDITION]")

#             sentence_lst["trg"].append("[PAD]")

#             x = data_args.sample / 3
#             if flag % x == 0:
#                 print(
#                     f"{flag}/{data_args.sample}  src:{sentence_lst['src'][flag-1]}——>trg:{sentence_lst['trg'][flag-1]}"
#                 )

#     vocab_dict = loaded_vocab
#     # helper_tokenize 对 sentence_lst 进行了编码、合并与掩码、填充, 返回一个包含训练数据集的 DatasetDict 对象
#     train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len, data_args)
#     return train_dataset


def get_corpus2(data_args, seq_len, data=None, loaded_vocab=None):
    """
    简化版的 get_corpus2 方法，功能不变但代码更简洁
    """
    sentence_lst = {"src": [], "trg": []}
    flag = 0
    total = len(data) if data is not None else data_args.sample

    # 处理真实数据
    if data is not None:
        for _, row in data.iterrows():
            flag += 1
            src = []
            skip_row = False

            # # 处理 ppgraph_len 部分
            # if data_args.ppgraph_len:
            #     try:
            #         smiles = row["smiles"].strip()
            #         vvs = my_ppgraph.pp_graph(smiles)
            #         transformer_model = my_ppgraph.TransformerEncoder(
            #             output_dim=data_args.ppgraph_len
            #         )
            #         vvs_new = transformer_model(vvs).tolist()
            #         src.extend([str(i) for i in vvs_new[0]])
            #     except Exception as e:
            #         total -= 1
            #         flag -= 1
            #         print(f"{row}\n处理时出错:\n{e}")
            #         skip_row = True

            # 处理 ppgraph_len 部分 - 直接从数据中获取 ppgraph 列
            if data_args.ppgraph_len:
                try:
                    # 直接从数据行的 ppgraph 列获取药效团信息
                    ppgraph = row["ppgraph"]
                    # 如果 ppgraph 是字符串形式的列表，将其转换为列表
                    if isinstance(ppgraph, str):
                        ppgraph = eval(ppgraph)
                    # 确保 ppgraph 是列表类型
                    if not isinstance(ppgraph, list):
                        raise ValueError(
                            f"ppgraph should be a list, got {type(ppgraph)}"
                        )
                    # 直接使用 ppgraph 列的值
                    src.extend([str(i) for i in ppgraph])
                except Exception as e:
                    total -= 1
                    flag -= 1
                    print(f"{row}\n处理时出错:\n{e}")
                    skip_row = True

            # 处理 props 部分（可能包含复杂性分数）
            if data_args.num_props and not skip_row:
                # 初始化属性列表
                prop_list = []
                # 如果需要计算复杂性分数
                if data_args.complexity:
                    prop_list.append(str(row["complexity"]).strip())
                # 添加其他属性值
                row_props = row[data_args.props].values.tolist()
                # prop_list.extend([str(i) for i in row_props])
                # 处理多个属性的情况
                if isinstance(data_args.props, list):
                    # 多个属性，需要逐个处理
                    for i, (prop_name, value) in enumerate(
                        zip(data_args.props, row_props)
                    ):
                        if prop_name == "tpsa":
                            # 对 tpsa 进行归一化处理
                            try:
                                # 尝试转换为数值
                                num_value = float(value)
                                # 归一化到 0-1 范围
                                normalized_value = num_value / 100.0
                                prop_list.append(str(normalized_value))
                            except (ValueError, TypeError):
                                # 如果无法转换为数值，使用默认值
                                prop_list.append("0")
                        else:
                            # 其他属性保持原样
                            prop_list.append(str(value))
                elif data_args.props == "tpsa":
                    # 单个属性且是 tpsa
                    try:
                        # 尝试转换为数值
                        num_value = float(row_props[0])
                        # 归一化到 0-1 范围
                        normalized_value = num_value / 140.0
                        prop_list.append(str(normalized_value))
                    except (ValueError, TypeError):
                        # 如果无法转换为数值，使用默认值
                        prop_list.append("0")
                else:
                    # 单个属性但不是 tpsa
                    prop_list.extend([str(i) for i in row_props])
                # 将属性列表添加到 src
                src.extend(prop_list)

            # 处理 scaffold 部分
            if data_args.scaffold and not skip_row:
                scaffold = row["scaffold_smiles"].strip()
                src.append(scaffold)

            # 处理无条件情况
            if (
                not any(
                    [data_args.num_props, data_args.scaffold, data_args.ppgraph_len]
                )
                and not skip_row
            ):
                src.append("[UNCONDITION]")

            if not skip_row:
                sentence_lst["src"].append(src)
                sentence_lst["trg"].append(row["smiles"].strip())

    # 处理模拟数据
    else:
        for _ in range(data_args.sample):
            flag += 1
            src = []

            # 处理 ppgraph_len 部分
            if data_args.ppgraph_len:
                src.extend(data_args.vvs)

            # 处理 props 部分（可能包含复杂性分数）
            if data_args.num_props:
                # 初始化属性列表
                prop_list = []
                # 如果需要模拟复杂性分数
                if data_args.complexity:
                    # 使用默认的复杂性分数
                    prop_list.append(str(data_args.complexity))
                # 添加其他属性值
                prop_list.extend([str(prop) for prop in data_args.props])
                # 将属性列表添加到 src
                src.extend(prop_list)

            # 处理 scaffold 部分
            if data_args.scaffold:
                src.append(data_args.scaffold)

            # 处理无条件情况
            if not any(
                [data_args.num_props, data_args.scaffold, data_args.ppgraph_len]
            ):
                src.append("[UNCONDITION]")

            sentence_lst["src"].append(src)
            sentence_lst["trg"].append("[PAD]")

    x = total / 3
    if flag > 0 and flag % max(1, round(x)) == 0:  # 避免除以零
        print(
            f"{flag}/{total}  src:{sentence_lst['src'][flag-1]}——>trg:{sentence_lst['trg'][flag-1]}"
        )

    # 使用分词器处理数据
    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len, data_args)
    return train_dataset
