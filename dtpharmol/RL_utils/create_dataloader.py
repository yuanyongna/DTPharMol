import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
import datasets
from datasets import Dataset as Dataset2
from diffumol.RL_utils import pp_graph

torch.set_printoptions(edgeitems=4)


def create_dataloader(
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

    training_data = get_corpus2(data_args, seq_len, data, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb,
        vocab_size=loaded_vocab.vocab_size,
    )
    # print("dataset:", dataset)

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

    raw_datasets = Dataset2.from_dict(sentence_lst)

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(
            examples["src"],
            ppgraph_len=data_args.ppgraph_len,
            num_props=data_args.num_props,
            scaffold=data_args.scaffold,
        )
        input_id_y = vocab_dict.encode_token(examples["trg"])
        result_dict = {"input_id_x": input_id_x, "input_id_y": input_id_y}
        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,  # 以批处理的方式处理数据
        num_proc=4,  # 使用 4 个进程进行处理
        remove_columns=["src", "trg"],  # 移除原始的 src 和 trg 列
        load_from_cache_file=True,  # 尝试从缓存中加载之前处理的结果
        desc="对数据集执行 tokenize 编码操作",  # 在处理过程中显示进度条
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
        num_proc=4,
        desc=f"合并与填充操作",
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
        desc=f"填充操作",
    )

    raw_datasets = datasets.DatasetDict()
    raw_datasets["train"] = lm_datasets

    return raw_datasets


class TextDataset(Dataset):

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

            # 自定义的条件判断处理
            if self.data_args.ppgraph and self.data_args.num_props:
                tmp[: self.data_args.ppgraph_len + self.data_args.num_props] = 0
            elif self.data_args.ppgraph:
                tmp[: self.data_args.ppgraph_len] = 0
            elif self.data_args.ppgraph:
                tmp[: self.data_args.num_props] = 0

            # print("\nprocessed_tmp:\n", tmp, tmp.size())
            hidden_state = self.model_emb(tmp)
            # print("\nhidden_state:\n", hidden_state, hidden_state.size())
            # arr = np.array(hidden_state, dtype=torch.float32)
            # print(hidden_state.device)
            arr = hidden_state.cpu().detach().numpy().astype(np.float32)
            # print("\narr:\n", arr)
            # out_kwargs 字典中包含输入 ID 和输入掩码, 都是 NumPy 数组格式
            out_kwargs = {}
            out_kwargs["input_ids"] = np.array(
                self.text_datasets["train"][idx]["input_ids"]
            )
            out_kwargs["input_mask"] = np.array(
                self.text_datasets["train"][idx]["input_mask"]
            )
            return arr, out_kwargs


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


def get_corpus2(data_args, seq_len, data=None, loaded_vocab=None):

    sentence_lst = {"src": [], "trg": []}
    flag = 0
    if data is not None:
        total = len(data)

    if data is not None:
        for _, row in data.iterrows():

            flag = flag + 1

            if data_args.ppgraph and data_args.num_props and data_args.scaffold:
                src = []
                smiles = row["smiles"].strip()
                vvs = pp_graph.pp_graph(smiles)
                try:
                    vvs = pp_graph.pp_graph(smiles)
                except Exception as e:
                    total = total - 1
                    flag = flag - 1
                    print(f"{row}\n处理时出错:\n{e}")
                    continue  # 跳过当前行，继续处理下一个
                transformer_model = pp_graph.TransformerEncoder(
                    output_dim=data_args.ppgraph_len
                )
                vvs_new = transformer_model(vvs).tolist()
                vvs_new = [str(i) for i in vvs_new[0]]
                src.extend(vvs_new)
                prop = row[data_args.props].values.tolist()
                prop = [str(i) for i in prop]
                src.extend(prop)
                scaffold = row["scaffold_smiles"].strip()
                src.append(scaffold)
                sentence_lst["src"].append(src)

            elif data_args.ppgraph and data_args.num_props:
                src = []
                smiles = row["smiles"].strip()
                vvs = pp_graph.pp_graph(smiles)
                transformer_model = pp_graph.TransformerEncoder(
                    output_dim=data_args.ppgraph_len
                )
                vvs_new = transformer_model(vvs).tolist()
                vvs_new = [str(i) for i in vvs_new[0]]
                src.extend(vvs_new)
                prop = row[data_args.props].values.tolist()
                prop = [str(i) for i in prop]
                src.extend(prop)
                # scaffold = row["scaffold_smiles"].strip()
                # src.append(scaffold)
                sentence_lst["src"].append(src)

            elif data_args.ppgraph and data_args.scaffold:
                src = []
                smiles = row["smiles"].strip()
                vvs = pp_graph.pp_graph(smiles)
                transformer_model = pp_graph.TransformerEncoder(
                    output_dim=data_args.ppgraph_len
                )
                vvs_new = transformer_model(vvs).tolist()
                vvs_new = [str(i) for i in vvs_new[0]]
                src.extend(vvs_new)
                # prop = row[data_args.props].values.tolist()
                # prop = [str(i) for i in prop]
                # src.extend(prop)
                scaffold = row["scaffold_smiles"].strip()
                src.append(scaffold)
                sentence_lst["src"].append(src)

            elif data_args.num_props and data_args.scaffold:
                src = []
                # smiles = row["smiles"].strip()
                # vvs = pp_graph.pp_graph(smiles)
                # transformer_model = pp_graph.TransformerEncoder()
                # vvs_new = transformer_model(vvs).tolist()
                # vvs_new = [str(i) for i in vvs_new[0]]
                # src.extend(vvs_new)
                prop = row[data_args.props].values.tolist()
                prop = [str(i) for i in prop]
                src.extend(prop)
                scaffold = row["scaffold_smiles"].strip()
                src.append(scaffold)
                sentence_lst["src"].append(src)

            elif data_args.ppgraph:
                src = []
                smiles = row["smiles"].strip()
                try:
                    vvs = pp_graph.pp_graph(smiles)
                except:
                    vvs = [
                        "-0.3625708222389221",
                        "-0.3750048875808716",
                        "0.5308328866958618",
                        "-0.2839233875274658",
                        "-1.1148442029953003",
                        "-0.5822687149047852",
                    ]
                transformer_model = pp_graph.TransformerEncoder(
                    output_dim=data_args.ppgraph_len
                )
                vvs_new = transformer_model(vvs).tolist()
                vvs_new = [str(i) for i in vvs_new[0]]
                src.extend(vvs_new)
                # prop = row[data_args.props].values.tolist()
                # prop = [str(i) for i in prop]
                # src.extend(prop)
                # scaffold = row["scaffold_smiles"].strip()
                # src.append(scaffold)
                sentence_lst["src"].append(src)

            elif data_args.num_props:
                # src = []
                # smiles = row["smiles"].strip()
                # vvs = pp_graph.pp_graph(smiles)
                # transformer_model = pp_graph.TransformerEncoder()
                # vvs_new = transformer_model(vvs).tolist()
                # vvs_new = [str(i) for i in vvs_new[0]]
                # src.extend(vvs_new)
                prop = row[data_args.props].values.tolist()
                prop = [str(i) for i in prop]
                src.extend(prop)
                # scaffold = row["scaffold_smiles"].strip()
                # src.append(scaffold)
                sentence_lst["src"].append(src)

            elif data_args.scaffold:
                src = []
                # smiles = row["smiles"].strip()
                # vvs = pp_graph.pp_graph(smiles)
                # transformer_model = pp_graph.TransformerEncoder()
                # vvs_new = transformer_model(vvs).tolist()
                # vvs_new = [str(i) for i in vvs_new[0]]
                # src.extend(vvs_new)
                # prop = row[data_args.props].values.tolist()
                # prop = [str(i) for i in prop]
                # src.extend(prop)
                scaffold = row["scaffold_smiles"].strip()
                src.append(scaffold)
                sentence_lst["src"].append(src)

            else:
                sentence_lst["src"].append("[UNCONDITION]")

            """
            if data_args.num_props and data_args.scaffold:
                prop=row[data_args.props].values.tolist()
                prop=[str(i) for i in prop]
                prop.append(row['scaffold_smiles'].strip())
                sentence_lst['src'].append(prop)
            elif data_args.num_props:
                sentence_lst['src'].append(row[data_args.props].values.tolist())
            elif data_args.scaffold:
                sentence_lst['src'].append(row['scaffold_smiles'].strip())
            else:
                sentence_lst['src'].append('[UNCONDITION]')
            """

            sentence_lst["trg"].append(row["smiles"].strip())

            if flag % 200 == 0:
                print(
                    f"{flag}/{total}  src:{sentence_lst['src'][flag-1]}——>trg:{sentence_lst['trg'][flag-1]}"
                )

    else:
        for _ in range(data_args.sample):

            flag = flag + 1

            if data_args.ppgraph and data_args.num_props and data_args.scaffold:
                sentence_lst["src"].append(
                    data_args.vvs
                    + [str(0.4), str(3.0), str(2.0), data_args.scaffold_cond]
                )

            elif data_args.ppgraph and data_args.num_props:
                sentence_lst["src"].append(
                    data_args.vvs + [str(val) for val in data_args.props_cond]
                )

            elif data_args.ppgraph and data_args.scaffold:
                sentence_lst["src"].append(data_args.vvs + data_args.scaffold_cond)
            elif data_args.num_props and data_args.scaffold:
                sentence_lst["src"].append([str(0.9), data_args.scaffold_cond])
            elif data_args.ppgraph:
                # print("生成数据构建_仅药效团")
                sentence_lst["src"].append(data_args.vvs + [])
            elif data_args.num_props:
                sentence_lst["src"].append([0.4, 3.0, 2.0])
            elif data_args.scaffold:
                sentence_lst["src"].append(data_args.scaffold_cond)
            else:
                sentence_lst["src"].append("[UNCONDITION]")

            """
            if data_args.num_props and data_args.scaffold:
                sentence_lst["src"].append([str(0.9), data_args.scaffold_cond])
            elif data_args.num_props:
                sentence_lst["src"].append([0.4,3.,2.])
            elif data_args.scaffold:
                sentence_lst["src"].append(data_args.scaffold_cond)
            else:
                sentence_lst["src"].append("[UNCONDITION]")
            """

            sentence_lst["trg"].append("[PAD]")

            # print(f"{flag}/{data_args.sample}  src:{sentence_lst['src'][flag-1]}——>trg:{sentence_lst['trg'][flag-1]}")

    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len, data_args)

    return train_dataset
