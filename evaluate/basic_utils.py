import argparse
import torch
import json, os
import time
import re
from diffumol import gaussian_diffusion as gd
from diffumol.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffumol.transformer_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from diffumol.utils import logger
from typing import List
import numpy as np
import regex  # 确保你已导入 regex


class myTokenizer:

    def __init__(self, args):
        """
        类的初始化方法
        如果 args.vocab 为 bert, 使用 args.config_name 指定的文件作为参数,
            调用 AutoTokenizer.from_pretrained 创建一个 tokenizer
        如果 args.vocab 是文件路径,
            首先初始化一个包含特殊标记的词汇字典 vocab_dict,
            注意: 此处如果要使用作者的参数 (没有条件约束且 vocab_size=99) 运行,
                需要使用不包含 "[PROP]":5 的词汇字典
            之后读取 args.vocab 指定的包含其他词汇的文件并添加至 vocab_dict,
                并对新添加的词汇赋予对应的 ID
            之后将 vocab_dict 直接赋予 tokenizer, 然后反转其键与值的顺序创建 rev_tokenizer
        创建 tokenizer 之后,
            获取 “分隔字符” 的对应 ID 到参数 self.sep_token_id,
            获取 “填充字符” 的对应 ID 到参数 self.pad_token_id,
            将加载的 tokenizer 保存至 args.checkpoint_path 指定的文件中
        之后, 计算词汇字典的大小 vocab_size
        """
        if args.vocab == "bert":
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            tokenizer.save_pretrained(args.checkpoint_path)
        else:
            vocab_dict = {
                "[START]": 0,
                "[END]": 1,
                "[UNK]": 2,
                "[PAD]": 3,
                "[UNCONDITION]": 4,
                "[PROP]": 5,
                "[MASK]": 6,
            }
            # vocab_dict = {"[START]": 0, "[END]": 1, "[UNK]":2, "[PAD]":3,"[UNCONDITION]":4}
            print(f"加载词汇表{args.vocab}")
            with open(args.vocab, "r", encoding="utf-8") as f:
                for row in f:
                    vocab_dict[row.strip().split(" ")[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict["[END]"]
            self.pad_token_id = vocab_dict["[PAD]"]
            # 仅允许主进程保存完整的词汇文件, 以避免多个进程同时写入文件
            if int(os.environ["LOCAL_RANK"]) == 0:
                path_save_vocab = f"{args.checkpoint_path}/vocab.json"
                with open(path_save_vocab, "w") as f:
                    json.dump(vocab_dict, f)
        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size
        self.mask = args.mask

    def encode_token(
        self, sentences, ppgraph_len=False, num_props=False, scaffold=False
    ):
        """
        首先定义一个模式 pattern, 并编译为正则表达式 regex, 之后根据参数设置执行不同的操作:
        如果同时提供了属性的数量 num_props!=0 和分子骨架信息 scaffold,
             (说明传入的是属性值与骨架构成的列表)
            对句子列表 sentences 中的每一个句子:
            截取出属性元素并转换为浮点数后赋给 prop, 然后直接赋给 tmp,
            截取出骨架元素赋值给 scaf, 然后使用正则表达式 regex 将骨架拆分,
                并使用 tokenizer 转换为 ID 后追加至 tmp (tokenizer 未找到的元素就设置为 [UNK])
            之后将 tmp 整个添加到 input_ids 中
        如果只提供了属性的数量 num_props!=0 (说明整个句子均是属性信息) ,
            对句子列表 sentences 中的每一个句子:
            如果句子是一个列表 (一般情况下都是) , 将其直接加入 tmp,
            如果不是列表就使用正则表达式和 tokenizer 进行处理后加入 tmp,
            之后将 tmp 整个添加到 input_ids 中
        如果没有提供任何额外信息 (说明传入的是一个纯粹的 SMILES 字符串或 [UNCONDITION]) ,
            对句子列表 sentences 中的每一个句子:
            先使用正则表达式和 tokenizer 处理后,
            之后在句子的编码结果前添加 0, 表示开始标记, 在编码结果后添加 1, 表示结束标记,
            然后加入到 tmp, 之后将 tmp 整个添加到 input_ids 中
        最终返回 input_ids
        参数:
            sentences (list): 要编码的句子列表
                在训练过程中有相关调用:
                input_id_x = vocab_dict.encode_token(examples["src"], num_props=data_args.num_props, scaffold=data_args.scaffold)
                input_id_y = vocab_dict.encode_token(examples["trg"])
                说明每个句子只可能为以下几种格式之一:
                    包含若干属性、一个分子骨架 (属性和骨架不一定存在, 由其他两个参数指示) ,
                        此时 num_props 为 >=0 的整数;
                    或只是一个 SMILES 字符串, 此时 num_props 为布尔类型 false;
                    均不存在时, 会用 [UNCONDITION] 表示
            num_props (int): 指示是否存在以及有多少属性
            scaffold (bool): 指示是否存在支架信息
        返回:
            input_ids (list): 编码后的 ID 序列
        """
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        if num_props and scaffold and ppgraph_len:
            input_ids = []
            for seq in sentences:
                tmp = []
                pp_graph = [float(i) for i in seq[0:ppgraph_len]]
                # print("pp_graph: ", pp_graph)
                prop = [float(j) for j in seq[ppgraph_len : ppgraph_len + num_props]]
                # print("prop: ", prop)
                scaf = seq[ppgraph_len + num_props]
                # print("scaf: ", scaf)
                tmp += pp_graph
                tmp += prop
                tmp += [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(scaf.strip())
                ]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif num_props and scaffold:
            input_ids = []
            for seq in sentences:
                tmp = []
                # print(seq, num_props)
                prop = [float(j) for j in seq[:num_props]]
                scaf = seq[num_props]
                tmp += prop
                tmp += [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(scaf.strip())
                ]
                input_ids.append(tmp)

        elif num_props and ppgraph_len:
            input_ids = []
            for seq in sentences:
                tmp = []
                pp_graph = [float(i) for i in seq[0:ppgraph_len]]
                prop = [float(j) for j in seq[ppgraph_len : ppgraph_len + num_props]]
                # scaf = seq[ppgraph_len+num_props]
                tmp += pp_graph
                tmp += prop
                # tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)

        elif scaffold and ppgraph_len:
            input_ids = []
            for seq in sentences:
                tmp = []
                pp_graph = [float(i) for i in seq[0:ppgraph_len]]
                # prop = [float(j) for j in seq[ppgraph_len : ppgraph_len+num_props]]
                scaf = seq[ppgraph_len + num_props]
                tmp += pp_graph
                # tmp += prop
                # tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)

        elif num_props:
            input_ids = []
            for seq in sentences:
                tmp = []
                prop = [float(j) for j in seq[:num_props]]
                tmp += prop
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)

        elif scaffold:
            input_ids = []
            for seq in sentences:
                tmp = []
                # pp_graph = [float(i) for i in seq[0 : ppgraph_len]]
                # prop = [float(j) for j in seq[ppgraph_len : ppgraph_len+num_props]]
                scaf = seq[ppgraph_len + num_props]
                # tmp += pp_graph
                # tmp += prop
                tmp += [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(scaf.strip())
                ]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif ppgraph_len:
            input_ids = []
            for seq in sentences:
                tmp = []
                # print("seq:", seq)
                # pp_graph = [float(i) for i in seq[0 : ppgraph_len]]
                pp_graph = [float(i) for i in seq[0][0:ppgraph_len]]
                # print("pp_graph: ", pp_graph)
                tmp += pp_graph
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)

        else:
            # 无掩码
            # for i, seq in enumerate(sentences[:10]):
            #     print(f"句子 {i}: 类型={type(seq)}, 内容={repr(seq)}")
            input_ids = [
                [0]
                + [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(seq.strip())
                ]
                + [1]
                for seq in sentences
            ]

            # 有掩码 —— 遮掩每条 token 序列中间的 10% token
            # if self.mask:
            #     mask_token = self.tokenizer.get("[MASK]", self.tokenizer["[UNK]"])
            #     input_ids_masked = []
            #     for seq in input_ids:
            #         masked_seq = seq.copy()
            #         l = len(seq)
            #         # 跳过太短的序列
            #         if l <= 2:
            #             input_ids_masked.append(masked_seq)
            #             continue
            #         # 不包含 [CLS]=0 和 [SEP]=1
            #         candidate_indices = list(range(1, l - 1))
            #         num_mask = max(1, int(len(candidate_indices) * 0.05))  # 至少遮一个
            #         mask_indices = np.random.choice(
            #             candidate_indices, size=num_mask, replace=False
            #         )
            #         for idx in mask_indices:
            #             masked_seq[idx] = mask_token
            #         input_ids_masked.append(masked_seq)
            #     input_ids = input_ids_masked

        return input_ids

    def decode_token(self, seq):
        """
        解码输入的 token ID 序列为可读的文本字符串
        如果 self.tokenizer 为字典类型 (说明是利用 vob.txt 自定义的 tokenizer) ,
            使用 squeeze(-1) 方法去掉序列中最后一个维度, 然后将其转换为 Python 一维列表
            之后将序列后端的 [PAD] 填充对应的 ID 删除, 只保留前面有意义的内容,
            然后使用列表推导式遍历序列, 通过 rev_tokenizer 将每个 ID 转换为对应的 token 字符,
            再使用 join() 方法将所有 token 字符连接成一个以空格分隔的字符串,
            使用 replace("__ ", "") 和 replace("@@ ", "") 移除特定的前缀,
            因为这些前缀通常用于 BPE (Byte Pair Encoding) 分词的标记
        如果 self.tokenizer 是 PreTrainedTokenizerFast 类型,
             (表示使用了 AutoTokenizer.from_pretrained 创建 tokenizer) ,
            前面的步骤相同, 仅在删除了填充信息之后,
            使用 self.tokenizer.decode() 方法直接将 token ID 列表转换为文本字符串
        如果 self.tokenizer 既不是字典也不是 PreTrainedTokenizerFast 类型,
            则触发断言并抛出错误信息, 表示词汇字典类型无效
        参数:
            seq: 包含 token ID 的序列,
                一般仅在生成的解码阶段会使用
        返回:
            tokens (str): 解码后的字符串
        """
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = (
                " ".join([self.rev_tokenizer[x] for x in seq])
                .replace("__ ", "")
                .replace("@@ ", "")
            )

        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)

        else:
            assert False, "invalid type of vocab_dict"

        return tokens


def load_model_emb(args, vocab_size):
    """
    对于每一个进程:
    所有进程都会使用 torch.nn.Embedding 创建一个新的嵌入模型 [vocab_size, args.hidden_dim],
        之后会检查指定路径下是否已有权重文件, 如果有则加载它
    一般情况下主进程到来时, 不会存在现有的权重文件,
        因此会使用 torch.nn.init.normal_ 方法将嵌入权重初始化为正态分布随机值,
        然后将自己初始化的模型权重保存为新的权重文件
    之后其他进程一般会直接调用该权重文件加载自己的嵌入模型,
        以保证自己的嵌入模型与主进程的嵌入模型保持一致
    参数:
        args: 参数对象
        vocab_size: tokenizer 创建的词汇表大小
    返回:
        model: 初始化的或加载的嵌入模型 (torch.nn.Embedding)
    """
    model = torch.nn.Embedding(vocab_size, args.hidden_dim)
    path_save = "{}/my_random_emb.torch".format(args.checkpoint_path)
    path_save_ind = path_save + ".done"
    # 仅允许主进程初始化和保存嵌入模型
    if int(os.environ["LOCAL_RANK"]) == 0:
        if os.path.exists(path_save):
            print(f"主进程从现有的嵌入模型权重文件 {path_save} 中加载参数:", model)
            model.load_state_dict(torch.load(path_save))
        else:
            print("主进程未找到现成的权重文件, 初始化一个随机嵌入模型:", model)
            torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()
            with open(path_save_ind, "x") as _:
                pass
    # 对于其他进程, 进入一个循环, 直到找到指示文件 (.done) , 以确保嵌入已经初始化完成
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print(f"其余进程从现有的权重文件 {path_save} 中加载参数: ", model)
        model.load_state_dict(torch.load(path_save))
    return model


def load_tokenizer(args):
    tokenizer = myTokenizer(args)
    return tokenizer


def load_defaults_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,  # 默认 128
    hidden_dim,  # 默认 128
    vocab_size,  # 默认 0, 但 arg 在 train.py 中已经重新计算
    config_name,  # 默认 "./datasets/model.json"
    use_plm_init,  # 默认 "no"
    dropout,  # 默认 0.1
    diffusion_steps,  # 默认 2000
    noise_schedule,  # 默认 "sqrt"
    learn_sigma,  # 默认 false
    timestep_respacing,  # 默认为 ""
    predict_xstart,  # 默认 true
    rescale_timesteps,  # 默认 true
    sigma_small,  # 默认 false
    rescale_learned_sigmas,  # 默认 false
    use_kl,  # 默认 false
    notes,
    **kwargs,
):
    # TransformerNetModel 实现了具有注意力机制和时间步嵌入的完整 Transformer 模型
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim * 2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init,
        **kwargs,
    )

    # 根据传入的调度名称和扩散时间步数生成对应的 beta 值
    # sqrt: 生成基于平方根的 beta 值
    # lambda t: 1 - np.sqrt(t + 0.0001): 这是一个匿名函数, 计算每个时间步的 beta 值
    # 由于 t 是从 0 开始的, np.sqrt(0 + 0.0001) 会产生一个小的值,
    # 因此 beta 值在开始时接近于 1, 随着 t 的增加, beta 值会逐渐减小
    # 最终, get_named_beta_schedule 函数将返回一个包含 2000 个 beta 值的 NumPy 数组
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    # SpacedDiffusion 类继承自 GaussianDiffusion, 改进在于允许用户指定保留的时间步, 跳过某些步骤
    diffusion = SpacedDiffusion(
        # space_timesteps 方法接收参数 diffusion_steps (总时间步数), timestep_respacing (指定每个部分的时间步数量)
        # 主要用途是在需要将时间步分成不同部分的情况下, 如在扩散模型中以不同的方式采样时间步
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas,
        num_props=kwargs["num_props"],  # 默认为 0
        ppgraph_len=kwargs["ppgraph_len"],
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    """
    将一个字典中的键值对添加为命令行参数
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    参数:
        args: 包含多个属性的对象
        keys: 从配置字典中提取的所有键
    函数内部使用字典推导式 {k: getattr(args, k) for k in keys},
    根据 keys 中的每个键 k 从 args 对象中获取对应的属性值, 并生成一个新的字典
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    在使用 argparse 时将字符串输入转换为布尔值的实用工具
    """
    if isinstance(v, bool):
        return v
    # 将字符串转换为小写, 并检查是否匹配常见的 True 或 False 表示
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    # 如果字符串不匹配任何预期值, 则抛出 ArgumentTypeError, 指示输入无效
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
