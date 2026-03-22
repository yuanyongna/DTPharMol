import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys

lib_path = os.path.abspath("/home/yuanyn/songhw/DIFFUMOL/RL_utils")
sys.path.append(lib_path)
lib_path = os.path.abspath("/home/yuanyn/songhw/DIFFUMOL")
sys.path.append(lib_path)
import json
import gc
import re
from rdkit import Chem
import pandas
import time
import numpy
import torch
import random
import argparse
from diffumol.RL_utils import dist_util
import torch.distributed as dist
from evaluate.basic_utils import create_model_and_diffusion
from diffumol.RL_utils.Tokenizer_forRL import Tokenizer_forRL
from transformers import set_seed
from diffumol.RL_utils.file_utils import load_phar_file
from diffumol.RL_utils.create_dataloader import create_dataloader
from diffumol.smiles_sample import smiles_sample
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "5286dc1a63fbde135489755cc7407102d649be44"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pp_graph_len = 6


def get_seed2(seed_file):
    """
    加载已使用的随机种子, 生成一个未使用的种子, 并保存它
    """
    used_seeds = set()
    # 安全读取种子文件
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            for line_number, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:  # 跳过空行
                    continue
                try:
                    seed = int(stripped)
                    used_seeds.add(seed)
                except ValueError:
                    print(
                        f"[警告] 种子文件第{line_number}行包含无效数据: '{stripped}'，已忽略"
                    )
    # 生成新种子
    max_attempts = 1000
    for _ in range(max_attempts):
        candidate = random.randint(0, 1000)
        if candidate not in used_seeds:
            with open(seed_file, "a") as f:
                f.write(f"{candidate}\n")
            return candidate
    raise RuntimeError(f"在{max_attempts}次尝试后仍未找到可用种子")


def load_defaults_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


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
    arr_norm = (text_emb**2).sum(-1).view(-1, 1)
    # 计算每个文本嵌入的 L2 范数并转换为形状 (M, 1), M 为文本嵌入的数量
    # torch.mm(model_emb, text_emb_t)：进行矩阵乘法，计算模型嵌入与文本嵌入的点积，结果形状为 (N, M)
    # 整个表达式计算出每对嵌入之间的欧几里得距离
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)
    # 确保所有距离值非负
    dist = torch.clamp(dist, 0.0, numpy.inf)
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


def get_mol(smiles_or_mol):
    """
    将 SMILES 字符串或分子对象加载到 RDKit 的分子对象中
    参数:
        smiles_or_mol: str 或 RDKit 分子对象
    返回:
        Mol: RDKit 分子对象, 如果输入无效, 则返回 None
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            # 对分子进行检查和处理, 以确保其符合化学规范
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


@torch.no_grad()
def generation_forRL(model_path, general_nums, general_temp_path, config_file):
    print("*" * 100)
    print("进入生成程序")
    # 参数加载
    # print("参数加载")
    with open(config_file, "r") as f:
        defaults = json.load(f)
    seed2 = get_seed2(args.seed_file)
    decode_defaults = dict(split="test", clamp_step=0, seed2=seed2, clip_denoised=False)
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v)
    parser.add_argument("--model_path", default=model_path)
    parser.add_argument("--sample", default=general_nums, type=int)
    args = parser.parse_args([])  # 传入空列表以避免从命令行读取
    args.batch_size = 128
    args.step = 2000
    args.split = "test"
    args.top_p = 5
    # print(f"参数汇总: {args}")

    # print("分布式进程设置")
    dist_util.setup_dist()
    # 检索分布式设置中涉及的总进程数, 如果未找到则默认为 1, 表示单个进程
    world_size = dist.get_world_size() or 1
    # 获取当前进程在分布式设置中的唯一标识符 (排名), 如果未找到则默认为 0
    rank = dist.get_rank() or 0
    # config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    model_dir = os.path.dirname(
        os.path.dirname(args.model_path)
    )  # 提取 ./RL_utils/model_ckpt
    config_path = os.path.join(model_dir, "training_args.json")
    print("生成过程 config_path: ", config_path)
    with open(
        config_path,
        "rb",
    ) as f:
        training_args = json.load(f)
    # print(f"读取模型超参数文件 {config_path}: {training_args}")
    training_args["batch_size"] = args.batch_size
    # 将 training_args 的内容添加到 args 对象中
    args.__dict__.update(training_args)

    print(f"初始化模型并赋予模型权重: {args.model_path}")
    # 初始化模型
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config(config_file).keys())
    )
    # 赋予模型权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval().requires_grad_(False).to(dist_util.dev())

    # 初始化 tokenizer
    tokenizer = Tokenizer_forRL(args)
    # 初始化嵌入层
    model_emb = (
        torch.nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=args.hidden_dim,
            _weight=model.word_embedding.weight.clone().cpu(),
        )
        .eval()
        .requires_grad_(False)
    )
    set_seed(args.seed2)

    """
    初始化格式数据集 all_test_data, 将 model_emb 移至 GPU, 设置分布式进程以及 all_test_data 的数据迭代器 iterator
    """
    print("初始化格式数据集")
    if args.pp_graph:
        import torch.nn as nn
        from data.my_ppgraph import GGCNEncoderBlock, TransformerEncoder

        pp_v_dim = 8
        pp_e_dim = 1
        hidden_dim = 384
        MAX_NUM_PP_GRAPHS = 6
        # pp_graph_len = 10

        pp_graphs = load_phar_file(args.phar_path)
        pp_v_init = nn.Linear(pp_v_dim, hidden_dim)
        pp_e_init = nn.Linear(pp_e_dim, hidden_dim)
        pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))
        pp_encoder = GGCNEncoderBlock(
            hidden_dim,
            hidden_dim,
            n_layers=4,
            dropout=0,
            readout_pooling="max",
            batch_norm=True,
            residual=True,
        )

        v = pp_v_init(pp_graphs.ndata["h"])
        e = pp_e_init(pp_graphs.edata["h"])
        v, e = pp_encoder.forward_feature(pp_graphs, v, e)
        vv = v.new_ones((MAX_NUM_PP_GRAPHS, v.shape[1])) * -999
        vv[: v.shape[0], :] = v
        v = vv
        vvs = vv + pp_seg_encoding

        transformer_model = TransformerEncoder(output_dim=pp_graph_len)
        vvs_new = transformer_model(vvs).tolist()
        vvs_new = [str(i) for i in vvs_new[0]]

        # print("vvs: ", vvs_new)
        args.vvs = vvs_new
    else:
        print("未指定药效团约束")

    data_valid = create_dataloader(
        batch_size=args.microbatch,
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
        print("迭代读取数据对象 data_valid 中的数据序列完成, 下面开始进行分子预测")

    model_emb.to(dist_util.dev())

    # 分布式进程相关设置
    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})
    iterator = iter(all_test_data)

    """
    构建嵌入后的数据 x_start, 更新掩码 input_ids_mask_ori,
    构建去噪过程的初始数据 x_noised (保留属性与骨架信息, 将 SMILES 与之后的填充部分用随机噪声替换), 
    构建去噪扩散模型 sample_fn 与样本形状参数 sample_shape
    """
    start_t = time.time()
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # general_path = os.path.join(current_directory, 'data_paras', 'general_tmp.csv')
    if os.path.exists(general_temp_path):
        os.remove(general_temp_path)
        # print(f"文件 {general_temp_path} 已存在，已删除。")
    print("生成文本序列缓存的文件路径: ", general_temp_path)
    # 对 iterator 中的每个元素进行迭代
    # 一般来将, all_test_data 列表中的每一个字典都会构成一个 iterator
    # 也就是说, 这里 cond 就是列表中的那唯一的字典, cond 中包含了两个列表 input_ids 和 input_mask
    for index, cond in enumerate(iterator):
        torch.cuda.empty_cache()
        gc.collect()  # 进行显式垃圾回收
        # print(f"\n### 序号: {index}, 批次大小 = {len(cond['input_ids'])}")
        smiles_sample(
            args,
            world_size,
            index,
            cond,
            model,
            diffusion,
            model_emb,
            tokenizer,
            general_temp_path,
            start_t,
        )
    print("\n采样过程共耗时: {:.2f}s".format(time.time() - start_t))

    df = pandas.read_csv(general_temp_path)
    smiles = df["smiles"].tolist()
    smiles = list(set(smiles))
    print("采样 smiles 数量: ", len(smiles))

    """
    检验样本的合理性
    """
    print("\n检验样本的合理性")
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

    # print(f"nums of smiles: {len(smiles)}")
    print(f"通过校验的分子数量: {len(molecules)}")

    mol_dict = []

    for i in molecules:
        mol_dict.append({"molecule": i, "smiles": Chem.MolToSmiles(i)})

    results = pandas.DataFrame(mol_dict)
    # print(results)
    tmp = results["smiles"].tolist()
    # print(tmp)

    return tmp


"""
model_path = "./RL_utils/model_ckpt/test/ema_0.9999_005000.pt"
general_onestep = 128
general_smiles_list = generation_forRL(model_path, general_onestep)
print("生成的分子总数: ", len(general_smiles_list), "\n", general_smiles_list)
"""
