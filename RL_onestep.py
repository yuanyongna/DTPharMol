import os
import json
import glob
import time
import torch
import psutil
import setproctitle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from diffumol.utils import logger
from transformers import set_seed
from diffumol.RL_utils.docking import vina_docking
from diffumol.RL_utils.train_forRL import train_forRL
from diffumol.RL_utils.generation_forRL import generation_forRL

# from diffumol.RL_utils.docking import run_docking_and_normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "5286dc1a63fbde135489755cc7407102d649be44"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
RDLogger.DisableLog("rdApp.*")


def main():
    logger.configure()
    print("*" * 100)
    setproctitle.setproctitle("shw_RL_onestep")
    current_process = psutil.Process(os.getpid())
    print(f"当前进程 ID: {current_process.pid}")
    print(f"当前进程名称: {current_process.name()}")
    print(f"父进程 ID: {current_process.ppid()}")
    print(f"进程状态: {current_process.status()}")
    print(f"进程创建时间: {time.ctime(current_process.create_time())}")
    print(f"内存信息: {current_process.memory_info()}")
    with open(config_file, "r") as f:
        parser = json.load(f)
    print(f"随机种子设置为: {parser['seed']}")
    set_seed(parser["seed"])

    """
    强化学习的启动阶段, 第一次强化学习, 需要借助第一阶段的两个结果
    """
    start_time = time.time()
    print("*" * 100)
    print("初始化分子对接数据集")
    init_docking_dir = os.path.join(RL_result_dir, "prepare")
    os.makedirs(init_docking_dir, exist_ok=True)
    init_docking_result_path = os.path.join(init_docking_dir, "docking_result.csv")
    if os.path.exists(init_docking_result_path):
        print("*" * 100)
        print(f"分子对接数据集已经被初始化: {init_docking_result_path}")
    else:
        print("*" * 100)
        print("执行初始分子对接过程")
        # 读取第一阶段生成分子的参数文件, 并将 smiles 列表单独复制保存至新的文件中
        init_data = pd.read_csv(init_moleculars_path)
        init_data = init_data.dropna(axis=0).reset_index(drop=True)
        init_data.columns = init_data.columns.str.lower()
        init_smiles = init_data["smiles"]
        init_smiles_path = os.path.join(init_docking_dir, "smiles_RL_init.csv")
        init_smiles.to_csv(init_smiles_path, mode="w", header=True, index=False)
        # 初始化分子对接字典
        data = pd.read_csv(init_smiles_path)
        smiles_list = data["smiles"].to_list()
        smiles_dict_list = [{"smiles": smiles, "score": 0} for smiles in smiles_list]
        # 开始对接
        valid_dacking = 0
        for i, smiles in tqdm(
            enumerate(smiles_dict_list), total=len(smiles_dict_list), desc="Docking——>"
        ):
            # 将 smiles 转换为 sdf 文件
            mol = Chem.MolFromSmiles(smiles["smiles"])
            if mol is not None:
                try:
                    mol = AllChem.AddHs(mol)  # 添加氢原子
                    AllChem.EmbedMolecule(mol)  # 嵌入分子
                    AllChem.MMFFOptimizeMolecule(mol)  # 优化分子
                    init_sdf_dir = os.path.join(init_docking_dir, "sdf")
                    os.makedirs(init_sdf_dir, exist_ok=True)
                    init_sdf_path = os.path.join(
                        init_sdf_dir, f"{i+1}.sdf"
                    )  # 使用索引作为文件名
                    Chem.MolToMolFile(mol, init_sdf_path)  # 保存为 sdf 文件
                except Exception as e:
                    print(f"无效的 smiles: {smiles}, 错误信息:\n{e}")
                    continue
            else:
                print(f"无效的 smiles: {smiles}, 错误信息:\n{e}")
                continue
            """
            针对靶点进行分子对接
            """
            try:
                # normalized_score_mTOR, vina_output_mTOR = run_docking_and_normalize(
                #     ligand_sdf_path=init_sdf_path,
                #     receptor_pdbqt_path="./RL_utils/target_data/mTOR.pdbqt",
                #     center=(-9.2, 26.8, 35.8),
                #     box_size=(126, 126, 126),
                #     mu=-9.0,
                #     sigma=1.0,
                #     minimize=True,
                # )
                # normalized_score_MEK1, vina_output_MEK1 = run_docking_and_normalize(
                #     ligand_sdf_path=init_sdf_path,
                #     receptor_pdbqt_path="./RL_utils/target_data/MEK1.pdbqt",
                #     center=(60.7, -27.0, 12.8),
                #     box_size=(126, 126, 126),
                #     mu=-9.0,
                #     sigma=1.0,
                #     minimize=True,
                # )
                docking_result_dir = os.path.join(init_docking_dir, "docking_result")
                os.makedirs(docking_result_dir, exist_ok=True)
                # 初始化当前分子的分数
                score = 0.0
                first_scores_all = 0
                second_scores_all = 0
                if first_receptor:
                    first_best_score, first_vina_output = vina_docking(
                        ligand_sdf=init_sdf_path,
                        receptor_pdbqt=first_receptor_pdbqt_path,
                        center=first_center,
                        box_size=first_box_size,
                    )
                    # 单文件保存
                    docking_result_path = os.path.join(
                        docking_result_dir, f"{i+1}_{first_receptor}.txt"
                    )
                    with open(docking_result_path, "w") as f:
                        f.write(
                            f"{first_vina_output}\n\nbest docking score: {first_best_score}"
                        )
                    # 集中保存
                    smiles_dict_list[i][f"score_{first_receptor}"] = first_best_score
                    first_scores_all += first_best_score
                    score += first_best_score
                if second_receptor:
                    second_best_score, second_vina_output = vina_docking(
                        ligand_sdf=init_sdf_path,
                        receptor_pdbqt=second_receptor_pdbqt_path,
                        center=second_center,
                        box_size=second_box_size,
                    )
                    docking_result_path = os.path.join(
                        docking_result_dir, f"{i+1}_{second_receptor}.txt"
                    )
                    with open(docking_result_path, "w") as f:
                        f.write(
                            f"{second_vina_output}\n\nbest docking score: {second_best_score}"
                        )
                    smiles_dict_list[i][f"score_{second_receptor}"] = second_best_score
                    second_scores_all += second_best_score
                    score += second_best_score
                smiles_dict_list[i]["score"] = score
                valid_dacking += 1
            except Exception as e:
                print(f"\n在处理第 {i} 个分子时出现错误:\n{e}")
                continue
        scores_all = first_scores_all + second_scores_all
        smiles_dict_list[i]["score"] = scores_all
        print("分子对接过程的平均对接分数: ", scores_all / valid_dacking)
        print("靶点 1 的平均对接分数: ", first_scores_all / valid_dacking)
        print("靶点 2 的平均对接分数: ", second_scores_all / valid_dacking)
        smiles_dict_list = sorted(
            smiles_dict_list, key=lambda x: x["score"], reverse=True
        )
        df = pd.DataFrame(smiles_dict_list)
        df.to_csv(init_docking_result_path, header=True, index=False)
        print(f"初始分子对接过程的结果已保存至: {init_docking_result_path}")

    init_time = time.time()

    """
    此处是手动执行指定 epoch 的单步 RL 过程
    nohup python -u RL_onestep.py >RL_onestep.log 2>&1 &
    watch -n 2 -d nvidia-smi
    """
    if epoch == 1:
        before_docking_result_path = init_docking_result_path
        before_model_path = init_model_path
    else:
        before_docking_result_dir = os.path.join(
            RL_result_dir, "general_molecular", f"epoch_{epoch-1}"
        )
        os.makedirs(before_docking_result_dir, exist_ok=True)
        before_docking_result_path = os.path.join(
            before_docking_result_dir, f"epoch{epoch-1}_docking_result.csv"
        )  # epoch - 1 轮次分子对接过程的结果
        before_model_ckpt_dir = os.path.join(
            RL_result_dir, "model_ckpt", f"epoch_{epoch-1}"
        )
        os.makedirs(before_model_ckpt_dir, exist_ok=True)
        before_model_path = os.path.join(
            before_model_ckpt_dir, "ema_0.9999_030000.pt"
        )  # epoch - 1 轮次训练得到的模型参数文件

    print("*" * 100)
    print("强化学习过程")
    model_dir = os.path.join(RL_result_dir, "model_ckpt")
    os.makedirs(model_dir, exist_ok=True)
    model_check_dir = os.path.join(model_dir, f"epoch_{epoch}")
    os.makedirs(model_check_dir, exist_ok=True)
    model_check_path = os.path.join(model_check_dir, "*.pt")
    model_files = glob.glob(model_check_path)
    if model_files:
        print(f"在 {model_check_path} 中发现模型文件, 跳过训练")
        model_path = model_files[0]
    else:
        print("*" * 100)
        print(f"epoch_{epoch} 的训练过程开始")
        model_path = train_forRL(
            docking_score_path=before_docking_result_path,
            epoch_now=epoch,
            model_ckpt=before_model_path,
            nums=nums,
            config_file=config_file,
            model_save_dir=model_check_dir,
        )
        torch.cuda.empty_cache()
        print("*" * 100)
        print(f"epoch_{epoch} 的训练过程结束, 模型参数文件保存至 {model_path}")

    train_time = time.time()

    moleculars_dir = os.path.join(RL_result_dir, "general_molecular")
    os.makedirs(moleculars_dir, exist_ok=True)
    moleculars_epoch_dir = os.path.join(moleculars_dir, f"epoch_{epoch}")
    os.makedirs(moleculars_epoch_dir, exist_ok=True)
    general_check_path = os.path.join(moleculars_epoch_dir, "general_dataset.csv")
    if glob.glob(general_check_path):
        print(f"发现生成分子文件 {general_check_path}, 跳过生成过程")
        gen_df = pd.read_csv(general_check_path)
        gen_list = gen_df["smiles"].tolist()
    else:
        print("*" * 100)
        print(f"epoch_{epoch} 的生成过程开始")
        general_temp_path = os.path.join(moleculars_epoch_dir, "general_temp.csv")
        gen_list = (
            generation_forRL(
                model_path=model_path,
                general_nums=general_num,
                general_temp_path=general_temp_path,
                config_file=config_file,
            )
            or []
        )
        if gen_list:
            gen_df = pd.DataFrame({"smiles": gen_list})
            gen_df.to_csv(general_check_path, index=False)
        torch.cuda.empty_cache()
        print("*" * 100)
        print(f"epoch_{epoch} 的生成过程结束, 实际生成的有效分子数量 {len(gen_list)}")

    print("*" * 100)
    print(f"构建 epoch_{epoch} 分子对接过程的初始字典")
    # 截取出 nums 个前一轮的对接结果, 也就是参与本次训练的分子
    before_df = pd.read_csv(before_docking_result_path)
    before_dict_list = before_df[["smiles", "score"]].head(nums).to_dict("records")
    gen_dict_list = (
        [{"smiles": smiles, "score": 0} for smiles in gen_list] if gen_list else []
    )
    combined_dict_list = before_dict_list + gen_dict_list
    dict_list_path = os.path.join(moleculars_epoch_dir, "combined_dataset.csv")
    df = pd.DataFrame(combined_dict_list)
    df.to_csv(dict_list_path, header=True, index=False)
    print("*" * 100)
    print(f"epoch_{epoch} 的构建分子对接过程初始字典过程结束")
    print(f"初始字典中的分子数量 {len(combined_dict_list)}")

    general_time = time.time()

    print("*" * 100)
    print(f"开始执行 epoch_{epoch} 的分子对接过程")
    for i, smiles in tqdm(
        enumerate(combined_dict_list), total=len(combined_dict_list), desc="Docking——>"
    ):
        # 跳过已经对接过的分子
        if combined_dict_list[i]["score"] != 0:
            continue
        # 将 smiles 转换为 sdf 文件
        mol = Chem.MolFromSmiles(smiles["smiles"])
        if mol is not None:
            try:
                mol = AllChem.AddHs(mol)  # 添加氢原子
                AllChem.EmbedMolecule(mol)  # 嵌入分子
                AllChem.MMFFOptimizeMolecule(mol)  # 优化分子
                sdf_dir = os.path.join(moleculars_epoch_dir, "sdf")
                os.makedirs(sdf_dir, exist_ok=True)
                sdf_path = os.path.join(sdf_dir, f"{i+1}.sdf")  # 使用索引作为文件名
                Chem.MolToMolFile(mol, sdf_path)  # 保存为 sdf 文件
            except Exception as e:
                print(f"无效的 smiles: {smiles}, 错误信息:\n{e}")
                continue
        else:
            print(f"无效的 smiles: {smiles} 错误信息:\n{e}")
            continue
        # 开始执行分子对接
        scores_all = 0
        first_scores_all = 0
        second_scores_all = 0
        valid_dacking = 0
        try:
            docking_result_dir = os.path.join(RL_result_dir, "docking_result")
            os.makedirs(docking_result_dir, exist_ok=True)
            if first_receptor:
                first_best_score, first_vina_output = vina_docking(
                    ligand_sdf=sdf_path,
                    receptor_pdbqt=first_receptor_pdbqt_path,
                    center=first_center,
                    box_size=first_box_size,
                )
                # 单文件保存
                docking_result_path = os.path.join(
                    docking_result_dir, f"{i+1}_{first_receptor}.txt"
                )
                with open(docking_result_path, "w") as f:
                    f.write(
                        f"{first_vina_output}\n\nbest docking score: {first_best_score}"
                    )
                # 集中保存
                combined_dict_list[i][f"score_{first_receptor}"] = first_best_score
                first_scores_all += first_best_score
            if second_receptor:
                second_best_score, second_vina_output = vina_docking(
                    ligand_sdf=sdf_path,
                    receptor_pdbqt=second_receptor_pdbqt_path,
                    center=second_center,
                    box_size=second_box_size,
                )
                docking_result_path = os.path.join(
                    docking_result_dir, f"{i+1}_{second_receptor}.txt"
                )
                with open(docking_result_path, "w") as f:
                    f.write(
                        f"{second_vina_output}\n\nbest docking score: {second_best_score}"
                    )
                combined_dict_list[i][f"score_{second_receptor}"] = second_best_score
                second_scores_all += second_best_score
            valid_dacking += 1
        except Exception as e:
            print(f"在处理第 {i} 个分子时出现错误:\n{e}")
            continue
        scores_all = first_scores_all + second_scores_all
        combined_dict_list[i]["score"] = scores_all
    if valid_dacking != 0:
        print("分子对接过程的平均对接分数: ", scores_all / valid_dacking)
        print("靶点 1 的平均对接分数: ", first_scores_all / valid_dacking)
        print("靶点 2 的平均对接分数: ", second_scores_all / valid_dacking)
    combined_dict_list = sorted(
        combined_dict_list, key=lambda x: x["score"], reverse=True
    )
    df = pd.DataFrame(combined_dict_list)
    docking_result_path = os.path.join(
        moleculars_epoch_dir, f"epoch{epoch}_docking_result.csv"
    )
    df.to_csv(docking_result_path, header=True, index=False)
    print("*" * 100)
    print(f"对接并排序后的分子信息已保存至 {docking_result_path}")

    docking_time = time.time()

    time_total = docking_time - start_time
    time_init = init_time - start_time
    time_train = train_time - init_time
    time_general = general_time - train_time
    time_docking = docking_time - general_time

    def format_duration(seconds):
        total_sec = int(round(seconds))
        hours = total_sec // 3600
        remainder = total_sec % 3600
        minutes = remainder // 60
        seconds = remainder % 60
        if hours > 0:
            return f"{hours}时{minutes}分{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分{seconds}秒"
        else:
            return f"{seconds}秒"

    print(f"总耗时: {format_duration(time_total)}")
    print(f"初始化耗时: {format_duration(time_init)}")
    print(f"训练耗时: {format_duration(time_train)}")
    print(f"生成耗时: {format_duration(time_general)}")
    print(f"对接耗时: {format_duration(time_docking)}")


"""
运行指令: nohup python -u RL_onestep.py >RL_onestep.log 2>&1 &
监控资源占用: nvitop
监控资源占用: watch -n 2 -d nvidia-smi
"""
if __name__ == "__main__":
    epoch = 1  # 当下准备执行的第 epoch 次迭代
    nums = 200  # 截取前一步结果中分数较高的前 nums 个分子
    general_num = 150  # 目标生成分子数量
    config_file = "RL_train_config.json"
    init_moleculars_path = "generation_data/Moses_qedsa_ppgraph/sa_qed_ppgraph_Moses.csv"  # 第一阶段生成分子的保存文件
    init_model_path = (
        "weight/Moses_qedsa_ppgraph/ema_0.9999_100000.pt"  # 第一阶段模型的参数文件
    )
    RL_result_dir = "RL_result/Moses_qedsa_ppgraph_GLP1"  # 强化学习过程中各种结果和中间结果的保存路径
    first_receptor = "GLP1"  # 蛋白质靶点的名称
    first_receptor_pdbqt_path = (
        "data/docking_target/GLP1/GLP1_7s15.pdbqt"  # 蛋白质靶点的 pdbqt 文件的保存路径
    )
    first_center = (68.8, 70.2, 60.5)  # 蛋白质靶点的搜索空间的中心坐标
    first_box_size = (20.4, 20.3, 21.5)  # 蛋白质靶点的搜索空间的大小
    second_receptor = ""  # 蛋白质靶点的名称
    second_receptor_pdbqt_path = (
        ""  # 第二个 (如果有的话) 蛋白质靶点的 pdbqt 文件的保存路径
    )
    second_center = ()  # 蛋白质靶点的搜索空间的中心坐标
    second_box_size = ()  # 蛋白质靶点的搜索空间的大小
    main()
