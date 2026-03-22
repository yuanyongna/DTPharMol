import subprocess
import re
import numpy as np
from openbabel import openbabel


class GaussianModifier:
    """
    生成符合正态分布的分数修正函数。
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.0))


class MinMaxGaussianModifier:
    """
    半高斯分数修正函数。
    对于minimize==True，当x <= mu时函数值为1.0，x > mu时逐渐减少至0。
    对于minimize==False，当x >= mu时函数值为1.0，x < mu时逐渐减少至0。
    """

    def __init__(self, mu: float, sigma: float, minimize=False) -> None:
        self.mu = mu
        self.sigma = sigma
        self.minimize = minimize
        self._full_gaussian = GaussianModifier(mu=mu, sigma=sigma)

    def __call__(self, x):
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


def sdf_to_pdbqt(input_sdf, output_pdbqt):
    """将SDF文件转换为PDBQT文件。"""

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_sdf)

    # 生成3D坐标（如果需要）
    builder = openbabel.OBBuilder()
    builder.Build(mol)

    # 计算Gasteiger部分电荷
    mol.AddHydrogens()  # 添加氢原子
    # openbabel.OBChargeModel.FindType("gasteiger").ComputeCharges(mol)
    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    if charge_model:
        charge_model.ComputeCharges(mol)

    # 写入输出的PDBQT文件
    # obConversion.WriteFile(mol, output_pdbqt)
    success = obConversion.WriteFile(mol, output_pdbqt)
    if not success:
        raise ValueError(f"Failed to convert {input_sdf} to PDBQT")


def run_docking_and_normalize(
    ligand_sdf_path, receptor_pdbqt_path, center, box_size, mu, sigma, minimize=False
):
    # 将输入配体SDF文件转换为PDBQT格式
    ligand_pdbqt_path = ligand_sdf_path.replace(".sdf", ".pdbqt")
    sdf_to_pdbqt(ligand_sdf_path, ligand_pdbqt_path)

    # 准备AutoDock Vina命令和参数
    vina_command = [
        "vina",  # Vina 可执行文件的路径
        "--receptor",
        receptor_pdbqt_path,  # 受体 PDBQT 文件
        "--ligand",
        ligand_pdbqt_path,  # 配体 PDBQT 文件
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),  # 搜索空间的中心坐标
        "--size_x",
        str(box_size[0]),
        "--size_y",
        str(box_size[1]),
        "--size_z",
        str(box_size[2]),  # 搜索空间的大小
        "--exhaustiveness",
        "8",  # 对接搜索强度
        "--num_modes",
        "1",  # 最佳构象数目
    ]

    # 运行Vina命令，抑制输出
    result = subprocess.run(
        vina_command,
        stdout=subprocess.PIPE,  # 重定向标准输出
        stderr=subprocess.PIPE,  # 重定向标准错误
    )

    # 提取对接分数
    output = result.stdout.decode("utf-8")

    # 使用正则表达式找到亲和力（affinity）分数
    match = re.search(r"\d+\s+(-\d+\.\d+)\s+", output)
    if match:
        best_score = float(match.group(1))

        # 使用MinMaxGaussianModifier进行归一化
        modifier = MinMaxGaussianModifier(mu=mu, sigma=sigma, minimize=minimize)
        normalized_score = modifier(best_score)
        # normalized_score = modifier(-5)
        # return normalized_score
        return normalized_score, output
    else:
        raise ValueError("未找到亲和力分数，请检查输出格式。")


def vina_docking(ligand_sdf, receptor_pdbqt, center, box_size):

    # 将配体 sdf 文件格式转换为 pdbqt 格式
    ligand_pdbqt = ligand_sdf.replace(".sdf", ".pdbqt")
    sdf_to_pdbqt(ligand_sdf, ligand_pdbqt)

    # 准备 AutoDock Vina 命令和参数
    vina_command = [
        "vina",  # Vina 可执行文件的路径
        "--receptor",
        receptor_pdbqt,  # 受体 PDBQT 文件
        "--ligand",
        ligand_pdbqt,  # 配体 PDBQT 文件
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),  # 搜索空间的中心坐标
        "--size_x",
        str(box_size[0]),
        "--size_y",
        str(box_size[1]),
        "--size_z",
        str(box_size[2]),  # 搜索空间的大小
        "--energy_range",
        "4",  # 最大允许能量差值​​
        "--exhaustiveness",
        "12",  # 对接搜索强度
        "--num_modes",
        "1",  # 最佳构象数目
    ]

    # 运行 Vina 命令，抑制输出
    result = subprocess.run(
        vina_command,
        stdout=subprocess.PIPE,  # 重定向标准输出
        stderr=subprocess.PIPE,  # 重定向标准错误
    )
    if result.returncode != 0:
        print("Vina 错误信息:\n", result.stderr)

    # 使用正则表达式找到亲和力 (affinity) 分数
    output = result.stdout.decode("utf-8")
    match = re.search(r"\d+\s+(-\d+\.\d+)\s+", output)
    if match:
        best_score = float(match.group(1))
        return best_score, output
    else:
        raise ValueError("未找到亲和力分数")


# # 示例调用
# import psutil
# import os

# try:
#     current_process = psutil.Process(os.getpid())
#     print(f"当前进程 ID: {current_process.pid}")
#     print(f"当前进程名称: {current_process.name()}")
#     best_score, output = vina_docking(
#         ligand_sdf_path="./RL_utils/ligand_data/init/sdf/1.sdf",
#         receptor_pdbqt_path="./RL_utils/target_data/mTOR.pdbqt",
#         center=(-9.2, 26.8, 35.8),
#         box_size=(126, 126, 126),
#         mu=-7.0,  # 设置高斯分数修正的平均值
#         sigma=1.2,  # 设置高斯分数修正的标准差
#         minimize=True,  # 使用最小化模式
#     )
#     print(f"对接分数: {score}")
# except Exception as e:
#     print(f"出现错误: {e}")
