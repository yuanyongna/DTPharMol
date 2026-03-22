from openbabel import openbabel
import subprocess
import re
import os
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_sdf(smiles, output_sdf):
    """将SMILES字符串转换为SDF文件"""
    try:
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法从SMILES创建分子: {smiles}")

        # 添加氢原子并生成3D构象
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        # 保存为SDF文件
        writer = Chem.SDWriter(output_sdf)
        writer.write(mol)
        writer.close()

        return output_sdf

    except Exception as e:
        raise RuntimeError(f"SMILES转SDF失败: {str(e)}")


def sdf_to_pdbqt(input_sdf, output_pdbqt):
    """将SDF文件转换为PDBQT文件"""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdbqt")
    mol = openbabel.OBMol()

    # 读取SDF文件
    if not obConversion.ReadFile(mol, input_sdf):
        raise ValueError(f"无法读取SDF文件: {input_sdf}")

    # 生成3D坐标（如果需要）
    builder = openbabel.OBBuilder()
    builder.Build(mol)

    # 添加氢原子并计算Gasteiger电荷
    mol.AddHydrogens()
    charge_model = openbabel.OBChargeModel.FindType("gasteiger")
    if charge_model:
        charge_model.ComputeCharges(mol)

    # 写入PDBQT文件
    if not obConversion.WriteFile(mol, output_pdbqt):
        raise ValueError(f"转换失败: {input_sdf} -> {output_pdbqt}")

    return output_pdbqt


def run_docking(
    ligand_smiles: str,
    receptor_pdbqt: str,
    center: tuple,
    box_size: tuple,
    num_modes: int = 1,
    output_complex_pdbqt: str = "docked_complex.pdbqt",
    temp_dir: str = "temp",
) -> float:
    """
    从SMILES字符串执行分子对接

    参数:
        ligand_smiles: 配体SMILES字符串
        receptor_pdbqt: 受体PDBQT文件路径
        center: 对接中心坐标 (x, y, z)
        box_size: 对接盒子尺寸 (x, y, z)
        num_modes: 保存的最佳构象数量
        output_complex_pdbqt: 对接后复合物输出路径
        temp_dir: 临时文件存储目录

    返回:
        best_score: 最佳对接分数
    """
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)

    # 生成唯一文件名
    import uuid

    temp_id = uuid.uuid4().hex[:8]

    try:
        # 步骤1: SMILES -> SDF
        temp_sdf = os.path.join(temp_dir, f"ligand_{temp_id}.sdf")
        smiles_to_sdf(ligand_smiles, temp_sdf)

        # 步骤2: SDF -> PDBQT
        temp_pdbqt = os.path.join(temp_dir, f"ligand_{temp_id}.pdbqt")
        sdf_to_pdbqt(temp_sdf, temp_pdbqt)

        # 准备Vina对接命令
        vina_command = [
            "vina",
            "--receptor",
            receptor_pdbqt,
            "--ligand",
            temp_pdbqt,
            "--out",
            output_complex_pdbqt,
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(box_size[0]),
            "--size_y",
            str(box_size[1]),
            "--size_z",
            str(box_size[2]),
            "--num_modes",
            str(num_modes),
            "--energy_range",
            "4",
            "--exhaustiveness",
            "12",
        ]

        # 执行对接命令
        result = subprocess.run(
            vina_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # 检查执行结果
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            raise RuntimeError(f"对接失败: {error_msg}")

        # 解析对接结果
        output = result.stdout
        best_score = None

        # 查找最佳对接分数（模式1的分数）
        pattern = r"^\s*1\s+([-\d.]+)\s+"
        match = re.search(pattern, output, re.MULTILINE)

        if match:
            best_score = float(match.group(1))
        else:
            raise ValueError("无法从输出中解析对接分数")

        print(f"对接完成! 最佳分数: {best_score:.2f} kcal/mol")
        print(f"复合物结构已保存至: {output_complex_pdbqt}")

        return best_score

    finally:
        # 清理临时文件
        if os.path.exists(temp_sdf):
            os.remove(temp_sdf)
        if os.path.exists(temp_pdbqt):
            os.remove(temp_pdbqt)


# 使用示例
if __name__ == "__main__":
    # 输入参数
    ligand_smiles = "CC(C)c1ccc(CC(=O)Nc2ccc3c(c2)CCC(=O)N3)cc1"
    receptor_pdbqt = "data/docking_target/CDK2/CDK2_1h00.pdbqt"  # 受体PDBQT文件
    center = (1.7, 26.2, 8.7)  # 对接中心坐标
    box_size = (25, 25, 25)  # 对接盒子尺寸
    num_modes = 1  # 最佳构象数量
    output_complex = "evaluate/results/docking_analysis/CDK2/11793__CC(C)c1ccc(CC(=O)Nc2ccc3c(c2)CCC(=O)N3)cc1/11793.pdbqt"  # 输出文件

    # 执行对接
    try:
        score = run_docking(
            ligand_smiles=ligand_smiles,
            receptor_pdbqt=receptor_pdbqt,
            center=center,
            box_size=box_size,
            num_modes=num_modes,
            output_complex_pdbqt=output_complex,
        )
        print(f"对接成功! 最佳分数: {score:.2f} kcal/mol")
    except Exception as e:
        print(f"对接过程中出错: {str(e)}")


"""
1.得到靶点和配体对接后的复合物的pdbqt文件。
2.在pymol中打开，同时打开靶点的pdbqt文件。
3.file -> export -> 选中配体和靶点 -> ok -> 保存为pdb文件。
4.在PLIP中打开，并analysis分析，之后下载pse文件。
5.在pymol中打开pse文件：
在右侧打开Structures的加号展开；
选中展开后第一个ProteinCarto，并点击S后选择Cartoon；
黄色的是配体，深蓝色的是相互作用；
set cartoon_transparency, 0.5：调整蛋白质的透明度；
set label_size, 30：修改标签大小；
点击每一个深蓝色的相互作用，就是右侧的sele，点击L后选择第二个residues；
点击右下角框里面右上角的Viewing，改为Editing，按住Ctrl用鼠标拖动标签文字；
set ray_opaque_background, 1：设置图片背景非透明；
png D://study//scientific_research//VinaDocking//docking_analysis//CDK2//05_CDK2.png, 4000,4000,ray=1：导出为高清图片
"""
