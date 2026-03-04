import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
from collections import Counter
from rdkit.Chem import rdmolops


def fragment_complexity(smiles):
    """计算分子片段的复杂性分数"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    # 使用 BRICS 生成分子片段
    fragments = list(BRICS.BRICSDecompose(mol))
    # 计算分子片段的数量和多样性
    fragment_set = set(fragments)
    diversity_score = len(fragment_set) / len(fragments) if fragments else 0
    return diversity_score


def shannon_entropy(smiles):
    """计算香农熵复杂性分数"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_counts = Counter(atom_types)
    total_atoms = sum(atom_counts.values())
    probabilities = [count / total_atoms for count in atom_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy


def ring_complexity(smiles):
    """计算环结构复杂性分数"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetRingInfo().NumRings()


def symmetry_complexity(smiles):
    """计算对称性复杂性分数"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        symmetry = rdmolops.CalcSymmSSSR(mol)
        return 1 / (1 + len(symmetry)) if symmetry else 0
    except:
        return 0


def compute_complexity(smiles, w_frag=0.4, w_entropy=0.3, w_ring=0.2, w_sym=0.1):
    """计算综合复杂性分数"""
    return (
        w_frag * fragment_complexity(smiles)
        + w_entropy * shannon_entropy(smiles)
        + w_ring * ring_complexity(smiles)
        + w_sym * symmetry_complexity(smiles)
    )


def add_complexity_to_csv(input_file, output_file=None, weights=None):
    """
    读取CSV文件，为每个分子计算复杂性分数，并保存到新列

    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（如果为None，则覆盖原文件）
        weights: 复杂性分数的权重元组 (w_frag, w_entropy, w_ring, w_sym)
    """
    # 读取数据
    try:
        df = pd.read_csv(input_file)
        print(f"成功读取文件: {input_file}, 共 {len(df)} 行")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    # 确保有smiles列
    if "smiles" not in df.columns:
        print("错误: 数据文件中缺少'smiles'列")
        return None

    # 如果输出文件未指定，覆盖原文件
    if output_file is None:
        output_file = input_file
        print(f"将覆盖原文件: {input_file}")

    # 检查是否已有complexity_score列
    if "complexity_score" in df.columns:
        print("警告: 文件中已存在'complexity_score'列，将被覆盖")

    # 设置权重
    if weights is None:
        weights = (0.4, 0.3, 0.2, 0.1)

    # 计算复杂性分数
    print("开始计算复杂性分数...")

    # 创建进度条
    progress = tqdm(total=len(df), desc="计算复杂性", unit="分子")

    # 计算每个分子的复杂性分数
    complexity_scores = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        try:
            score = compute_complexity(
                smiles,
                w_frag=weights[0],
                w_entropy=weights[1],
                w_ring=weights[2],
                w_sym=weights[3],
            )
        except Exception as e:
            print(f"计算复杂性时出错 ({smiles}): {e}")
            score = 0.0

        complexity_scores.append(score)

        # 更新进度条
        progress.update(1)
        progress.set_postfix(
            {
                "当前分子": smiles[:20] + "..." if len(smiles) > 20 else smiles,
                "分数": f"{score:.4f}",
            }
        )

    # 关闭进度条
    progress.close()

    # 将复杂性分数添加到数据框
    print("将复杂性分数添加到数据框...")
    df["complexity"] = complexity_scores

    # 保存结果
    try:
        df.to_csv(output_file, index=False)
        print(f"成功保存结果到: {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None

    return df


if __name__ == "__main__":
    # 1. 设置输入文件路径
    input_file = "data/Moses.csv"

    # 2. 设置输出文件路径（设为None将覆盖原文件）
    output_file = "data/Moses_complexity.csv"

    # 3. 设置权重（可选，默认为(0.4, 0.3, 0.2, 0.1)）
    custom_weights = (0.25, 0.25, 0.25, 0.25)

    print("开始计算分子复杂性...")
    result_df = add_complexity_to_csv(
        input_file=input_file, output_file=output_file, weights=custom_weights
    )

    if result_df is not None:
        print("\n处理结果摘要:")
        print(f"分子总数: {len(result_df)}")
        print(f"平均复杂性分数: {result_df['complexity_score'].mean():.4f}")
        print(f"最小复杂性分数: {result_df['complexity_score'].min():.4f}")
        print(f"最大复杂性分数: {result_df['complexity_score'].max():.4f}")

        print("\n前5个分子预览:")
        print(result_df[["smiles", "complexity_score"]].head())

        print("\n处理完成！")
