#!/usr/bin/env python3
"""
手动计算生成分子的四大指标：
- Validity：有效分子比例（能被 RDKit 解析）
- Uniqueness：有效分子中去重比例
- Novelty：生成分子中未在训练集中出现的比例
- Diversity：有效唯一分子间的平均 Tanimoto 相似度（越低越多样）
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from tqdm import tqdm  # 可选，用于进度条

# -------------------------------
# 文件路径
# -------------------------------
train_file = "./data/Moses.csv"  # 训练集，需有 smiles 列
generated_file = "./RL_result/Moses_qedsa_ppgraph_MEK1_mTOR/prepare/docking_result.csv"  # 生成集，需有 smiles 列


# -------------------------------
# 读取数据，只保留 smiles 列
# -------------------------------
def load_smiles(file_path):
    df = pd.read_csv(file_path)
    if "smiles" not in df.columns:
        raise ValueError(f"文件 {file_path} 中未找到 'smiles' 列")
    return df["smiles"].dropna().astype(str).tolist()


train_smiles = load_smiles(train_file)
generated_smiles = load_smiles(generated_file)

print(f"📥 训练集样本数: {len(train_smiles)}")
print(f"📥 生成集样本数: {len(generated_smiles)}")


# -------------------------------
# 1. Validity：有效分子比例
# -------------------------------
def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


valid_mols = [smi for smi in tqdm(generated_smiles, desc="检查有效性")]
valid_mols = [smi for smi in valid_mols if is_valid_smiles(smi)]

valid_count = len(valid_mols)
total_count = len(generated_smiles)
validity = valid_count / total_count if total_count > 0 else 0.0

print(f"✅ Validity（有效性）: {validity:.4f} ({valid_count}/{total_count})")

# -------------------------------
# 2. Uniqueness：唯一性（有效分子中去重比例）
# -------------------------------
unique_smiles = list(set(valid_mols))
uniqueness = len(unique_smiles) / valid_count if valid_count > 0 else 0.0

print(f"✅ Uniqueness（唯一性）: {uniqueness:.4f} ({len(unique_smiles)}/{valid_count})")

# -------------------------------
# 3. Novelty：新颖性（未出现在训练集中的比例）
# -------------------------------
train_set = set(train_smiles)  # 训练集所有 smiles


def is_novel(smi):
    return smi not in train_set


novel_mols = [smi for smi in unique_smiles if is_novel(smi)]
novelty = len(novel_mols) / len(unique_smiles) if len(unique_smiles) > 0 else 0.0

print(f"✅ Novelty（新颖性）: {novelty:.4f} ({len(novel_mols)}/{len(unique_smiles)})")


# -------------------------------
# 4. Diversity：多样性（平均 Tanimoto 相似度）
# -------------------------------
def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


# 获取所有唯一有效分子的指纹
unique_mol_list = unique_smiles  # 已经是去重后的有效分子
fingerprints = []
valid_fps = []

for smi in tqdm(unique_mol_list, desc="生成指纹"):
    fp = get_fingerprint(smi)
    if fp is not None:
        valid_fps.append(fp)

print(f"🧪 用于计算多样性的分子数: {len(valid_fps)}")

if len(valid_fps) < 2:
    diversity = 0.0
    print(f"⚠️  多样性：分子对不足 2 个，无法计算，设为 {diversity}")
else:
    # 计算所有分子两两之间的 Tanimoto 相似度，然后取平均
    similarity_matrix = []
    for i in tqdm(range(len(valid_fps)), desc="计算相似度"):
        sims = BulkTanimotoSimilarity(valid_fps[i], valid_fps[i + 1 :])
        similarity_matrix.extend(sims)

    if len(similarity_matrix) > 0:
        avg_similarity = np.mean(similarity_matrix)
        # 多样性定义为 1 - 平均相似度，或者直接报告平均相似度（越低越多样）
        diversity = 1.0 - avg_similarity  # 或者直接用 avg_similarity 表示“平均相似度”
        print(f"✅ Diversity（多样性，1 - 平均相似度）: {diversity:.4f}")
        print(f"   （平均 Tanimoto 相似度: {avg_similarity:.4f}）")
    else:
        diversity = 0.0
        print(f"⚠️  多样性：无法计算，设为 {diversity}")

# -------------------------------
# 所有指标总结
# -------------------------------
print("\n" + "=" * 50)
print("📊 最终评估指标总结")
print("=" * 50)
print(f"1. Validity（有效性）: {validity:.4f}")
print(f"2. Uniqueness（唯一性）: {uniqueness:.4f}")
print(f"3. Novelty（新颖性）: {novelty:.4f}")
print(f"4. Diversity（多样性，1 - 平均相似度）: {diversity:.4f}")
