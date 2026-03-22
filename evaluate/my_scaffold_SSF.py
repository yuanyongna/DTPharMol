#!/usr/bin/env python3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # 禁止 RDKit 日志输出

# ========================
# 文件路径和条件骨架设置
# ========================
"""
检查有效性: 100%|███████████████████████████████| 11815/11815 [00:00<00:00, 12734.32it/s]
✅ Validity: 1.0000
✅ Uniqueness: 0.9990
✅ Novelty: 0.9407
🧪 用于多样性计算的分子数: 11803
计算相似度: 100%|████████████████████████████████| 11803/11803 [00:03<00:00, 3636.62it/s]
✅ Diversity（1-平均相似度）: 0.8058
✅ SSF（Same Scaffold Fraction）: 0.9961 (11769/11815)

============================================================
📊 最终评估指标总结
============================================================
Validity : 1.0000
Uniqueness : 0.9990
Novelty : 0.9407
Diversity : 0.8058
SSF : 0.9961 (11769/11815)
============================================================
"""
# train_file = "./data/Moses.csv"
# generated_file = (
#     "./generation_data/Moses_qedsa_scaffold_c1ccccc1/Moses_qedsa_scaffold.csv"
# )
# condition_scaffold_smiles = "c1ccccc1"

"""
检查有效性: 100%|████████████████████| 9653/9653 [00:01<00:00, 9195.73it/s]
✅ Validity: 1.0000
✅ Uniqueness: 0.9987
✅ Novelty: 0.9660
🧪 用于多样性计算的分子数: 9640
计算相似度: 100%|█████████████████████| 9640/9640 [00:02<00:00, 4411.17it/s]
✅ Diversity（1-平均相似度）: 0.7607
✅ SSF（Same Scaffold Fraction）: 0.9336 (9012/9653)

============================================================
📊 最终评估指标总结
============================================================
Validity : 1.0000
Uniqueness : 0.9987
Novelty : 0.9660
Diversity : 0.7607
SSF : 0.9336 (9012/9653)
============================================================
"""
# train_file = "./data/Moses.csv"
# generated_file = "./generation_data/Moses_qedsa_scaffold_O=C(Nc1ccccc1)c1ccccc1/Moses_qedsa_scaffold.csv"
# condition_scaffold_smiles = "O=C(Nc1ccccc1)c1ccccc1"

"""
检查有效性: 100%|█████████████████████████████████| 9498/9498 [00:01<00:00, 9051.90it/s]
✅ Validity: 1.0000
✅ Uniqueness: 0.9954
✅ Novelty: 0.9678
🧪 用于多样性计算的分子数: 9454
计算相似度: 100%|█████████████████████████████████| 9454/9454 [00:02<00:00, 4490.94it/s]
✅ Diversity（1-平均相似度）: 0.7759
✅ SSF（Same Scaffold Fraction）: 0.7179 (6819/9498)

============================================================
📊 最终评估指标总结
============================================================
Validity : 1.0000
Uniqueness : 0.9954
Novelty : 0.9678
Diversity : 0.7759
SSF : 0.7179 (6819/9498)
============================================================
"""
# train_file = "./data/Moses.csv"
# generated_file = "./generation_data/Moses_qedsa_scaffold_O=C(NCc1ccccc1)c1ccccc1/Moses_qedsa_scaffold.csv"
# condition_scaffold_smiles = "O=C(NCc1ccccc1)c1ccccc1"

"""
检查有效性: 100%|█████████████████████████████████| 9271/9271 [00:01<00:00, 9164.12it/s]
✅ Validity: 1.0000
✅ Uniqueness: 0.9943
✅ Novelty: 0.9337
🧪 用于多样性计算的分子数: 9218
计算相似度: 100%|█████████████████████████████████| 9218/9218 [00:01<00:00, 4646.52it/s]
✅ Diversity（1-平均相似度）: 0.7265
✅ SSF（Same Scaffold Fraction）: 0.8211 (7612/9271)

============================================================
📊 最终评估指标总结
============================================================
Validity : 1.0000
Uniqueness : 0.9943
Novelty : 0.9337
Diversity : 0.7265
SSF : 0.8211 (7612/9271)
============================================================
"""
# train_file = "./data/Moses.csv"
# generated_file = (
#     "./generation_data/Moses_qedsa_scaffold_c1ccncc1/Moses_qedsa_scaffold.csv"
# )
# condition_scaffold_smiles = "c1ccncc1"

"""
检查有效性: 100%|██████████████████████████████| 11426/11426 [00:00<00:00, 13494.78it/s]
✅ Validity: 1.0000
✅ Uniqueness: 0.9889
✅ Novelty: 0.9874
🧪 用于多样性计算的分子数: 11299
计算相似度: 100%|███████████████████████████████| 11299/11299 [00:03<00:00, 3752.53it/s]
✅ Diversity（1-平均相似度）: 0.7731
✅ SSF（Same Scaffold Fraction）: 0.9989 (11413/11426)

============================================================
📊 最终评估指标总结
============================================================
Validity : 1.0000
Uniqueness : 0.9889
Novelty : 0.9874
Diversity : 0.7731
SSF : 0.9989 (11413/11426)
============================================================
"""
train_file = "./data/Moses.csv"
generated_file = (
    "./generation_data/Moses_qedsa_scaffold_c1ccncc1/Moses_qedsa_scaffold.csv"
)
condition_scaffold_smiles = "c1ccncc1"


# ========================
# 读数据
# ========================
def load_data(file_path):
    df = pd.read_csv(file_path)
    if "smiles" not in df.columns:
        raise ValueError(f"{file_path} 中未找到 'smiles' 列")
    return df


train_df = load_data(train_file)
generated_df = load_data(generated_file)

train_smiles = train_df["smiles"].dropna().astype(str).tolist()
generated_smiles = generated_df["smiles"].dropna().astype(str).tolist()


# ========================
# 1. Validity
# ========================
def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


valid_indices, valid_mols = [], []
for idx, smi in enumerate(tqdm(generated_smiles, desc="检查有效性")):
    if is_valid_smiles(smi):
        valid_indices.append(idx)
        valid_mols.append(smi)

valid_count = len(valid_mols)
validity = valid_count / len(generated_smiles)
print(f"✅ Validity: {validity:.4f}")

# ========================
# 2. Uniqueness
# ========================
unique_smiles = list(set(valid_mols))
uniqueness = len(unique_smiles) / valid_count
print(f"✅ Uniqueness: {uniqueness:.4f}")

# ========================
# 3. Novelty
# ========================
train_set = set(train_smiles)
novel_mols = [smi for smi in unique_smiles if smi not in train_set]
novelty = len(novel_mols) / len(unique_smiles)
print(f"✅ Novelty: {novelty:.4f}")


# ========================
# 4. Diversity
# ========================
def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


fps = [
    get_fingerprint(smi) for smi in unique_smiles if get_fingerprint(smi) is not None
]
print(f"🧪 用于多样性计算的分子数: {len(fps)}")

similarities = []
for i in tqdm(range(len(fps)), desc="计算相似度"):
    sims = BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
    similarities.extend(sims)

avg_sim = np.mean(similarities) if similarities else 0
diversity = 1 - avg_sim
print(f"✅ Diversity（1-平均相似度）: {diversity:.4f}")


# ========================
# 5. SSF（Same Scaffold Fraction）
# ========================
def get_murcko_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold_smi
    except:
        return None


condition_scaffold = Chem.MolFromSmiles(condition_scaffold_smiles)
if condition_scaffold is None:
    raise ValueError("条件骨架 SMILES 无效")

# 生成集中的骨架
scaffold_matches = 0
scaffold_total = 0
for smi in valid_mols:
    scaf_smi = get_murcko_scaffold(smi)
    if scaf_smi is not None:
        scaffold_total += 1
        if scaf_smi == condition_scaffold_smiles:
            scaffold_matches += 1

SSF = scaffold_matches / scaffold_total if scaffold_total > 0 else 0
print(
    f"✅ SSF（Same Scaffold Fraction）: {SSF:.4f} ({scaffold_matches}/{scaffold_total})"
)

# ========================
# 汇总输出
# ========================
print("\n" + "=" * 60)
print("📊 最终评估指标总结")
print("=" * 60)
print(f"Validity : {validity:.4f}")
print(f"Uniqueness : {uniqueness:.4f}")
print(f"Novelty : {novelty:.4f}")
print(f"Diversity : {diversity:.4f}")
print(f"SSF : {SSF:.4f} ({scaffold_matches}/{scaffold_total})")
print("=" * 60)
