import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


def get_murcko_scaffold(smiles):
    """计算Murcko支架"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else None


def extract_murcko_scaffolds_and_count(similes_path: str):
    # 1. 读取所有smiles数据
    data = pd.read_csv(similes_path)
    smiles_list = data["SMILES"].tolist()

    # 2. 提取 MURCO 支架
    scaffolds = []
    for smile in smiles_list:
        scaffold = get_murcko_scaffold(smile)
        if scaffold:
            scaffolds.append(scaffold)

    # 3. 统计支架出现次数
    scaffold_count = defaultdict(int)
    for scaffold in scaffolds:
        scaffold_count[scaffold] += 1

    # 4. 输出数量最多的五种支架
    sorted_scaffolds = sorted(scaffold_count.items(), key=lambda x: x[1], reverse=True)
    top_five_scaffolds = sorted_scaffolds[:5]

    return top_five_scaffolds


top_scaffolds = extract_murcko_scaffolds_and_count("./data/moses2.csv")
print("数量最多的五种 MURCO 支架: ")
for scaffold, count in top_scaffolds:
    print(f"支架: {scaffold}, 计数: {count}")

"""
数量最多的五种 MURCO 支架: 
支架: c1ccccc1, 计数: 88559
支架: O=C(Nc1ccccc1)c1ccccc1, 计数: 18080
支架: O=C(NCc1ccccc1)c1ccccc1, 计数: 8079
支架: O=C(COc1ccccc1)Nc1ccccc1, 计数: 7029
支架: c1ccncc1, 计数: 7006
"""
