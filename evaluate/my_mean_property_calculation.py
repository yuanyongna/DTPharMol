import pandas as pd
import warnings
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem
from collections import defaultdict

warnings.filterwarnings("ignore")

def calculate_average_properties(df):
    """计算指定属性的平均值"""
    averages = {
        'Average LogP': df['logp'].mean(),
        'Average QED': df['qed'].mean(),
        'Average SAS': df['SAS'].mean(),
        'Average TPSA': df['TPSA'].mean()
    }
    
    for key, value in averages.items():
        print(f'{key}: {value}')

def get_murcko_scaffold(smiles):
    """计算Murcko支架"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else None

def get_fingerprint(smiles):
    """计算分子的指纹"""
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2) if mol else None

def cluster_scaffolds(df, similarity_threshold):
    """骨架相似性聚类"""
    df['scaffold'] = df['SMILES'].apply(get_murcko_scaffold)
    df['fingerprint'] = df['SMILES'].apply(get_fingerprint)

    buckets = defaultdict(list)

    for _, row in df.iterrows():
        scaffold = row['scaffold']
        fingerprint = row['fingerprint']

        # 聚类逻辑
        for existing_row in buckets[scaffold]:
            existing_fingerprint = existing_row['fingerprint']
            if FingerprintSimilarity(fingerprint, existing_fingerprint) >= similarity_threshold:
                buckets[scaffold].append(row)
                break
        else:
            buckets[scaffold].append(row)

    # 统计骨架出现次数和分子数量
    scaffold_counts = {scaffold: len(compounds) for scaffold, compounds in buckets.items()}
    return sorted(scaffold_counts.items(), key=lambda x: x[1], reverse=True)[:10]

def main(file, similarity, sample_size):
    """主计算函数"""
    df = pd.read_csv(file)

    # 计算平均属性
    calculate_average_properties(df)

    # 随机抽样指定数量的分子
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=1)  # random_state用于可复现性

    # 进行骨架相似性聚类并输出结果
    top_scaffolds = cluster_scaffolds(df, similarity)
    for scaffold, count in top_scaffolds:
        print(f'Scaffold: {scaffold}, Molecule Count: {count}')


"""
nohup python -u data/mean_property_calculation.py >data/mean_property_calculation.log 2>&1 &
"""
if __name__ == "__main__":
    data_file = 'data/small_moses2.csv'
    similarity_threshold = 0.8
    sample_size = 100000  # 指定随机抽样的分子数量
    main(data_file, similarity_threshold, sample_size)
