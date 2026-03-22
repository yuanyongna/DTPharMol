"""
nohup python -u evaluate/get_metrics.py >Get_metrics.log 2>&1 &
"""

import pandas as pd
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem
from utils import check_novelty, canonic_smiles


path = "./generation_data/all_conditions_Moses/all_conditions_Moses.csv"
props = []
scaffold_cond = False
scaffold_cond = ""


def calculate_un(results, moses):
    # 计算唯一性和新颖性比例
    canon_smiles = [canonic_smiles(s) for s in results["smiles"]]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = check_novelty(
        unique_smiles, set(moses[moses["split"] == "train"]["smiles"])
    )  # replace 'source' with 'split' for moses
    return len(unique_smiles) / len(results), novel_ratio


if __name__ == "__main__":
    data = pd.read_csv(path)
    print("生成数据集的列名 data.columns:\n", data.columns)
    moses = pd.read_csv("data/moses2.csv")
    moses = moses.dropna(axis=0).reset_index(drop=True)
    moses.columns = moses.columns.str.lower()
    if scaffold_cond:
        data["scaffold_cond"] = scaffold_cond
        # 处理 scaffold_cond 列
        data["scaffold_cond"] = data["scaffold_cond"].apply(
            lambda x: x.replace("<", "")
        )
        # 生成分子骨架和指纹
        data["mol_scaf"] = data["smiles"].apply(lambda x: MurckoScaffoldSmiles(x))
        data["fp"] = data["mol_scaf"].apply(
            lambda x: FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x))
        )
        data["cond_fp"] = data["scaffold_cond"].apply(
            lambda x: FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x))
        )
        # 初始化相似度列
        data["similarity"] = -1
        # # 计算相似度
        for idx, row in data.iterrows():
            data.loc[idx, "similarity"] = TanimotoSimilarity(row["fp"], row["cond_fp"])
        # # 计算每个 scaffold_cond 的样本数量, 以及相似度为 1 的样本数量
        x = data["scaffold_cond"].value_counts()
        y = data[data["similarity"] == 1]["scaffold_cond"].value_counts()
        print("相似度为 1 的比例: ", y.divide(x))
        new_df = []
        for cond in data["scaffold_cond"].unique():
            # # 计算当前条件的样本数量
            scaffold_samples = len(
                data[data["scaffold_cond"] == cond].reset_index(drop=True)
            )
            # # 获取相似度大于 0.8 的结果
            results = data[
                (data["scaffold_cond"] == cond) & (data["similarity"] > 0.8)
            ].reset_index(drop=True)
            val = len(results) / scaffold_samples
            previous_validity = results["validity"][0]
            uniqueness, novelty = calculate_un(results, moses)
            results["validity"] = val * previous_validity
            results["unique"] = uniqueness
            results["novelty"] = novelty
            new_df.append(results)
        # # 合并所有条件的结果
        data = pd.concat(new_df).reset_index(drop=True)
        # # 计算并打印有效性、唯一性和新颖性的平均值
        avg_validity = data.groupby("scaffold_cond")["validity"].mean()
        avg_unique = data.groupby("scaffold_cond")["unique"].mean()
        avg_novelty = data.groupby("scaffold_cond")["novelty"].mean()
        print(
            "Validity \n",
            avg_validity,
            "\n Uniqueness \n",
            avg_unique,
            "\n Novelty \n",
            avg_novelty,
        )
        if len(props) == 1:
            data["difference"] = abs(data["condition"] - data[props[0]])
            print(f"\n Mean Absolute Difference: {props[0]} \n")
            print(data.groupby("scaffold_cond")["difference"].mean())
            print(f"\n Standard Deviation of the Difference: {props[0]} \n")
            print(data.groupby("scaffold_cond")["difference"].std())
        if len(props) > 1:
            for idx, p in enumerate(props):
                data[f"{p}_condition"] = data["condition"].apply(
                    lambda x: tuple(float(s) for s in x.strip("()").split(","))[idx]
                )
                data["difference"] = abs(data[f"{p}_condition"] - data[p])
                print(f"\n Mean Absolute Difference: {p} \n")
                print(data.groupby("scaffold_cond")["difference"].mean())
                print(f"\n Standard Deviation of the Difference: {p} \n")
                print(data.groupby("scaffold_cond")["difference"].std())
        else:
            pass
    else:
        avg_validity = data["validity"].mean()
        avg_unique = data["unique"].mean()
        avg_novelty = data["novelty"].mean()
        print(
            "Validity \n",
            avg_validity,
            "\n Uniqueness \n",
            avg_unique,
            "\n Novelty \n",
            avg_novelty,
        )
        if len(props) == 1:
            data["difference"] = abs(data["condition"] - data[props[0]])
            print(f"\n Mean Absolute Difference: {props[0]} \n")
            print(data["difference"].mean())
            print(f"\n Standard Deviation of the Difference: {props[0]} \n")
            print(data["difference"].std())
        else:
            for idx, p in enumerate(props):
                data[f"{p}_condition"] = data["condition"].apply(
                    lambda x: tuple(float(s) for s in x.strip("()").split(","))[idx]
                )
                data["difference"] = abs(data[f"{p}_condition"] - data[p])
                print(f"\n Mean Absolute Difference: {p} \n")
                print(data["difference"].mean())
                print(f"\n Standard Deviation of the Difference: {p} \n")
                print(data["difference"].std())
