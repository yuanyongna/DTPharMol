# coding: utf-8
import time
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA

from miscs.my_calculateScore import my_calculateScore


def calculate_molecular_properties(input_csv, output_csv, chunksize=10000):
    """
    批量计算分子属性并保存结果
    Args:
        input_csv: 输入 CSV 文件路径
        output_csv: 输出 CSV 文件路径
        chunksize: 分块处理大小 (内存优化)
    """
    # 定义属性计算器
    property_calculators = [
        ("mw", lambda m: Descriptors.MolWt(m)),
        ("logp", lambda m: Crippen.MolLogP(m)),  # Descriptors.MolLogP
        (
            "hba",
            lambda m: Lipinski.NumHAcceptors(m),
        ),  # rdMolDescriptors.CalcNumLipinskiHBA
        (
            "hbd",
            lambda m: Lipinski.NumHDonors(m),
        ),  # rdMolDescriptors.CalcNumLipinskiHBD
        ("qed", lambda m: QED.qed(m)),
        ("sa", lambda m: my_calculateScore(m)),
        ("tpsa", lambda m: CalcTPSA(m)),  # Descriptors.TPSA
        ("NumRotBonds", lambda m: CalcNumRotatableBonds(m)),
        (
            "chiral_center",
            lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
        ),
    ]

    # 初始化统计指标
    stats = {"total": 0, "success": 0, "failed": 0}
    start_time = time.time()

    # 分块读取文件
    reader = pd.read_csv(input_csv, chunksize=chunksize)

    for chunk_idx, df_chunk in enumerate(reader):
        # 添加分子列
        df_chunk["mol"] = df_chunk["smiles"].apply(
            lambda s: Chem.MolFromSmiles(str(s)) if pd.notnull(s) else None
        )

        # 初始化结果容器
        results = []

        # 迭代处理
        with tqdm(
            df_chunk.iterrows(),
            total=len(df_chunk),
            desc=f"Processing chunk {chunk_idx + 1}",
            unit="mol",
        ) as pbar:

            for _, row in pbar:
                stats["total"] += 1
                mol = row["mol"]
                prop_dict = {"smiles": row["smiles"], "flag": False}

                # noinspection PyBroadException
                try:
                    if not mol or mol.GetNumAtoms() == 0:
                        raise ValueError("Invalid molecule object")

                    # 强制更新分子属性缓存
                    mol.UpdatePropertyCache(strict=False)
                    Chem.GetSSSR(mol)  # 初始化环信息

                    # 计算所有属性
                    for prop, func in property_calculators:
                        prop_dict[prop] = func(mol)

                    prop_dict["flag"] = True
                    stats["success"] += 1

                except Exception as e:
                    stats["failed"] += 1
                    prop_dict.update({prop: None for prop, _ in property_calculators})

                results.append(prop_dict)
                pbar.set_postfix_str(
                    f"Success: {stats['success']}, Failed: {stats['failed']}"
                )

        # 保存分块结果
        result_df = pd.DataFrame(results)
        write_header = chunk_idx == 0
        result_df.to_csv(output_csv, mode="a", header=write_header, index=False)

    # 生成统计报告
    total_time = time.time() - start_time
    stats["time"] = f"{total_time:.2f} seconds"
    stats["rate"] = (
        f"{stats['success'] / stats['total'] * 100:.1f}%" if stats["total"] else "N/A"
    )

    return stats


if __name__ == "__main__":
    stats = calculate_molecular_properties(
        input_csv="../original_dataset/chembl_34.csv",
        output_csv="property_chembl_34.csv",
        chunksize=50000,
    )

    print(
        f"Calculation Report:\n"
        f"Total Molecules: {stats['total']}\n"
        f"Success Rate: {stats['rate']}\n"
        f"Time Consumed: {stats['time']}"
    )
