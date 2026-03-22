# coding: utf-8
import time
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from typing import Dict
from rdkit.Chem.Scaffolds import MurckoScaffold


def extract_molecular_scaffolds(
    input_csv: str, output_csv: str, chunksize: int = 10000
) -> Dict[str, str]:
    """
    批量提取分子 Murcko 骨架并保存结果
    Args:
        input_csv: 输入 CSV 文件路径
        output_csv: 输出 CSV 文件路径
        chunksize: 分块处理大小
    Returns:
        统计字典: 包含总耗时、成功率等统计指标
    """
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

        results = []
        with tqdm(
            df_chunk.iterrows(),
            total=len(df_chunk),
            desc=f"Processing chunk {chunk_idx + 1}",
            unit="mol",
        ) as pbar:

            for _, row in pbar:
                stats["total"] += 1
                scaf_dict = {"smiles": row["smiles"], "flag": False, "scaffold": None}

                # noinspection PyBroadException
                try:
                    mol = row["mol"]
                    if not mol or mol.GetNumAtoms() == 0:
                        raise ValueError("Invalid molecule object")

                    # 校验分子有效性
                    mol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(mol)  # 增加分子消毒
                    Chem.GetSSSR(mol)

                    # Murcko 骨架要求分子至少含有一个环, 需过滤无环结构
                    if not mol.GetRingInfo().NumRings():
                        raise ValueError("Molecule contains no rings")

                    # 强制更新分子属性缓存
                    mol.UpdatePropertyCache(strict=False)
                    Chem.GetSSSR(mol)  # 初始化环信息

                    # 生成 Murcko 骨架 (启用立体化学和规范 Kekule 形式)
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaf_dict["scaffold"] = Chem.MolToSmiles(
                        scaffold, isomericSmiles=True, kekuleSmiles=False
                    )
                    scaf_dict["flag"] = True
                    stats["success"] += 1

                except Exception as e:
                    stats["failed"] += 1

                results.append(scaf_dict)
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
    stats = extract_molecular_scaffolds(
        input_csv="../original_dataset/chembl_34.csv",
        output_csv="scaffold_chembl_34.csv",
        chunksize=50000,
    )

    print(
        f"Calculation Report:\n"
        f"Total Molecules: {stats['total']}\n"
        f"Success Rate: {stats['rate']}\n"
        f"Time Consumed: {stats['time']}"
    )
