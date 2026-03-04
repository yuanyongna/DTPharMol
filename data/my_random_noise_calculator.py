import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs
from multiprocessing import Pool, cpu_count

# 禁用 RDKit 日志
RDLogger.DisableLog("rdApp.*")


class Structure_SVN_Encoder(nn.Module):
    """优化版结构变分编码器（支持批量处理）"""

    def __init__(self, input_dim=2048, latent_dim=128):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x_batch):
        x_batch = self.compressor(x_batch)
        mean = self.fc_mean(x_batch)
        log_var = self.fc_logvar(x_batch)
        return mean, log_var


def process_smiles(smiles):
    """并行处理 SMILES 生成分子和指纹"""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if not mol:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        return fp_array
    except:
        return None


def compute_batch_noise(fp_batch, latent_dim=128, batch_size=4096):
    """批量计算噪声（核心优化）"""
    encoder = Structure_SVN_Encoder(latent_dim=latent_dim)
    encoder.eval()
    noise_results = []

    # 分批处理避免内存溢出
    for i in range(0, len(fp_batch), batch_size):
        batch = fp_batch[i : i + batch_size]
        fp_tensor = torch.tensor(np.array(batch)).float()

        with torch.no_grad():
            mean, log_var = encoder(fp_tensor)
            std_dev = torch.exp(0.5 * log_var)
            z = torch.rand_like(mean) * 2 - 1  # U(-1,1)
            batch_noise = mean + std_dev * z

            if latent_dim < 32:
                high_freq = torch.randn_like(batch_noise) * 0.1
                batch_noise += high_freq

            noise_results.extend(batch_noise.numpy())

    return noise_results


def random_noise_calculator(
    input_csv,
    output_csv,
    chunksize=10000,
    latent_dim=128,
    max_workers=None,
    batch_size=4096,
):

    # 初始化统计指标
    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    start_time = time.time()

    # 自动设置并行进程数
    num_workers = max_workers or max(1, cpu_count() // 2)

    for chunk_idx, df_chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize)):
        chunk_smiles = df_chunk["smiles"].tolist()
        valid_indices = []
        valid_smiles = []
        # 第一阶段并行：分子生成和指纹计算
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_smiles, chunk_smiles),
                    total=len(chunk_smiles),
                    desc=f"Chunk {chunk_idx+1}: Fingerprints",
                    unit="mol",
                )
            )
        # 筛选有效结果
        fp_batch = []
        for i, res in enumerate(results):
            stats["total"] += 1
            if res is None:
                stats["failed"] += 1
            else:
                valid_indices.append(i)
                valid_smiles.append(chunk_smiles[i])
                fp_batch.append(res)
                stats["success"] += 1

        # 第二阶段：批量生成噪声
        if fp_batch:
            noise_vectors = compute_batch_noise(fp_batch, latent_dim, batch_size)
        else:
            noise_vectors = []

        # 组装结果
        results = []
        noise_iter = iter(noise_vectors)
        for i in range(len(chunk_smiles)):
            if i in valid_indices:
                noise = next(noise_iter)
                results.append(
                    {
                        "smiles": chunk_smiles[i],
                        "flag": True,
                        "random_noise": noise.tolist(),
                    }
                )
            else:
                results.append(
                    {"smiles": chunk_smiles[i], "flag": False, "random_noise": None}
                )
                stats["skipped"] += 1

        # 保存分块结果
        result_df = pd.DataFrame(results)
        write_header = not os.path.exists(output_csv)
        result_df.to_csv(output_csv, mode="a", header=write_header, index=False)

    # 性能报告
    total_time = time.time() - start_time
    mols_per_sec = stats["success"] / total_time if total_time > 0 else 0

    return {
        "total_molecules": stats["total"],
        "succeeded": stats["success"],
        "failed": stats["failed"],
        "skipped": stats["skipped"],
        "time_sec": f"{total_time:.2f}",
        "speed": f"{mols_per_sec:.2f} mols/sec",
    }


if __name__ == "__main__":
    result = random_noise_calculator(
        input_csv="dataset/original_dataset/chembl_34.csv",
        output_csv="dataset/features_calculation/random_noise_chembl_34.csv",
        chunksize=10000,
        latent_dim=128,
        max_workers=8,  # 根据 CPU 核心数调整
        batch_size=1024,  # 根据 GPU 内存调整
    )

    for k, v in result.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
