#!/usr/bin/env python3
"""
使用 MOSES 官方工具评估生成分子数据集
nohup python -u ./evaluate/my_metrics_moses.py >./evaluate/my_metrics_moses.log 2>&1 &

conda create -n shw_moses
conda activate shw_moses
conda install python==3.7
conda install -yq -c rdkit rdkit
pip install molsets
conda install pandas
"""

import moses
import pandas as pd
import time

# --------------------------
# 输入文件路径
# --------------------------
train_file = "./data/guacamol.csv"
generated_file = "./generation_data/Guacamol/Guacamol.csv"

def main():
    start_time = time.time()
    try:
        print("正在计算分子指标...")
        data = pd.read_csv(generated_file)
        metrics = moses.get_all_metrics(list(data['smiles'].values), device = 'cuda')
        print(metrics)
        # 计算时间
        elapsed = time.time() - start_time
        print(f"总耗时: {elapsed:.2f} 秒")
        
    except Exception as e:
        print(f"评估失败: {str(e)}")

if __name__ == "__main__":
    main()