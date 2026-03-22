import pandas as pd


def save_first_n_rows(input_csv, output_csv, n):
    """
    读取指定 CSV 文件的前 n 行，并保存至指定的新文件。

    参数:
    input_csv (str): 输入 CSV 文件的路径。
    output_csv (str): 输出 CSV 文件的路径。
    n (int): 要读取的行数。
    """
    # 读取指定数量的行
    try:
        # 读取整个 CSV 文件
        data = pd.read_csv(input_csv)
        # 计算需要选取的行数
        train_count = int(n * 0.9)
        val_count = n - train_count
        # 根据条件过滤数据
        train_data = data[data["source"] == "train"].sample(
            n=train_count, random_state=1
        )
        cal_data = data[data["source"] == "val"].sample(n=val_count, random_state=1)
        # 合并选取的数据
        sampled_data = pd.concat([train_data, cal_data])
        # 保存到新文件
        sampled_data.to_csv(output_csv, index=False)
        print(f"成功保存样本到 {output_csv}")
    except Exception as e:
        print(f"发生错误: {e}")


input_file = "data/guacamol.csv"
output_file = "data/guacamol_small.csv"
num_rows = 5000

save_first_n_rows(input_file, output_file, num_rows)
