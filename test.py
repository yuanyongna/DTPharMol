import csv

# 设置输入和输出文件路径
input_file = "RL_result/Moses_qedsa_ppgraph/prepare/docking_result_old.csv"
output_file = "RL_result/Moses_qedsa_ppgraph/prepare/docking_result.csv"


def safe_float(value, default=0.0):
    """安全地将字符串转换为浮点数"""
    try:
        # 检查字符串是否为空或仅包含空白字符
        if value.strip() == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        # 如果转换失败，返回默认值
        return default


# 读取所有数据
rows = []
with open(input_file, "r") as infile:
    reader = csv.reader(infile)

    # 读取标题行
    headers = next(reader)

    # 确定列索引位置
    score_idx = headers.index("score")
    mek_idx = headers.index("score_MEK1_7M0Y")
    mtor_idx = headers.index("score_mTOR_3FAP")

    # 处理数据行
    for row in reader:
        # 安全地转换MEK和mTOR分数
        mek_val = safe_float(row[mek_idx])
        mtor_val = safe_float(row[mtor_idx])

        # 计算新score值（保留3位小数）
        new_score = round(mek_val + mtor_val, 3)
        row[score_idx] = str(new_score)

        # 保存更新后的行
        rows.append(row)

# 根据score从大到小排序
rows.sort(key=lambda row: safe_float(row[score_idx]), reverse=False)

# 写入排序后的结果
with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)

    # 写入标题行
    writer.writerow(headers)

    # 写入排序后的数据行
    writer.writerows(rows)

# 验证排序结果（可选）
print(f"处理完成！结果已排序并保存至 {output_file}")
print(f"共处理 {len(rows)} 行数据")
print(f"最高分: {safe_float(rows[0][score_idx]) if rows else 'N/A'}")
print(f"最低分: {safe_float(rows[-1][score_idx]) if rows else 'N/A'}")
