import pandas as pd

df = pd.read_csv("./RL_result/Moses_qedsa_ppgraph_MEK1_mTOR/prepare/docking_result.csv")
avg = df["score_mTOR_3FAP"].mean()

print(f"Average score_CDK2: {avg:.4f}")
