import pandas as pd
from scipy.sparse import csr_matrix

df = pd.read_csv('error_10w_008002.csv')

#B = pd.read_csv('error_10w_008002.csv')

# 检查NA值
na_count = df["beer/name"].isna().sum()
print(f"NA值的数量: {na_count}")

# 检查csr_matrix类型
csr_count = df["beer/name"].apply(lambda x: isinstance(x, csr_matrix)).sum()
print(f"csr_matrix类型值的数量: {csr_count}")

# 查看具体包含csr_matrix的行
if csr_count > 0:
    csr_rows = df[df["beer/name"].apply(lambda x: isinstance(x, csr_matrix))]
    print("\n包含csr_matrix的行:")
    print(csr_rows)