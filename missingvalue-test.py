import pandas as pd
import numpy as np
from missingvalue import MissingValueChecker

def test_missing_value_checker():
    # 创建测试数据
    df = pd.DataFrame({
        "numerical_col": [1, 2, np.nan, 4],
        "categorical_col": ["cat", "dog", None, "cat"],
    })

    # 测试数值型数据：均值填充
    checker = MissingValueChecker(data_type="numerical", strategy="mean")
    result = checker.transform(df["numerical_col"])
    print("Numerical with mean:")
    print(result)  # 应该返回 [1.0, 2.0, 2.333..., 4.0]

    # 测试数值型数据：最常见值填充
    checker = MissingValueChecker(data_type="numerical", strategy="most_common")
    result = checker.transform(df["numerical_col"])
    print("Numerical with most common:")
    print(result)  # 应该返回 [1.0, 2.0, 1.0, 4.0]

    # 测试类别型数据：最常见值填充
    checker = MissingValueChecker(data_type="categorical", strategy="most_common")
    result = checker.transform(df["categorical_col"])
    print("Categorical with most common:")
    print(result)  # 应该返回 ["cat", "dog", "cat", "cat"]

    # 测试自定义函数
    checker = MissingValueChecker(data_type="numerical", strategy="custom", custom_func=lambda x: x.min() - 1)
    result = checker.transform(df["numerical_col"])
    print("Numerical with custom function:")
    print(result)  # 应该返回 [1.0, 2.0, 0.0, 4.0]

    # 测试完整 DataFrame
    checker = MissingValueChecker(data_type="numerical", strategy="mean")
    df_transformed = checker.transform(df)
    print("Transformed DataFrame:")
    print(df_transformed)

if __name__ == "__main__":
    test_missing_value_checker()
