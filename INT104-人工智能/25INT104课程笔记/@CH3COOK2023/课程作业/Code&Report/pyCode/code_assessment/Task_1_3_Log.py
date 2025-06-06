import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler

# 设置字体
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False


def log_transform_and_standardize(data, columns=None, add_one=True):
    """
    对DataFrame进行对数变换后再进行标准化（z-score）

    参数:
    data (pd.DataFrame): 输入数据
    columns (list, optional): 需要转换的列名列表，默认为None(处理所有数值列)
    add_one (bool): 是否在对数变换前加1，防止log(0)错误

    返回:
    pd.DataFrame: 转换后的DataFrame
    fig: 包含所有特征原始和转换后分布对比的图形对象
    """
    # 如果未指定列，则选择所有数值列
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # 创建副本，避免修改原始数据
    data_transformed = data.copy()

    # 初始化标准化器
    scaler = StandardScaler()

    num_cols = len(columns)
    fig, axes = plt.subplots(2, num_cols, figsize=(15, 6))
    for i, col in enumerate(columns):
        # 对数变换
        if add_one:
            data_transformed[col] = np.log1p(data[col])  # log(1+x)
        else:
            data_transformed[col] = np.log(data[col])

        # 应用标准化
        data_transformed[col] = scaler.fit_transform(data_transformed[[col]])

        # 原始数据分布
        sns.histplot(data[col], kde=True, ax=axes[0, i])
        axes[0, i].set_title(f'{col} Original Distribution')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')

        # 转换后数据分布
        sns.histplot(data_transformed[col], kde=True, ax=axes[1, i])
        axes[1, i].set_title(f'{col} After Log + Standardization')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')

    plt.tight_layout()
    return data_transformed, fig


# 示例使用
if __name__ == "__main__":
    # 生成示例数据（可替换为你的实际数据）
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    data = pd.read_excel('training_set.xlsx')

    # 指定需要转换的列（仅数值列）
    columns_to_transform = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

    # 执行对数变换和标准化
    transformed_data, fig = log_transform_and_standardize(
        data,
        columns=columns_to_transform,
        add_one=True  # 对于可能包含0值的数据，建议设为True
    )

    # 保存转换后的数据
    transformed_data.to_csv('transformed_data.csv', index=False)

    # 保存图形
    fig.savefig('Task_1_3_Log.pdf', format='pdf', dpi=300)

    # 显示图形
    plt.show()