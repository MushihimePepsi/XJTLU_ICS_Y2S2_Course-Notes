import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from scipy import stats
import Tools as F
data = pd.read_excel('training_set.xlsx')
# Apply Robust Standardize ======================================================
def robust(data):
    scaler = RobustScaler()
    data_robust = scaler.fit_transform(data)
    data_robust = pd.DataFrame(data_robust, columns=data.columns)
    return data_robust
data_robust = robust(data)

# Apply Log Standardize ======================================================
def log_transform_and_standardize(data, columns=None, add_one=True):
    """
    对DataFrame进行对数变换后再进行标准化（z-score）

    参数:
    data (pd.DataFrame): 输入数据
    columns (list, optional): 需要转换的列名列表，默认为None(处理所有数值列)
    add_one (bool): 是否在对数变换前加1，防止log(0)错误

    返回:
    pd.DataFrame: 转换后的DataFrame
    """
    # 如果未指定列，则选择所有数值列
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # 创建副本，避免修改原始数据
    data_transformed = data.copy()

    # 初始化标准化器
    scaler = StandardScaler()

    for col in columns:
        # 对数变换
        if add_one:
            data_transformed[col] = np.log1p(data[col])  # log(1+x)
        else:
            data_transformed[col] = np.log(data[col])

        # 应用标准化
        data_transformed[col] = scaler.fit_transform(data_transformed[[col]])

    return data_transformed
data_log_transformed = log_transform_and_standardize(data)

# Apply BoxCox Standardize ======================================================
def box_cox_transform_and_normalize(data):
    """
    对DataFrame进行Box - Cox变换后再进行标准化

    参数:
    data (pd.DataFrame): 输入数据

    返回:
    pd.DataFrame: 转换后的DataFrame
    """
    # 复制数据以避免修改原始数据
    processed_data = data.copy()

    # 确定需要转换的列（假设为所有数值列）
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # 存储变换后的数据
    transformed_array = np.zeros((len(data), len(numeric_cols)))

    # 对每个数值特征进行Box - Cox变换
    lambda_values = {}
    for i, col in enumerate(numeric_cols):
        try:
            # 添加小常数以确保所有值为正
            transformed, lambda_val = stats.boxcox(data[col] + 1e-10)
            transformed_array[:, i] = transformed
            lambda_values[col] = lambda_val
        except Exception as e:
            print(f"列 {col} 无法进行Box-Cox变换: {e}")
            # 如果变换失败，使用原始值
            transformed_array[:, i] = data[col].values
            lambda_values[col] = None

    # 标准化处理
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(transformed_array)

    # 只保留标准化后的列，删除中间步骤的列
    for i, col in enumerate(numeric_cols):
        processed_data[col] = normalized_data[:, i]
        # 移除Box-Cox变换后的中间列
        if f'{col}_box_cox' in processed_data.columns:
            processed_data.drop(f'{col}_box_cox', axis=1, inplace=True)

    return processed_data

data_BoxCox = box_cox_transform_and_normalize(data)


def calculate_mean(data):
    """
    计算数据集中每个特征的均值

    参数:
    data (array-like): 输入数据，二维数组，形状为 [样本数, 特征数]

    返回:
    np.ndarray: 每个特征的均值，一维数组
    """
    # 将输入转换为 numpy 数组
    data_array = np.asarray(data)

    # 计算每个特征的均值（沿 axis=0 计算）
    return np.mean(data_array, axis=0)


def calculate_std(data):
    """
    计算数据集中每个特征的标准差

    参数:
    data (array-like): 输入数据，二维数组，形状为 [样本数, 特征数]

    返回:
    np.ndarray: 每个特征的标准差，一维数组
    """
    # 将输入转换为 numpy 数组
    data_array = np.asarray(data)

    # 计算每个特征的标准差（沿 axis=0 计算）
    return np.std(data_array, axis=0)

print(f"raw = {calculate_mean(data).mean():.3f}")
print(f"robust = {calculate_mean(data_robust).mean():.3f}")
print(f"log = {calculate_mean(data_log_transformed).mean():.3f}")
print(f"boxCox = {calculate_mean(data_BoxCox).mean():.3f}")
print("===================================================")
print(f"raw = {calculate_std(data).mean():.3f}")
print(f"robust = {calculate_std(data_robust).mean():.3f}")
print(f"log = {calculate_std(data_log_transformed).mean():.3f}")
print(f"boxCox = {calculate_std(data_BoxCox).mean():.3f}")

data_pca = F.pcaData(data,2)
F.draw(data_pca)
