from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
import sklearn
import Tools as T

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_optimal_eps(X, min_samples=5, k=None, eps_range=None, num_steps=100):
    """
    寻找DBSCAN的最佳eps参数

    参数:
    X: 输入数据
    min_samples: DBSCAN的min_samples参数
    k: k近邻的k值，默认为min_samples
    eps_range: 搜索eps的范围，默认为自动计算
    num_steps: 搜索步数

    返回:
    最佳eps值、聚类数量、DBI分数和所有尝试结果
    """
    # 设置k值
    if k is None:
        k = min_samples

    # 计算k近邻距离
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    # 排序距离
    distances = np.sort(distances, axis=0)
    distances = distances[:, -1]  # 使用第k个近邻距离而非第二个

    # 如果未指定eps范围，自动计算
    if eps_range is None:
        min_eps = np.percentile(distances, 5)  # 降低下界以增加搜索范围
        max_eps = np.percentile(distances, 95)  # 提高上界以增加搜索范围
        eps_range = (max(min_eps, 1e-6), max_eps)  # 确保最小值不为0

    # 生成候选eps值
    eps_values = np.linspace(eps_range[0], eps_range[1], num_steps)

    best_eps = None
    best_score = float('inf')  # DBI越小越好，初始化为无穷大
    best_n_clusters = 0

    results = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # 计算聚类数量（排除噪声点）
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)

        # 仅当聚类数量大于1时计算DBI
        if n_clusters > 1:
            try:
                # 不使用core_samples_mask，直接使用所有非噪声点
                valid_points = X[labels != -1]
                valid_labels = labels[labels != -1]

                # 确保有足够的聚类点来计算DBI
                if len(set(valid_labels)) > 1:
                    dbi_score = davies_bouldin_score(valid_points, valid_labels)
                    results.append((eps, n_clusters, dbi_score, noise_ratio))

                    # 更新最佳参数（DBI越小越好）
                    if dbi_score < best_score:
                        best_score = dbi_score
                        best_eps = eps
                        best_n_clusters = n_clusters
                else:
                    results.append((eps, n_clusters, None, noise_ratio))
            except ValueError as e:
                # 处理DBI计算错误
                results.append((eps, n_clusters, None, noise_ratio))
        else:
            results.append((eps, n_clusters, None, noise_ratio))

    # 如果没有找到有效聚类，返回具有最多聚类数的eps
    if best_eps is None:
        # 寻找聚类数最多的eps
        max_clusters = max(r[1] for r in results)
        candidates = [r for r in results if r[1] == max_clusters]
        # 从候选中选择噪声比最小的
        best_candidate = min(candidates, key=lambda x: x[3])
        best_eps = best_candidate[0]
        best_n_clusters = max_clusters
        best_score = None

    return best_eps, best_n_clusters, best_score, results

def plot_clusters(data, labels, figsize=(6, 4),name = '.pdf',title = ''):
    """
    Visualize clustering results based on labels.

    Parameters:
    data (array-like): Input data, can be 2D or 3D.
    labels (array-like): Clustering labels.
    figsize (tuple): Figure size.
    """
    # Convert to numpy array
    data = np.array(data)

    # Check data dimension
    if data.ndim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D.")

    # Check number of columns for data dimension
    if data.ndim == 2 and data.shape[1] != 2:
        raise ValueError("2D data must have two columns.")
    if data.ndim == 3 and data.shape[1] != 3:
        raise ValueError("3D data must have three columns.")

    # Get unique labels
    unique_labels = np.unique(labels)

    # Create figure
    plt.figure(figsize=figsize)

    # 2D data visualization
    if data.ndim == 2:
        for label in unique_labels:
            cluster_points = data[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        label=f'Cluster {label}', alpha=0.7)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    # 3D data visualization
    else:
        ax = plt.axes(projection='3d')
        for label in unique_labels:
            cluster_points = data[labels == label]
            ax.scatter3D(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                         label=f'Cluster {label}', alpha=0.7)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.show()


def plot_multi_dataframes_2D(names, *dataframes, figsize=(12, 8)):
    """
    Plot multiple 2D DataFrames in separate subplots

    Parameters:
    names (list): List of four subplot titles
    *dataframes: Four DataFrame objects
    figsize (tuple): Figure size, default (12, 8)
    """
    # Ensure four DataFrames and four names are provided
    if len(dataframes) != 4:
        raise ValueError("Exactly four DataFrames must be provided")
    if len(names) != 4:
        raise ValueError("Names list must contain four titles")

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array to 1D

    # Iterate over each DataFrame and plot subplot
    for i, (df, name) in enumerate(zip(dataframes, names)):
        # Check if DataFrame has at least two columns
        if df.shape[1] < 2:
            raise ValueError(f"DataFrame {i + 1} must have at least two columns")

        # Get first two columns as x and y
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Plot scatter plot
        axes[i].scatter(x, y, alpha=0.7)

        # Set subplot title
        axes[i].set_title(name)

        # Hide axis labels
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Automatically adjust layout
    plt.tight_layout()

    plt.show()
def plot_multi_dataframes_3D(names, *dataframes, figsize=(12, 8)):
    """
    Plot multiple 3D DataFrames in separate subplots

    Parameters:
    names (list): List of four subplot titles
    *dataframes: Four DataFrame objects
    figsize (tuple): Figure size, default (12, 8)
    """
    # Ensure four DataFrames and four names are provided
    if len(dataframes) != 4:
        raise ValueError("Exactly four DataFrames must be provided")
    if len(names) != 4:
        raise ValueError("Names list must contain four titles")

    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=figsize)

    # Iterate over each DataFrame and plot subplot
    for i, (df, name) in enumerate(zip(dataframes, names)):
        # Check if DataFrame has at least three columns
        if df.shape[1] < 3:
            raise ValueError(f"DataFrame {i + 1} must have at least three columns")

        # Get first three columns as x, y, and z
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df.iloc[:, 2]

        # Add 3D subplot
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Plot 3D scatter plot
        ax.scatter(x, y, z, alpha=0.7)

        # Set subplot title
        ax.set_title(name)

        # Hide axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    # Automatically adjust layout
    plt.tight_layout()

    plt.show()
def plot_multi_dataframes_2D_3D(names, *dataframes, figsize=(12, 12)):
    """
    Plot multiple 2D and 3D DataFrames in a single figure

    Parameters:
    names (list): List of subplot titles (8 titles required)
    *dataframes: Eight DataFrame objects (first four 2D, next four 3D)
    figsize (tuple): Figure size, default (16, 12)
    """
    # Ensure eight DataFrames and eight names are provided
    if len(dataframes) != 8:
        raise ValueError("Exactly eight DataFrames must be provided")
    if len(names) != 8:
        raise ValueError("Names list must contain eight titles")

    # Split DataFrames into 2D (first four) and 3D (next four)
    dfs_2d = dataframes[:4]
    dfs_3d = dataframes[4:]

    # Create a 4x2 subplot layout (4 rows, 2 columns)
    fig = plt.figure(figsize=figsize)

    # Plot 2D DataFrames in the first column
    for i, (df, name) in enumerate(zip(dfs_2d, names[:4])):
        # Check if DataFrame has at least two columns
        if df.shape[1] < 2:
            raise ValueError(f"2D DataFrame {i + 1} must have at least two columns")

        # Get first two columns as x and y
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Add 2D subplot (left column)
        ax = fig.add_subplot(4, 2, 2 * i + 1)

        # Plot scatter plot
        ax.scatter(x, y, alpha=0.7)

        # Set subplot title
        ax.set_title(name)

        # Hide axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Plot 3D DataFrames in the second column
    for i, (df, name) in enumerate(zip(dfs_3d, names[4:])):
        # Check if DataFrame has at least three columns
        if df.shape[1] < 3:
            raise ValueError(f"3D DataFrame {i + 5} must have at least three columns")

        # Get first three columns as x, y, and z
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df.iloc[:, 2]

        # Add 3D subplot (right column)
        ax = fig.add_subplot(4, 2, 2 * i + 2, projection='3d')

        # Plot 3D scatter plot
        ax.scatter(x, y, z, alpha=0.7)

        # Set subplot title
        ax.set_title(name)

        # Hide axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    # Automatically adjust layout
    plt.tight_layout()

    plt.savefig('output.pdf')
    plt.show()


def calculate_ari(true_labels, predicted_labels):
    """
    计算并返回聚类结果的调整兰德指数(ARI)

    参数:
    true_labels (array-like): 真实标签数组
    predicted_labels (array-like): 预测标签数组

    返回:
    float: ARI值，范围从-1到1，值越大表示聚类效果越好
    """
    # 检查输入标签长度是否一致
    if len(true_labels) != len(predicted_labels):
        raise ValueError("真实标签和预测标签的长度必须相同")

    # 计算ARI值
    ari = adjusted_rand_score(true_labels, predicted_labels)

    return ari
def calculate_dbi(X, labels):
    """
        通过计算簇内距离与簇间距离的比值来评估聚类结果。
    参数:
    X : 类数组，形状为 (n_samples, n_features)
        输入数据。
    labels : 类数组，形状为 (n_samples,)
        每个样本的聚类标签。

    返回值:
    float
        簇内距离与簇间距离的比值。
    """
    unique_labels = np.unique(labels)

    # 计算簇内距离
    intra_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_distance = np.mean(pairwise_distances(cluster_points))
            intra_distances.append(intra_distance)

    # 计算簇间距离
    inter_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            inter_distance = np.mean(pairwise_distances(cluster_i, cluster_j))
            inter_distances.append(inter_distance)

    # 计算平均簇内距离和平均簇间距离
    avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
    avg_inter_distance = np.mean(inter_distances) if inter_distances else 1  # 防止 / 0

    # 计算 ratio
    ratio = avg_intra_distance / avg_inter_distance if avg_inter_distance != 0 else float('inf')

    return ratio

def robust(data):
    scaler = RobustScaler()
    data_robust = scaler.fit_transform(data)
    data_robust = pd.DataFrame(data_robust, columns=data.columns)
    return data_robust
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


data = pd.read_excel('training_set.xlsx')

featI = T.concat_multiple_dfs(
    T.select_column(data,'Grade'),
    T.select_column(data,'Q1'),
    T.select_column(data,'Q2'),
)
featII = T.concat_multiple_dfs(
    T.select_column(data,'Grade'),
    T.select_column(data,'Q3'),
    T.select_column(data,'Q4'),
)
featIII = T.concat_multiple_dfs(
    T.select_column(data,'Grade'),
    T.select_column(data,'Q4'),
    T.select_column(data,'Q5'),
)
raw_featI = featI
raw_featII = featII
raw_featIII = featIII

robust_featI = robust(featI)
robust_featII = robust(featII)
robust_featIII = robust(featIII)

log_featI = log_transform_and_standardize(featI)
log_featII = log_transform_and_standardize(featII)
log_featIII = log_transform_and_standardize(featIII)

box_featI = box_cox_transform_and_normalize(featI)
box_featII = box_cox_transform_and_normalize(featII)
box_featIII = box_cox_transform_and_normalize(featIII)

raw_combination = T.concat_multiple_dfs(raw_featI, raw_featII, raw_featIII)
robust_combination = T.concat_multiple_dfs(robust_featI,robust_featII,robust_featIII)
log_combination = T.concat_multiple_dfs(log_featI,log_featII,log_featIII)
box_combination = T.concat_multiple_dfs(box_featI,box_featII,box_featIII)

# =======================
raw_pca2 = T.pcaData(raw_combination,2)
robust_pca2 = T.pcaData(robust_combination,2)
log_pca2 = T.pcaData(log_combination,2)
box_pca2 = T.pcaData(box_combination,2)
#  plot_multi_dataframes_2D(['a','b','c','d'],raw_pca2,robust_pca2,log_pca2,box_pca2)


raw_pca3 = T.pcaData(raw_combination,3)
robust_pca3 = T.pcaData(robust_combination,3)
log_pca3 = T.pcaData(log_combination,3)
box_pca3 = T.pcaData(box_combination,3)
#  plot_multi_dataframes_3D(['a','b','c','d'],raw_pca3,robust_pca3,log_pca3,box_pca3)

# nameArr = [
#             'Original Data PCA 2 Dim.',
#             'Robust Standardize Data PCA 2 Dim.',
#             'Logarithmic Standardize Data PCA 2 Dim.',
#             'Box-Cox Standardize Data PCA 2 Dim.',
#
#             'Original Data PCA 3 Dim.',
#             'Robust Standardize Data PCA 3 Dim.',
#             'Logarithmic Standardize Data PCA 3 Dim.',
#             'Box-Cox Standardize Data PCA 3 Dim.'
#            ]
# plot_multi_dataframes_2D_3D(nameArr,raw_pca2,robust_pca2,log_pca2,box_pca2,raw_pca3,robust_pca3,log_pca3,box_pca3)
#
# kmeans = KMeans(n_clusters=4, random_state=0)
# labels = kmeans.fit_predict(raw_pca2)
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(raw_pca2,labels))
# plot_clusters(raw_pca2, labels,name = 'KMeans_image1.pdf',title='Original Data KMeans Results')
#
#
# kmeans = KMeans(n_clusters=4, random_state=0)
# labels = kmeans.fit_predict(robust_pca2)
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(robust_pca2,labels))
# plot_clusters(robust_pca2, labels,name = 'KMeans_image2.pdf',title='Robust Standardize Data KMeans Results')
#
# kmeans = KMeans(n_clusters=4, random_state=0)
# labels = kmeans.fit_predict(log_pca2)
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(log_pca2,labels))
# plot_clusters(log_pca2, labels,name = 'KMeans_image3.pdf',title='Logarithmic Standardize Data KMeans Results')
#
# kmeans = KMeans(n_clusters=4, random_state=0)
# labels = kmeans.fit_predict(box_pca2)
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(box_pca2,labels))
# plot_clusters(box_pca2, labels,name = 'KMeans_image4.pdf',title='Box Cox Standardize Data KMeans Results')

# dbscan = DBSCAN(eps=1.65, min_samples=1)
# labels = dbscan.fit_predict(raw_pca2)
# plot_clusters(raw_pca2, labels,name = 'DBSCAN_image1.pdf',title='Original Data DBSCAN')
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels))
# print(calculate_dbi(raw_pca2, labels))
#
#
#
#
# dbscan = DBSCAN(eps=0.70, min_samples=1)
# labels = dbscan.fit_predict(robust_pca2)
# plot_clusters(robust_pca2, labels,name = 'DBSCAN_image2.pdf',title='Robust Standardize DBSCAN')
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels))
# print(calculate_dbi(robust_pca2, labels))
#
#
# dbscan = DBSCAN(eps=0.8, min_samples=1)
# labels = dbscan.fit_predict(log_pca2)
# plot_clusters(log_pca2, labels,name = 'DBSCAN_image3.pdf',title='Logarithmic Standardize DBSCAN')
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels))
# print(calculate_dbi(log_pca2, labels))
#
# dbscan = DBSCAN(eps=0.6, min_samples=1)
# labels = dbscan.fit_predict(box_pca2)
# plot_clusters(box_pca2, labels,name = 'DBSCAN_image4.pdf',title='Box-Cox Standardize DBSCAN')
# print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels))
# print(calculate_dbi(box_pca2, labels))

gmm = GaussianMixture(n_components=4, covariance_type='full')
labels = gmm.fit_predict(raw_pca2)
plot_clusters(raw_pca2,labels,name='GMM_image1.pdf',title='Original Data GMM')
print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(raw_pca2,labels))

gmm = GaussianMixture(n_components=4, covariance_type='full')
labels = gmm.fit_predict(robust_pca2)
plot_clusters(robust_pca2,labels,name='GMM_image2.pdf',title='Robust Standardize GMM')
print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(robust_pca2,labels))

gmm = GaussianMixture(n_components=4, covariance_type='full')
labels = gmm.fit_predict(log_pca2)
plot_clusters(log_pca2,labels,name='GMM_image3.pdf',title='Logarithmic Standardize GMM')
print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(log_pca2,labels))


gmm = GaussianMixture(n_components=4, covariance_type='full')
labels = gmm.fit_predict(box_pca2)
plot_clusters(box_pca2,labels,name='GMM_image4.pdf',title='Box-Cox Standardize GMM')
print(calculate_ari(T.select_column(data,'Programme').values.flatten(), labels),calculate_dbi(box_pca2,labels))

