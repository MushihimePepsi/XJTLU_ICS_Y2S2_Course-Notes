# 不需要更改这个包

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
import pandas as pd


# 评估聚类结果的函数
def evaluate_clustering(X, labels):
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


# K-Means 聚类，其中 n_attempt 参数用于指明训练多少次，尽可能找到最好质心
def kmeans_clustering(feature_matrix, n_attempt='auto', n_clusters=4):
    """
    使用 K-Means 对输入特征矩阵进行聚类

    Parameters:
    feature_matrix (numpy.ndarray): 特征矩阵，行表示样本，列表示特征
    n_attempt (int): 初始值尝试次数，'auto' 表示使用默认值
    n_clusters (int): 聚类数量，默认为 4

    Returns:
    tuple: (评估比率, 聚类标签)
    """
    # 设置初始值尝试次数
    if n_attempt == 'auto':
        n_init = 10
    else:
        n_init = n_attempt

    # 初始化 K-Means 模型
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)

    # 拟合模型并获取聚类标签
    labels = kmeans.fit_predict(feature_matrix)

    # 计算评估指标
    ratio = evaluate_clustering(feature_matrix, labels)

    return ratio, labels

# GMM 聚类，其中 n_attempt 参数用于指明训练多少次，尽可能找到最好的局部最优解
def gmm_clustering(feature_matrix, n_attempt='auto', n_components=4):
    """
    使用高斯混合模型 (GMM) 对输入特征矩阵进行聚类

    Parameters:
    feature_matrix (numpy.ndarray): 特征矩阵，行表示样本，列表示特征
    n_attempt (int): 初始值尝试次数，'auto' 表示使用默认值
    n_components (int): 高斯分量数量，默认为 4

    Returns:
    tuple: (评估比率, 聚类标签)
    """
    # 设置初始值尝试次数
    if n_attempt == 'auto':
        n_init = 1
    else:
        n_init = n_attempt

    # 初始化 GMM 模型
    gmm = GaussianMixture(n_components=n_components, n_init=n_init, random_state=42)

    # 拟合模型并获取聚类标签
    gmm.fit(feature_matrix)
    labels = gmm.predict(feature_matrix)

    # 计算评估指标
    ratio = evaluate_clustering(feature_matrix, labels)

    return ratio, labels


# 层次聚类
def hierarchical_clustering(feature_matrix, method='ward', n_clusters=4):
    """
    使用层次聚类对输入特征矩阵进行聚类

    Parameters:
    feature_matrix (numpy.ndarray): 特征矩阵，行表示样本，列表示特征
    method (str): 距离计算方法，默认为 'ward'
    n_clusters (int): 聚类数量，默认为 4

    Returns:
    tuple: (评估比率, 聚类标签)
    """
    # 执行层次聚类
    Z = linkage(feature_matrix, method=method)

    # 分配聚类标签
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # 计算评估指标
    ratio = evaluate_clustering(feature_matrix, labels)

    return ratio, labels
