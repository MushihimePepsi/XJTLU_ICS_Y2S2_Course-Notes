import DataProcessing as DP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# KNN
from sklearn.neighbors import KNeighborsClassifier
# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
# 决策树
from sklearn.tree import DecisionTreeClassifier
# 集成学习
from sklearn.ensemble import VotingClassifier

# 传入一个data(Dataframe格式)，指定标签，划分为features和标签
def getFeaturesAndLabels(data, targetLabels):
    labels = DP.select_column(data,targetLabels).values
    labels = labels.ravel()
    features = DP.remove_column(data,targetLabels)
    return features, labels

# 数据预处理和划分训练集和测试集
def preprocessAndSplit(features, labels, test_size=0.2, random_state=0):
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # 划分训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(X_scaled, labels, test_size=test_size, random_state=random_state)
    return features_train, labels_train, features_test, labels_test, scaler

# 训练 kNN 分类器
def train_knn_classifier(features_train, labels_train,features_test,labels_test, n_neighbors):
    # 创建 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # 训练模型
    knn.fit(features_train, labels_train)
    # 计算训练集的准确率
    y_pred = knn.predict(features_test)
    accuracy = accuracy_score(labels_test, y_pred)
    print(f"KNN has achieved an accuracy of {accuracy:.2f} for the training dataset. n_neighbors={n_neighbors}")
    return knn, accuracy

# 训练朴素贝叶斯分类器
def train_naive_bayes_classifier(features_train, labels_train,features_test,labels_test, priors=None):
    # 创建朴素贝叶斯分类器
    nb = GaussianNB(priors=priors)
    # 训练模型
    nb.fit(features_train, labels_train)
    # 计算训练集的准确率
    y_pred = nb.predict(features_test)
    accuracy = accuracy_score(labels_test, y_pred)
    print(f"Naive Bayes Classifier has achieved an accuracy of {accuracy:.2f} for the training dataset.")
    return nb

# 训练决策树分类器
def train_decision_tree_classifier(features_train, labels_train,features_test,labels_test, dt_criterion,random_state=0):
    # 创建决策树分类器
    dt = DecisionTreeClassifier(criterion=dt_criterion,random_state=random_state)
    # 训练模型
    dt.fit(features_train, labels_train)
    # 计算训练集的准确率
    y_pred = dt.predict(features_test)
    accuracy = accuracy_score(labels_test, y_pred)
    print(f"Decision Tree has achieved an accuracy of {accuracy:.2f} for the training dataset. dt_criterion = {dt_criterion}")
    return dt

# 训练集成分类器
def train_ensemble_classifier(features_train, labels_train,features_test, labels_test, knn, nb, dt, ec_voting='hard'):
    # 创建集成分类器
    ensemble = VotingClassifier(estimators=[('knn', knn), ('nb', nb), ('dt', dt)], voting=ec_voting)
    # 训练集成模型
    ensemble.fit(features_train, labels_train)
    # 计算训练集的准确率
    y_pred = ensemble.predict(features_test)
    accuracy = accuracy_score(labels_test, y_pred)
    print(f"Ensemble Model has achieved an accuracy of {accuracy:.2f} for the training dataset. ec_voting={ec_voting}")
    return ensemble

#=============================================================================================================
#=============================================================================================================
#=============================================================================================================
#=============================================================================================================

# 不需要更改这个包
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# 删除某个列
def remove_column(df: pd.DataFrame, M: str) -> pd.DataFrame:
    """
    从DataFrame中移除指定列并返回结果

    参数:
    df (pd.DataFrame): 输入的DataFrame
    M (str): 需要移除的列名

    返回:
    pd.DataFrame: 不包含指定列的新DataFrame
    """
    # 检查列是否存在
    if M not in df.columns:
        print(f"警告: 列 '{M}' 不存在于DataFrame中，返回原DataFrame")
        return df

    # 返回删除指定列后的DataFrame
    return df.drop(columns=[M])

# 传入一个矩阵，返回 standardize 后的矩阵
def standardizeData(data):
    scaler = StandardScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    # 如果输入是 DataFrame，将标准化后的 ndarray 转换回 DataFrame
    if isinstance(data, pd.DataFrame):
        standardized_data = pd.DataFrame(standardized_data, columns=data.columns, index=data.index)
    return standardized_data

# 传入一个矩阵，返回 minMax 后的矩阵
def minMaxData(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    if isinstance(data, pd.DataFrame):
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    return scaled_data

# 传入一个矩阵，返回 normalize 后的矩阵
def normalizeData(data):
    scaler = Normalizer()
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    if isinstance(data, pd.DataFrame):
        normalized_data = pd.DataFrame(normalized_data, columns=data.columns, index=data.index)
    return normalized_data

# 传入一个矩阵，选择若干特征值，返回选择的矩阵
def select_column(raw_data, *col_indices):
    """
    该函数用于从输入的 DataFrame 中提取指定列，并将它们拼接成一个新的 DataFrame。
    可以传入列索引或者列名，根据类型选择使用 iloc 或 loc 提取列。
    :param raw_data: 输入的 pandas DataFrame
    :param col_indices: 要提取的列的索引或列名
    :return: 包含指定列的新 DataFrame
    """
    selected_columns = []
    for col in col_indices:
        if isinstance(col, str):
            # 如果是字符串，使用 loc 选择列
            selected_columns.append(raw_data.loc[:, col])
        else:
            # 否则使用 iloc 选择列
            selected_columns.append(raw_data.iloc[:, col])

    # 拼接选中的列
    return pd.concat(selected_columns, axis=1)

# 传入一个矩阵，选择[n,m]行，返回选择的矩阵
def slice_rows(raw_data, n, m):
    """
    该函数用于从输入的 DataFrame 中截取从第 n 行到第 m 行的数据。
    :param raw_data: 输入的 pandas DataFrame
    :param n: 起始行索引
    :param m: 结束行索引
    :return: 包含从第 n 行到第 m 行的新 DataFrame
    """
    return raw_data.iloc[n:m+1,:]

# 拼接 np.dataFrame
def concat_multiple_dfs(*dfs):
    """
    该函数用于拼接多个具有相同行数的 DataFrame。
    :param dfs: 若干个 pandas DataFrame
    :return: 拼接后的 DataFrame
    """
    return pd.concat(dfs, axis=1)

# 传入一个矩阵，返回压缩到n维的矩阵(PCA)
def pcaData(data, n):
    """
    该函数用于对输入的 DataFrame 进行 PCA 降维。
    :param data: 输入的 pandas DataFrame 特征矩阵
    :param n: 降维后的维度数
    :return: 降维后的 DataFrame
    """
    pca = PCA(n_components=n)
    pca_result = pca.fit_transform(data)
    columns = [f'PC{i+1}' for i in range(n)]
    return pd.DataFrame(pca_result, columns=columns, index=data.index)

# 绘制
def draw(data):
    num_columns = data.shape[1]

    if num_columns == 1:
        # 一维矩阵，绘制折线图
        plt.plot(data)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('1D Data Plot')
    elif num_columns == 2:
        # 二维矩阵，绘制散点图
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title('2D Data Plot')
    elif num_columns == 3:
        # 三维矩阵，绘制三维散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_zlabel(data.columns[2])
        plt.title('3D Data Plot')
    else:
        print("输入矩阵的维度应不大于三维。")
        return

    plt.show()

# 处理NaN数据为replaceTo
def fillNaNValueTo(X, replaceTo):
    return X.fillna(replaceTo)

# 含有NaN数据的row删掉
def deleteNaNValueRow(X):
    return X.dropna()