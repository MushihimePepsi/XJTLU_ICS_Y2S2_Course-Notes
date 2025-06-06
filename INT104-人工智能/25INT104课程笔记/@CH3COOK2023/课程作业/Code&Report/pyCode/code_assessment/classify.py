# 传入一个矩阵，选择若干特征值，返回选择的矩阵
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import stats
import Tools as T
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize

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
# 传入一个data(Dataframe格式)，指定标签，划分为features和标签
def getFeaturesAndLabels(data, targetLabels):
    labels = select_column(data,targetLabels).values
    labels = labels.ravel()
    features = remove_column(data,targetLabels)
    return features, labels
# 数据预处理和划分训练集和测试集
def preprocessAndSplit(features, labels, test_size=0.2, random_state=0):
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # 划分训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(X_scaled, labels, test_size=test_size, random_state=random_state)
    return features_train, labels_train, features_test, labels_test, scaler
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
def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title='Confusion Matrix', cmap='Blues',
                          figsize=(8, 6),fileName = '.pdf'):
    """
    Visualize a confusion matrix using matplotlib

    Parameters:
    y_true (array-like): Ground truth (correct) target values
    y_pred (array-like): Estimated targets as returned by a classifier
    class_names (list): List of class names (in label order)
    normalize (bool): Whether to show normalized percentages (default False shows absolute counts)
    title (str): Plot title (default 'Confusion Matrix')
    cmap (str): Colormap scheme (default 'Blues')
    figsize (tuple): Figure size in inches (default (8,6))
    """
    # Basic validation
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true labels and predicted labels do not match")
    if len(class_names) != len(np.unique(y_true)):
        raise ValueError("Number of class names does not match the number of unique classes in true labels")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # Normalization handling
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # Show percentages with 2 decimal places
    else:
        fmt = 'd'  # Show integer counts

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Add color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Sample Proportion' if normalize else 'Sample Count', rotation=-90, va='bottom')

    # Configure axes
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Class',
           xlabel='Predicted Class')

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate matrix values
    threshold = im.norm(cm.max()) / 2.0  # Threshold for text color contrast
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if im.norm(cm[i, j]) > threshold else "black"
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", color=color,
                    fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(fileName)
    fig.show()
# 首先，划分feature和labels
data = pd.read_excel('training_set.xlsx')
feat1 = T.select_column(data,'Q1','Q2','Q3','Q4','Q5')
feat2 = T.select_column(data,'Grade','Gender')
feat3 = T.select_column(data,'Q1','Q3')
feat4 = T.select_column(data,'Q2','Q4')
feat5 = T.select_column(data,'Q3','Q5')
feat6 = T.select_column(data,'Q1','Q2','Q3')
feat7 = T.select_column(data,'Q2','Q3','Q4')
feat8 = T.select_column(data,'Q3','Q4','Q5')
feat9 = T.select_column(data,'Q4','Q5')
feat10 = T.select_column(data,'Q1','Q5')
feat11 = T.select_column(data,'Q1','Q2')

# ==============[Apply 3 Trans.]===================
# feat1     = robust(feat1)
# feat2     = robust(feat2)
# feat3     = robust(feat3)
# feat4     = robust(feat4)
# feat5     = robust(feat5)
# feat6     = robust(feat6)
# feat7     = robust(feat7)
# feat8     = robust(feat8)
# feat9     = robust(feat9)
# feat10    = robust(feat10)
# feat11    = robust(feat11)

# feat1   = log_transform_and_standardize(feat1)
# feat2   = log_transform_and_standardize(feat2)
# feat3   = log_transform_and_standardize(feat3)
# feat4   = log_transform_and_standardize(feat4)
# feat5   = log_transform_and_standardize(feat5)
# feat6   = log_transform_and_standardize(feat6)
# feat7   = log_transform_and_standardize(feat7)
# feat8   = log_transform_and_standardize(feat8)
# feat9   = log_transform_and_standardize(feat9)
# feat10  = log_transform_and_standardize(feat10)
# feat11  = log_transform_and_standardize(feat11)

feat1   = box_cox_transform_and_normalize(feat1)
feat2   = box_cox_transform_and_normalize(feat2)
feat3   = box_cox_transform_and_normalize(feat3)
feat4   = box_cox_transform_and_normalize(feat4)
feat5   = box_cox_transform_and_normalize(feat5)
feat6   = box_cox_transform_and_normalize(feat6)
feat7   = box_cox_transform_and_normalize(feat7)
feat8   = box_cox_transform_and_normalize(feat8)
feat9   = box_cox_transform_and_normalize(feat9)
feat10  = box_cox_transform_and_normalize(feat10)
feat11  = box_cox_transform_and_normalize(feat11)
# =================================================

feat1 = T.pcaData(feat1 ,1)
feat2 = T.pcaData(feat2 ,1)
feat3 = T.pcaData(feat3 ,1)
feat4 = T.pcaData(feat4 ,1)
feat5 = T.pcaData(feat5 ,1)
feat6 = T.pcaData(feat6 ,1)
feat7 = T.pcaData(feat7 ,1)
feat8 = T.pcaData(feat8 ,1)
feat9 = T.pcaData(feat9 ,1)
feat10 = T.pcaData(feat10,1)
feat11 = T.pcaData(feat11,1)

featComb = T.concat_multiple_dfs(feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8,feat9,feat10,feat11
                                 ,T.select_column(data,'Programme'))

features, labels = getFeaturesAndLabels(featComb, 'Programme')
labels = labels - 1
# 数据预处理和划分
# 70%特征（用于训练），70% 正确标签，30%特征（用于验证），30%正确标签
features_train, labels_train, features_test, labels_test, scaler = preprocessAndSplit(features, labels, 0.3,0)

# ===================================================================================
# ===================================================================================
logistic = LogisticRegression()
logistic.fit(features_train, labels_train)
labels = logistic.predict(features_test)
print(accuracy_score(labels_test, labels))
plot_confusion_matrix(labels_test, labels, title='Confusion matrix',
                      class_names=['Prog. A','Prog. B','Prog. C','Prog. D'],
                      fileName ='ConfusionMatrix_1.pdf')
# ===================================================================================
# ===================================================================================

# 初始化Boosting类模型（XGBoost）
boost_model = XGBClassifier(
    n_estimators=70,  # 基分类器（树）的数量（可根据数据调整）
    max_depth=1,       # 单棵树的最大深度（控制复杂度，防止过拟合）
    learning_rate=0.4,  # 学习率（较小值需更多树，但泛化性更好）
    random_state=0
)
# 用训练集训练模型（使用你已定义的features_train和labels_train）
boost_model.fit(features_train, labels_train)
# 用测试集预测标签（使用你已定义的features_test）
labels_pred_boost = boost_model.predict(features_test)
# 计算并打印准确率（使用你已定义的labels_test）
print("Boosting模型准确率:", accuracy_score(labels_test, labels_pred_boost))
plot_confusion_matrix(labels_test, labels_pred_boost, title='Confusion matrix',
                      class_names=['Prog. A','Prog. B','Prog. C','Prog. D'],
                      fileName ='ConfusionMatrix_2.pdf')
# ===================================================================================
# ===================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 假设你已经准备好的数据集（需要替换为实际数据）
# features_train: (n_train, 3) numpy数组
# labels_train: (n_train,) numpy数组（类别标签，0-3）
# features_test: (n_test, 3) numpy数组
# labels_test: (n_test,) numpy数组（实际使用时如果已有真实标签可用于评估）

class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0)]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, d_model=128, nhead=8, num_layers=3, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入嵌入层（将3维特征转换为d_model维嵌入）
        self.embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=False  # 输入格式为(seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 分类头（取序列最后一个位置的输出）
        self.classifier = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 注意这里的维度顺序
        """
        # 转换维度顺序为(seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)

        # 嵌入层：(seq_len, batch_size, d_model)
        x = self.embedding(x) * np.sqrt(self.d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码：(seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出：(batch_size, d_model)
        x = x[-1, :, :]

        # 分类头：(batch_size, output_dim)
        logits = self.classifier(x)
        return logits


def train_model(features_train, labels_train, features_test, model, epochs=50, batch_size=32, lr=1e-4):
    # 转换为Tensor
    train_tensor = TensorDataset(
        torch.tensor(features_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            # 输入格式：(batch_size, seq_len=1, input_dim=3)
            batch_features = batch_features.unsqueeze(1)  # 增加时间步维度

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')


def predict_labels(features_test, model):
    model.eval()
    with torch.no_grad():
        # 转换为Tensor并添加时间步维度
        test_tensor = torch.tensor(features_test, dtype=torch.float32).unsqueeze(1)
        outputs = model(test_tensor)
        predicted_labels = torch.argmax(outputs, dim=1).numpy()
    return predicted_labels


if __name__ == "__main__":
    # 超参数设置（可根据实际情况调整）
    INPUT_DIM = 11  # 输入特征维度（用户需求）
    OUTPUT_DIM = 4  # 输出类别数（用户需求）
    D_MODEL = 128  # Transformer隐藏层维度
    NHEAD = 2  # 多头注意力头数
    NUM_LAYERS = 3  # Transformer编码层数
    EPOCHS = 150  # 训练轮数
    BATCH_SIZE = 32  # 批次大小
    LR = 1e-3  # 学习率

    # 初始化模型
    model = TransformerClassifier(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM
    )

    # 训练模型（请替换为实际数据）
    train_model(features_train, labels_train, features_test, model, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    # 预测测试集标签（示例调用）
    labels = predict_labels(features_test, model)

    print(accuracy_score(labels_test,labels))
    plot_confusion_matrix(labels_test, labels, title='Confusion matrix',
                          class_names=['Prog. A', 'Prog. B', 'Prog. C', 'Prog. D'],
                          fileName='ConfusionMatrix_3.pdf')


