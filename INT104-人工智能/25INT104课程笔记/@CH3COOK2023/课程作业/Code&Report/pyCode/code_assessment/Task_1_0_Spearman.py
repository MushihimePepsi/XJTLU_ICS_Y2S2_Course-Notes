import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成示例数据，这里你可以替换为你自己的dataframe
data = pd.read_excel('training_set.xlsx')

# 计算Spearman相关性矩阵
corr_matrix = data.corr(method='spearman')

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Heatmap')
plt.savefig('Task_1_0_Spearman.pdf', format='pdf', bbox_inches='tight')
plt.show()
