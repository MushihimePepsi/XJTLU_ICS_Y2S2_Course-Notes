import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 创建连接矩阵
Z = np.array([
    [0, 3, 1, 2],  # A(0) 和 D(3) 在高度1合并
    [1, 2, 1, 2],  # B(1) 和 C(2) 在高度1合并
    [4, 5, 3, 3],  # {B,C}(4) 和 E(5) 在高度3合并
    [6, 7, 5, 5]  # {A,D}(6) 和 {B,C,E}(7) 在高度5合并
], dtype=np.float64)

labels = ['A', 'B', 'C', 'D', 'E']

plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    labels=labels,
    leaf_rotation=0,
    leaf_font_size=12,
    color_threshold=0
)
plt.title('凝聚式聚类树状图')
plt.xlabel('数据点')
plt.ylabel('距离')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()