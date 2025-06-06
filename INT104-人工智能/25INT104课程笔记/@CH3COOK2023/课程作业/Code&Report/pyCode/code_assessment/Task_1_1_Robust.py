import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# Load data from Excel file
data = pd.read_excel('training_set.xlsx')

# Apply Robust Scaling
scaler = RobustScaler()
data_robust = scaler.fit_transform(data)
data_robust = pd.DataFrame(data_robust, columns=data.columns)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot boxplot for original data
sns.boxplot(data=data, ax=axes[0, 0])
axes[0, 0].set_title('Boxplot of Original Data')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot boxplot for robust scaled data
sns.boxplot(data=data_robust, ax=axes[0, 1])
axes[0, 1].set_title('Boxplot of Robust Scaled Data')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot density plot for original data
data.plot(kind='density', ax=axes[1, 0])
axes[1, 0].set_title('Density Plot of Original Data')
axes[1, 0].set_xlabel('Feature Values')
axes[1, 0].set_ylabel('Density')

# Plot density plot for robust scaled data
data_robust.plot(kind='density', ax=axes[1, 1])
axes[1, 1].set_title('Density Plot of Robust Scaled Data')
axes[1, 1].set_xlabel('Feature Values')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.savefig('Task_1_1_Robust.pdf', format='pdf', bbox_inches='tight')
plt.show()
