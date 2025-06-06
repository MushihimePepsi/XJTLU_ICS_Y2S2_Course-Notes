import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from scipy import stats
# Load data
data = pd.read_excel('training_set.xlsx')

# Select features to scale
features_to_scale = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# Create a copy of the original data and scale selected features
data_robust = data.copy()
scaler = RobustScaler()
data_robust[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Perform Shapiro-Wilk test
alpha = 0.05  # Significance level
results = []

for feature in features_to_scale:
    # Test for original data
    stat_orig, p_orig = stats.shapiro(data[feature])
    is_normal_orig = p_orig > alpha

    # Test for scaled data
    stat_robust, p_robust = stats.shapiro(data_robust[feature])
    is_normal_robust = p_robust > alpha

    # Record the results
    results.append({
        'Feature': feature,
        'Original Statistic': stat_orig,
        'Original p-value': p_orig,
        'Original Normal': is_normal_orig,
        'Robust Statistic': stat_robust,
        'Robust p-value': p_robust,
        'Robust Normal': is_normal_robust
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print Shapiro-Wilk test results
print("\nShapiro-Wilk Normality Test Results (α=0.05):")
print("=" * 60)
print("{:<8} {:<15} {:<15} {:<15} {:<15}".format(
    "Feature", "Original Statistic", "Original p-value", "Robust Statistic", "Robust p-value"))
print("-" * 60)

for _, row in results_df.iterrows():
    feature = row['Feature']
    orig_stat = f"{row['Original Statistic']:.4f}"
    orig_p = f"{row['Original p-value']:.4f}"
    robust_stat = f"{row['Robust Statistic']:.4f}"
    robust_p = f"{row['Robust p-value']:.4f}"

    # Add asterisk for significant results
    if row['Original p-value'] < alpha:
        orig_p += "*"
    if row['Robust p-value'] < alpha:
        robust_p += "*"

    print("{:<8} {:<15} {:<15} {:<15} {:<15}".format(
        feature, orig_stat, orig_p, robust_stat, robust_p))

print("\n* Indicates significant deviation from normality at α=0.05 level")