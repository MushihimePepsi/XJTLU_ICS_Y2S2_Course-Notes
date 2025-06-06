import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

# Set English font for matplotlib
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # Ensure minus sign is displayed correctly
warnings.filterwarnings('ignore')  # Ignore warnings


def box_cox_transform(feature):
    """Apply Box-Cox transformation to a feature"""
    # Add a small constant to ensure all values are positive
    feature = feature + 1e-10
    transformed, lambda_val = stats.boxcox(feature)
    return transformed, lambda_val


def normalize_data(data):
    """Standardize the data using StandardScaler"""
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def create_qq_plots(data, transformed_data, columns):
    """Create QQ plots comparing original and transformed data"""
    n_features = len(columns)

    # Adjust figure size and subplot layout for horizontal display
    fig, axes = plt.subplots(2, n_features, figsize=(5 * n_features, 10))

    for i, col in enumerate(columns):
        # QQ plot for original data (top row)
        ax1 = axes[0, i]
        stats.probplot(data[col], dist="norm", plot=ax1)
        ax1.set_title(f'Original {col}', fontsize=10)
        ax1.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax1.set_ylabel('Sample Quantiles', fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=8)

        # QQ plot for transformed data (bottom row)
        ax2 = axes[1, i]
        stats.probplot(transformed_data[:, i], dist="norm", plot=ax2)
        ax2.set_title(f'Transformed {col}', fontsize=10)
        ax2.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax2.set_ylabel('Sample Quantiles', fontsize=9)
        ax2.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    return fig


def process_data(data):
    """Process data: apply Box-Cox transformation, standardization, and generate QQ plots"""
    # Copy the data to avoid modifying the original
    processed_data = data.copy()

    # Identify columns to transform (exclude categorical variables)
    numeric_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

    # Store transformed data
    transformed_array = np.zeros((len(data), len(numeric_cols)))

    # Apply Box-Cox transformation to each numeric feature
    lambda_values = {}
    for i, col in enumerate(numeric_cols):
        transformed, lambda_val = box_cox_transform(data[col])
        transformed_array[:, i] = transformed
        lambda_values[col] = lambda_val
        processed_data[f'{col}_box_cox'] = transformed

    # Standardize the transformed data
    normalized_data = normalize_data(transformed_array)

    # Add normalized data back to the DataFrame
    for i, col in enumerate(numeric_cols):
        processed_data[f'{col}_normalized'] = normalized_data[:, i]

    # Create QQ plots
    fig = create_qq_plots(data, normalized_data, numeric_cols)

    # For categorical variables, perform one-hot encoding and standardization
    categorical_cols = ['Programme', 'Gender']
    for col in categorical_cols:
        # Convert categorical variable to numerical using one-hot encoding
        dummies = pd.get_dummies(data[col], prefix=col)
        # Standardize the encoded data
        normalized_dummies = normalize_data(dummies)
        # Add back to the processed data
        for j, dummy_col in enumerate(dummies.columns):
            processed_data[f'{dummy_col}_normalized'] = normalized_dummies[:, j]

    return processed_data, lambda_values, fig


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    data = pd.read_excel('training_set.xlsx')

    # Process the data
    processed_data, lambda_values, fig = process_data(data)

    # Display results
    print("First few rows of processed data:")
    print(processed_data.head())

    print("\nLambda values for Box-Cox transformations:")
    for col, lambda_val in lambda_values.items():
        print(f"{col}: {lambda_val:.4f}")

    # Save QQ plots
    fig.savefig('Task_1_4_BoxCox.pdf', format='pdf', dpi=300)
    plt.show()