import numpy as np
import pandas as pd
import csv
from sklearn.metrics import pairwise_distances

def evaluate_clustering(X, labels, output_file='predicted_labels.csv'):
    """
    Evaluate the clustering result by calculating the ratio of intra-cluster distance
    to inter-cluster distance.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    labels : array-like, shape (n_samples,)
        The cluster labels for each sample.

    Returns:
    float
        The ratio of intra-cluster distance to inter-cluster distance.
    """
    unique_labels = np.unique(labels)

    # Calculate intra-cluster distances
    intra_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_distance = np.mean(pairwise_distances(cluster_points))
            intra_distances.append(intra_distance)

    # Calculate inter-cluster distances
    inter_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            inter_distance = np.mean(pairwise_distances(cluster_i, cluster_j))
            inter_distances.append(inter_distance)

    # Calculate the average intra-cluster and inter-cluster distances
    avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
    avg_inter_distance = np.mean(inter_distances) if inter_distances else 1  # Avoid division by zero

    # Calculate the ratio
    ratio = avg_intra_distance / avg_inter_distance if avg_inter_distance != 0 else float('inf')

    # Save label information to a CSV file
    label_df = pd.DataFrame({'cluster_index': labels})
    label_df.to_csv(output_file, index=False)
    print("File saved successfully in", output_file)

    # Print key metric
    print(f"Intra-cluster to Inter-cluster distance ratio: {ratio:.4f}")

    return ratio

def evaluate_classification(predicted_labels, filename='predicted_labels.csv'):
    """
    Save predicted labels to a specified CSV file.

    Parameters:
    predicted_labels (list): A list of predicted labels to save.
    filename (str): The name of the CSV file where the labels will be saved.
    """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for label in predicted_labels:
                writer.writerow([label])  # Write each label in a new row
        print(f"Predicted labels successfully saved to {filename}.")
    except Exception as e:
        print(f"An error occurred while saving labels: {e}")