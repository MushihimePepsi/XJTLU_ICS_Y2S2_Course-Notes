# This is a file that demonstrates what you need to do over
# the live demonstration session. You need to
# 1. Load the given data from the file.
# 2. Cluster the data.
# 3. Run the evaluation function.

import pandas as pd
from sklearn.cluster import KMeans
from evaluate import evaluate_clustering

# Step 1: Read data from an Excel file
# Make sure to replace 'data.xlsx' with the path to your Excel file.
# Change the file name accordingly.
data = pd.read_excel('sample_data.xlsx')

# Step 2: Prepare the data for clustering
# Assuming the label information is in the first column and other columns
# contain feature information of samples.
# Adjust the column selection as necessary.
X = data.iloc[:, 1:].values  # Use all columns except the last one for clustering

# Step 3: Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Step 4: Evaluate the clustering results
# Remember to change the file name for different algorithms!
ratio = evaluate_clustering(X, labels, output_file='clustering_kmeans.csv')

# For the live demonstration for the supervised learning, you need to:
# 1. Load the given data from the file.
# 2. Prepare the data.
# 3. Classify the samples
# 4. Run the evaluation function.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from evaluate import evaluate_classification

# Step 1 is the same with the previous sections hence will not
# be repeated.

# Step 2: Prepare the data for clustering
# Assuming the label information is in the first column and other columns
# contain feature information of samples.
# Adjust the column selection as necessary.
X2 = data.iloc[:, 1:].values  # Use all columns except the last one for clustering
label_gt = data.iloc[:, 0].values

# Set up the training dataset and the testing dataset
X_train, X_test, y_train, y_test = train_test_split(X2, label_gt, test_size=0.2, random_state=42)

# Step 3: Train a classifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

evaluate_classification(y_pred, "classification_knn.csv")