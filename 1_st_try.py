import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Mall Customers dataset
data = pd.read_csv('Mall_Customers.csv')

# Select the relevant features for clustering
selected_features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = data[selected_features]

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid()
plt.show()

# Based on the Elbow plot, choose the optimal number of clusters (e.g., K=5)

# Perform K-means clustering with the chosen number of clusters
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
data['Cluster'] = clusters

# Explore the clusters and their characteristics
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_features)
cluster_centers['Cluster'] = range(0, k)
cluster_sizes = data['Cluster'].value_counts().sort_index()

# Analyze and visualize the results
for cluster_num in range(k):
    cluster_data = data[data['Cluster'] == cluster_num]
    print(f"Cluster {cluster_num} ({cluster_sizes[cluster_num]} customers):")
    print(cluster_centers.iloc[cluster_num])
    print()

# You can now use the cluster labels to personalize marketing strategies for each customer segment
