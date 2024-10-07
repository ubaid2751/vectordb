from dbscan import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic dataset
n_samples = 300
n_clusters = 4
random_state = 42

# Use make_blobs to create clear clusters
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=random_state)

# Set DBSCAN parameters
ε = 1.5  
MinPts = 5  

dbscan_model = DBSCAN(ε, MinPts)
cluster_labels = dbscan_model.fit(X)

clusters = {i: X[cluster_labels == i] for i in range(np.max(cluster_labels) + 1)}
noise = X[cluster_labels == -1]

print("Clusters:")
for idx, points in clusters.items():
    print(f"Cluster {idx + 1}: {points}")

print("\nNoise points:", noise)

# Visualization
plt.scatter(X[:, 0], X[:, 1], color='black', label='Data Points')

for cluster_id, cluster in clusters.items():
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {cluster_id + 1}')

if noise.size > 0:
    plt.scatter(noise[:, 0], noise[:, 1], color='red', label='Noise', marker='x')

plt.title('DBSCAN Clustering with Clear Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()