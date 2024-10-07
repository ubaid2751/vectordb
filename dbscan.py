import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class DBSCAN:
    def __init__(self, eps: float, MinPts: int):
        self.MinPts = MinPts
        self.eps = eps
    
    def getNeighbors(self, idx, X):
        neighbors = []
        P = X[idx]
        for i, Q in enumerate(X):
            if np.linalg.norm(P - Q) < self.eps:
                neighbors.append(i)
        return neighbors
    
    def expandCluster(self, P_idx, NeighborPts, X, visited, cluster_labels, C_id):
        cluster_labels[P_idx] = C_id
        
        i = 0
        while i < len(NeighborPts):
            Q_idx = NeighborPts[i]
            
            if not visited[Q_idx]:
                visited[Q_idx] = True
                Q_NeighborPts = self.getNeighbors(Q_idx, X)
                
                if len(Q_NeighborPts) >= self.MinPts:
                    NeighborPts.extend(Q_NeighborPts)
                    
            if cluster_labels[Q_idx] == -1:  # Add to cluster if not already assigned
                cluster_labels[Q_idx] = C_id
            i += 1

    def fit(self, X):
        m, _ = X.shape
        visited = np.full(m, False)
        cluster_labels = np.full(m, -1)  # -1 indicates noise
        C_id = 0
        
        for idx in range(m):
            if not visited[idx]:
                visited[idx] = True
                
                NeighborPts = self.getNeighbors(idx, X)
                if len(NeighborPts) < self.MinPts:
                    continue  # Mark as noise (-1 by default in cluster_labels)
                else:
                    self.expandCluster(idx, NeighborPts, X, visited, cluster_labels, C_id)
                    C_id += 1
        
        return cluster_labels
    
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
