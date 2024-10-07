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
    
