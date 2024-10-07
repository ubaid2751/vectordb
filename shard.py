import numpy as np

class Shard:
    def __init__(self, shard_count:int, shard_id: int, dimension: int):
        self.shard_id = shard_id
        self.shard_count = shard_count
        self.dimension = dimension
        self.vectors = {}
        self.leader_vector = None
        
    def add_vector(self, vectors):
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(f"Vector Dimensionality must be {self.dimension}")
            
            if self.shard_count < len(vectors):
                vector_id = hash(tuple(vector))
                self.vectors[vector_id] = vector
                
    def update_leader(self):
        if len(self.vectors) == 0:
            return
        
        self.leader_vector = self.compute_medoid()
        
    