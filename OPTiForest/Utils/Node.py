from __future__ import annotations
import numpy as np
from Utils.L2SH import L2SH
from Utils.Data import Data
from scipy.spatial import distance


class Node:
    def __init__(self, data: Data, children: list[Node] = None, center: np.ndarray = None, size: int = None, bucket_size=3):
        self.data = data
        self.children = {}
        if children is not None:
            for index, node in enumerate(children):
                self.children[index] = node
        self.center = data.mean() if center is None else center
        self.size = data.size() if size is None else size
        self.clustered = children is not None
        self.bucket_size = bucket_size
        self.l2sh = L2SH(n_features=self.center.shape[-1], bucket_size=bucket_size)

    def generate_children(self, depth=0, max_depth=5):
        if depth == max_depth or self.size == 1:
            self.children = None
            return

        buckets = self.l2sh.get_buckets(self.data.X)
        unique_buckets = np.unique(buckets)
        self.children = {}

        for bucket in unique_buckets:
            indices = np.nonzero(np.asarray(buckets == bucket))
            data = Data(self.data.X[indices], self.data.indices[indices])
            child = Node(data=data, bucket_size=self.bucket_size)
            child.generate_children(depth=depth + 1, max_depth=max_depth)
            self.children[bucket] = child

    def get_next(self, x: np.ndarray):
        if self.children is None:
            return None

        if self.clustered:
            argmin = 0
            dist_min = float("inf")
            for index, node in self.children.items():
                dist = distance.euclidean(node.center, x)
                if dist < dist_min:
                    dist_min = dist
                    argmin = index
            return self.children[argmin]
        else:
            bucket = self.l2sh.get_bucket(x)
            if bucket in self.children:
                return self.children[self.l2sh.get_bucket(x)]
            else:
                return None

    def get_branching_factor(self) -> (int, int):
        if self.children is None:
            return 0, 0

        branches, nodes = len(self.children.keys()), 1
        for child in self.children.values():
            branch, node = child.get_branching_factor()
            branches += branch
            nodes += node

        return branches, nodes
