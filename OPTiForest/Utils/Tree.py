from __future__ import annotations
from Utils.Data import Data
from Utils.Node import Node
from threading import Thread
import numpy as np
from typing import Dict, Any


class Tree:
    def __init__(self, data: Data, bucket_size=4, granularity=1):
        psi = np.random.uniform(low=6, high=11)
        sample_size = min(int(2.0 ** psi), data.X.shape[0])
        max_depth = int(2.0 * psi + 0.8327)
        self.branching_factor = 0
        self.ref_depth = max_depth
        data = data.sample(size=sample_size)

        self.granularity = granularity
        self.root = Node(data=data, bucket_size=bucket_size)
        self.thread = Thread(target=self.root.generate_children, args=(0, max_depth,))
        self.thread.start()

    def set_branching_factor(self):
        branches, nodes = self.root.get_branching_factor()
        self.branching_factor = 1.0 * branches / nodes
        self.ref_depth = self.get_mu(self.root.data.X.shape[0])

    def predict_one(self, x: np.ndarray):
        depth = 0
        th_depth = 0
        node = self.root
        adjustment = 0

        while True:
            depth += 1
            nxt = node.get_next(x)
            if nxt is None:
                if node.children is not None:
                    depth += 1
                    th_depth += 2
                else:
                    th_depth += 1
                    adjustment = self.get_mu(node.data.X.shape[0])
                break

            if len(node.children.keys()) > 1:
                th_depth += 1
            node = nxt

        depth_score = th_depth * np.power(1.0 * depth / max(th_depth, 1), self.granularity) + adjustment
        return 2.0 ** (-1.0 * depth_score / self.ref_depth)

    def predict(self, x: np.ndarray, store_array: np.ndarray, store_index: int):
        for index, x_item in enumerate(x):
            store_array[index, store_index] = self.predict_one(x_item)

    def get_mu(self, size):
        if size <= 1:
            return 0
        elif size <= round(self.branching_factor):
            return 1
        else:
            return (np.log(size) + np.log(self.branching_factor - 1.0) + 0.5772) / np.log(self.branching_factor) - 0.5
