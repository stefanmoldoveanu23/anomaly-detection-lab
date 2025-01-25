from __future__ import annotations
from Utils.Data import Data
from Utils.Node import Node
from threading import Thread
import numpy as np


class Tree:
    def __init__(self, data: Data, bucket_size=4):
        psi = np.random.uniform(low=6, high=11)
        sample_size = min(int(2.0 ** psi), data.X.shape[0])
        self.max_depth = int(2.0 * psi + 0.8327)
        data = data.sample(size=sample_size)

        self.root = Node(data=data, bucket_size=bucket_size)
        self.thread = Thread(target=self.root.generate_children, args=(0, self.max_depth,))
        self.thread.start()

    def predict_one(self, x: np.ndarray):
        depth = 0
        node = self.root

        while True:
            node = node.get_next(x)
            depth += 1
            if node is None:
                break

        return 2.0 ** (-1.0 * depth / self.max_depth)

    def predict(self, x: np.ndarray, store_array: np.ndarray, store_index: int):
        for index, x_item in enumerate(x):
            store_array[index, store_index] = self.predict_one(x_item)
