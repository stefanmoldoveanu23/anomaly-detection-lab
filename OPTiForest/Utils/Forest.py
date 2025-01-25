from Utils.Tree import Tree
from Utils.Data import Data
import numpy as np
from threading import Thread


class Forest:
    def __init__(self, data: Data, bucket_size=4, granularity=1, n_trees=100):
        self.trees = [Tree(data, bucket_size=bucket_size, granularity=granularity) for _ in range(n_trees)]

        for tree in self.trees:
            tree.thread.join()

        for tree in self.trees:
            tree.set_branching_factor()

    def predict(self, x: np.ndarray) -> np.ndarray:
        results = np.array([[0.0] * len(self.trees) for _ in range(x.shape[0])])
        for index, tree in enumerate(self.trees):
            tree.thread = Thread(target=tree.predict, args=(x, results, index,))
            tree.thread.start()

        for tree in self.trees:
            tree.thread.join()

        return np.average(results, axis=1)
