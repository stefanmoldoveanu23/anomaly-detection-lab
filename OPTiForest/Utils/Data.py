from __future__ import annotations
import numpy as np


class Data:
    def __init__(self, x: np.ndarray, indices: np.ndarray):
        self.X = x
        self.indices = indices

    def sample(self, size) -> Data:
        indices = np.random.choice(len(self.indices), size=size, replace=False)
        return Data(self.X[indices], self.indices[indices])

    def mean(self) -> np.ndarray:
        return np.mean(self.X, axis=0)

    def size(self) -> int:
        return self.X.shape[0]
