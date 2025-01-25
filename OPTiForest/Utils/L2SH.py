from numpy.random import normal, uniform
import numpy as np


class L2SH:
    def __init__(self, n_features=3, bucket_size=4):
        self.n_features = n_features
        self.bucket_size = bucket_size

        self.a = normal(0, 1, n_features)
        self.b = uniform(0, bucket_size)

    def get_bucket(self, value: np.ndarray) -> int:
        return np.floor((np.dot(value, self.a) + self.b) / self.bucket_size)

    def get_buckets(self, values: np.ndarray) -> np.ndarray:
        return np.floor(((values @ np.atleast_2d(self.a).T) + self.b) / self.bucket_size).flatten().astype(int)
