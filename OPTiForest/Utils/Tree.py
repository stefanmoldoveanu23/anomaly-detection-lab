from __future__ import annotations
from Utils.Data import Data
from Utils.Node import Node
from threading import Thread
import numpy as np
from scipy.spatial import distance


class Tree:
    def __init__(self, data: Data, bucket_size=4, granularity=1, threshold=403):
        psi = np.random.uniform(low=6, high=11)
        sample_size = min(int(2.0 ** psi), data.size())
        max_depth = int(2.0 * psi + 0.8327)
        self.branching_factor = 0
        self.ref_depth = max_depth
        data = data.sample(size=sample_size)

        self.granularity = granularity
        self.threshold = threshold
        self.root = Node(data=data, bucket_size=bucket_size)
        self.thread = Thread(target=self.build, args=(max_depth,))
        self.thread.start()

    def build(self, max_depth):
        self.root.generate_children(depth=0, max_depth=max_depth)
        self.optimize_tree()
        self.set_branching_factor()

    def set_branching_factor(self):
        branches, nodes = self.root.get_branching_factor()
        self.branching_factor = 1.0 * branches / nodes
        self.ref_depth = self.get_mu(self.root.size)

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
                    adjustment = self.get_mu(node.size)
                break

            if len(node.children.keys()) > 1:
                th_depth += 1
            node = nxt

        depth_score = th_depth * np.power(1.0 * depth / max(th_depth, 1), self.granularity) + adjustment
        return np.power(2.0, (-1.0 * depth_score / self.ref_depth))

    def predict(self, x: np.ndarray, store_array: np.ndarray, store_index: int):
        for index, x_item in enumerate(x):
            store_array[index, store_index] = self.predict_one(x_item)

    def get_clusters(self, node: Node, nodes: list[Node]):
        if node.children is None or node.size < self.threshold:
            nodes.append(node)
            return

        for child in node.children.values():
            self.get_clusters(child, nodes)

    @staticmethod
    def get_center(nodes: list[Node], indices: list[int]) -> np.ndarray:
        center: np.ndarray = np.zeros(nodes[0].center.shape[-1])
        data_size = 0
        for index in indices:
            center = center + (nodes[index].center * nodes[index].size)
            data_size += nodes[index].size

        center = center / data_size
        return center

    @staticmethod
    def get_distance(nodes: list[Node], indices: list[int]):
        center = Tree.get_center(nodes, indices)
        dist = 0.0
        for index in indices:
            dist += (distance.euclidean(center, nodes[index].center) * nodes[index].size)

        return dist

    @staticmethod
    def get_new_cluster(nodes: list[Node], branching_factor: int) -> list[int]:
        best_distance = float("inf")
        sol = []

        if branching_factor == 2:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    dist = Tree.get_distance(nodes, [i, j])
                    if dist < best_distance:
                        best_distance = dist
                        sol = [j, i]
        else:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        dist = Tree.get_distance(nodes, [i, j, k])
                        if dist < best_distance:
                            best_distance = dist
                            sol = [k, j, i]

        return sol

    def optimize_tree(self):
        nodes = []
        self.get_clusters(self.root, nodes)

        while len(nodes) > 1:
            branching_factor = min(2 if np.random.uniform(low=0, high=1) < 0.282 else 3, len(nodes))
            sol_indices = Tree.get_new_cluster(nodes, branching_factor)
            center = Tree.get_center(nodes, sol_indices)
            data_size = 0
            children = []
            bucket_size = nodes[sol_indices[0]].bucket_size

            for index in sol_indices:
                children.append(nodes[index])
                data_size += nodes[index].size
                del nodes[index]

            new_node = Node(Data(np.zeros([0]), np.zeros([0])), children=children, center=center, size=data_size, bucket_size=bucket_size)
            nodes.append(new_node)

        self.root = nodes[0]

    def get_mu(self, size):
        if size <= 1:
            return 0
        elif size <= round(self.branching_factor):
            return 1
        else:
            return (np.log(size) + np.log(self.branching_factor - 1.0) + 0.5772) / np.log(self.branching_factor) - 0.5
