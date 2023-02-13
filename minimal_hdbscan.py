# A minimal HDBSCAN designed for reimplementation in other languages
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from itertools import combinations
from typing import List, Tuple, Dict, Set, Optional, Union
from dataclasses import dataclass, field
import numpy as np

K_NEIGHBORS = 5
# disjoint set specialized for single-linkage clustering
class DoubleDisjointSet:
    def __init__(self, n):
        self.n = n
        self.parent = np.arange(2*n-1)
        self.next_label = n
    
    def find_set(self, u):
        st = []
        while self.parent[u] != u:
            st.append(u)
            u = self.parent[u]
        for v in st:
            self.parent[v] = u
        return u

    def _alloc_next_label(self):
        res = self.next_label
        self.next_label += 1
        return res
    
    def union_sets(self, u, v):
        up = self.find_set(u)
        vp = self.find_set(v)
        if up != vp:
            new_parent = self._alloc_next_label()
            self.parent[up] = new_parent
            self.parent[vp] = new_parent
            return True, up, vp
        else:
            return False, up, vp

@dataclass
class DoubleEdge:
    parent : int
    left : Optional[int]
    right : Optional[int]

    def delete_left_child(self) -> "DoubleEdge":
        return DoubleEdge(self.parent, None, self.right)
    
    def delete_right_child(self) -> "DoubleEdge":
        return DoubleEdge(self.parent, self.left, None)

class SimplifiedDendrogram:
    def __init__(self, n):
        self.n = n
        self.dsu = DoubleDisjointSet(n)
        self.next_label = n
        self.edges = []
        self.numchildren = np.ones(n * 2 - 1, dtype=np.int32)
        self.forget = np.zeros(n * 2 - 1, dtype=np.bool)
        self.stability = np.zeros(n * 2 - 1, dtype=np.float32)
        self.weight_birth = np.zeros(n * 2 - 1, dtype=np.float32)
        self.weight_death = np.zeros(n * 2 - 1, dtype=np.float32)
        self.marked = set()
        for i in range(n):
            self.numchildren[i] = 1

    def try_join(self, u, v, w):
        success, u, v = self.dsu.union_sets(u, v)
        if success:
            next_label = self.next_label
            self.next_label += 1
            self.edges.append(DoubleEdge(next_label, u, v))
            self.numchildren[next_label] = self.numchildren[u] + self.numchildren[v]
            self.weight_birth[next_label] = w
            self.weight_death[u] = self.weight_death[v] = w
            self.assign_initial_stability(u)
            self.assign_initial_stability(v)
    
    def assign_initial_stability(self, u):
        if self.numchildren[u] == 1:
            self.stability[u] = 0
        else:
            s = (1 / self.weight_birth[u] - 1 / self.weight_death[u]) * self.numchildren[u]
            self.stability[u] = s

    def condense(self, min_points = 5):
        for e in self.edges:
            if self.numchildren[e.left] < min_points and self.numchildren[e.right] < min_points:
                e.left = -e.left - 1
                e.right = -e.right - 1
            elif self.numchildren[e.left] < min_points:
                e.left = -e.left - 1
                self.stability[e.right] += self.stability[e.parent]
            elif self.numchildren[e.right] < min_points:
                e.right = -e.right - 1
                self.stability[e.left] += self.stability[e.parent]

    def _mark_clusters(self):
        self.marked.clear()
        for e in self.edges:
            if e.left >= 0 and e.right >= 0:
                # print(e.left, e.right, e.parent)
                summed_stability = self.stability[e.left] + self.stability[e.right]
                if summed_stability < self.stability[e.parent]:
                    self.marked.add(e.parent)
                else:
                    self.stability[e.parent] = summed_stability
            elif e.left >= 0:
                self.stability[e.parent] = self.stability[e.left]
                if e.left in self.marked:
                    self.marked.add(e.parent)
            elif e.right >= 0:
                self.stability[e.parent] = self.stability[e.right]
                if e.right in self.marked:
                    self.marked.add(e.parent)
            else:
                pass
                # self.marked.add(e.parent)

    def extract_clusters(self):
        self._mark_clusters()
        labels = np.full(self.n, -1)
        parent_label_map = {}
        next_label = 0
        for e in reversed(self.edges):
            is_marked = e.parent in self.marked
            if is_marked and e.parent not in parent_label_map:
                parent_label_map[e.parent] = next_label
                next_label += 1
            normalized_left = e.left if e.left >= 0 else -e.left - 1
            normalized_right = e.right if e.right >= 0 else -e.right - 1
            if e.parent in parent_label_map:
                if normalized_left < self.n:
                    labels[normalized_left] = parent_label_map[e.parent]
                if normalized_right < self.n:
                    labels[normalized_right] = parent_label_map[e.parent]
                parent_label_map[normalized_left] = parent_label_map[normalized_right] = parent_label_map[e.parent]
        return labels

# driver class for running HDBSCAN on points in R^n
class HDBScan:
    kdtree: KDTree
    n: int

    def __init__(self, points: Union[List[Tuple[float, float]], np.ndarray]):
        data_matrix = np.array(points)
        self.kdtree = KDTree(data_matrix)
        self.n = len(points)

    def mutual_reachability_distance(self, p1: int, p2: int) -> float:
        """
        Calculates the mutual reachability distance between two points.
        """
        return max(
            euclidean(self.kdtree.data[p1], self.kdtree.data[p2]),
            self.kdtree.query(self.kdtree.data[p1], k=K_NEIGHBORS)[0][-1],
            self.kdtree.query(self.kdtree.data[p2], k=K_NEIGHBORS)[0][-1],
        )

    def join_sequence(self):
        # FIXME: this is a brute force way to enumerate join orders
        distances = [
            (self.mutual_reachability_distance(u, v), u, v)
            for u, v in combinations(range(self.n), 2)
        ]
        for d, u, v in sorted(distances):
            yield u, v, d