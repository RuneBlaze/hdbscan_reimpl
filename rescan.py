from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from itertools import combinations
from typing import List, Tuple, Dict, Set, Optional, Union
from dataclasses import dataclass, field
import numpy as np

K_NEIGHBORS = 5

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

@dataclass
class WeightedEdge:
    u: int
    v: int
    weight: float

def zeros(n):
    return [0 for i in range(n)]

def ones(n):
    return [1 for i in range(n)]

class DisjointSet:
    def __init__(self, n):
        self.n = n
        self.parent = np.arange(n)
    
    def find_set(self, u):
        st = []
        while self.parent[u] != u:
            st.append(u)
            u = self.parent[u]
        for v in st:
            self.parent[v] = u
        return u

    def union_sets(self, u, v):
        up = self.find_set(u)
        vp = self.find_set(v)
        if up != vp:
            self.parent[up] = vp
            return True, up, vp
        else:
            return False, up, vp

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
class DendrogramWithStability:
    n: int
    parent: List[int]
    firstchild: List[int]
    nextsibling: List[int]
    numchildren: List[int]
    weight_birth: List[int]
    weight_death: List[int]
    stability: List[int]
    total_nodes: List[int]
    forget: List[bool]
    to_alloc: int

    onechild: Set[int] = field(default_factory=set)
    marked: Set[int] = field(default_factory=set)
    roots: Set[int] = field(default_factory=set)

    @staticmethod
    def starshaped(n):
        parent = list(range(2 * n - 1))
        nextsibling = list(range(2 * n - 1))
        numchildren = [0 for i in range(2 * n - 1)]
        firstchild = list(range(2 * n))
        res = DendrogramWithStability(
            n,
            parent,
            firstchild,
            nextsibling,
            numchildren,
            zeros(2 * n - 1),
            zeros(2 * n - 1),
            zeros(2 * n - 1),
            ones(2 * n - 1),
            [False for _ in range(2 * n - 1)],
            n,
        )
        res.roots = set(range(n))
        return res

    def find(self, u):
        while self.parent[u] != u:
            u = self.parent[u]
        return u

    def union(self, u, v, w):
        z = self.to_alloc
        self.to_alloc += 1
        self.nextsibling[u] = v
        self.parent[u] = z
        self.parent[v] = z
        self.roots.remove(u)
        self.roots.remove(v)
        self.roots.add(z)
        self.firstchild[z] = u
        self.numchildren[z] = 2
        self.weight_death[u] = w
        self.weight_death[v] = w
        self.assign_initial_stability(u)
        self.assign_initial_stability(v)
        self.weight_birth[z] = w
        self.total_nodes[z] = self.total_nodes[u] + self.total_nodes[v]

    def assign_initial_stability(self, u):
        if self.weight_birth[u] == 0:
            self.stability[u] = 0
            return
        s = (1 / self.weight_birth[u] - 1 / self.weight_death[u]) * self.total_nodes[u]
        self.stability[u] = s
    
    def try_join(self, u, v, w):
        up = self.find(u)
        vp = self.find(v)
        if up != vp:
            self.union(up, vp, w)

    def children(self, u):
        if self.numchildren[u] > 0:
            v = self.firstchild[u]
            yield v
            while self.nextsibling[v] != v:
                v = self.nextsibling[v]
                yield v

    def preorder(self, u=None):
        if u is None:
            for u in self.roots:
                yield from self.preorder(u)
            return
        yield u
        for v in self.children(u):
            yield from self.preorder(v)

    def stoppable_preorder(self, f, u = None):
        if u is None:
            for u in self.roots:
                self.stoppable_preorder(f, u)
            return
        if not f(u):
            for v in self.children(u):
                self.stoppable_preorder(f, v)

    def leaves(self, u = None):
        if u is None:
            u = self.n * 2 - 2
        if self.numchildren[u] == 0:
            yield u
        else:
            for v in self.children(u):
                yield from self.leaves(v)
    
    def postorder(self, u=None):
        if u is None:
            for u in self.roots:
                yield from self.postorder(u)
            return
        for v in self.children(u):
            yield from self.postorder(v)
        yield u

    def to_newick(self, u=None):
        if u is None:
            u = self.n * 2 - 2
        if self.numchildren[u] == 0:
            return f"{u}:{self.stability[u]}" + ("X" if self.forget[u] else "")
        newicks = ", ".join([self.to_newick(c) for c in self.children(u)])
        return f"({newicks}){u}:{self.stability[u]}" + ("X" if self.forget[u] else "")

    def condense(self, min_points = 5):
        onechild = set()
        thres = min_points
        for u in self.preorder():
            if self.forget[u]:
                continue
            if self.numchildren[u] == 0:
                continue
            children = sorted([(self.total_nodes[c], c) for c in self.children(u)])
            tt_nodes0, tt_nodes1 = children[0][0], children[1][0]
            c0, c1 = children[0][1], children[1][1]
            if tt_nodes0 < thres and tt_nodes1 < thres:
                self.forget[c0] = self.forget[c1] = True
            elif tt_nodes0 < thres:
                self.forget[c0] = True
                onechild.add(u)
                self.stability[c1] += self.stability[u]
            elif tt_nodes1 < thres:
                self.forget[c1] = True
                onechild.add(u)
                self.stability[c0] += self.stability[u]
            else:
                pass
        self.onechild = onechild
    
    def extract_clusters(self, min_points = 5):
        for u in self.postorder():
            marked = self.marked
            if self.forget[u]:
                continue
            s_stability = sum(self.stability[c] for c in self.children(u) if not self.forget[c])
            if u in self.onechild:
                self.stability[u] = s_stability
                if any(c in marked for c in self.children(u)):
                    marked.add(u)
                    for c in self.children(u):
                        self.forget[c] = True
                continue
            if s_stability >= self.stability[u]:
                self.stability[u] = s_stability
            else:
                marked.add(u)
                for c in self.children(u):
                    self.forget[c] = True
    
        next_label = 0
        labels = np.full(self.n, -1)
        def traverse(u):
            nonlocal next_label
            nonlocal labels
            if self.forget[u]:
                return True
            if u in marked:
                assigned = list(self.leaves(u))
                if len(assigned) < min_points:
                    return True
                for v in assigned:
                    labels[v] = next_label
                next_label += 1
                return True
            return False
        self.stoppable_preorder(traverse)
        return labels


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