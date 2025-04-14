import numpy as np

try:
    from collections.abc import Sequence
except:
    from collections import Sequence


class KrylovIterator:
    def __init__(self, vectors):
        self._vectors = vectors
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current >= self._vectors.shape[0]:
            raise StopIteration
        self._current += 1
        return self._vectors[self._current - 1].reshape((self._vectors.shape[1], 1))


class KrylovBasis:
    def __init__(self, N, dtype, v0=None, capacity=100):
        if v0 is None:
            self.vectors = np.empty((capacity, N), dtype=dtype, order="C")
            self.size = 0
        else:
            p = v0.shape[1]
            self.vectors = np.empty((2 * p, v0.shape[0]), dtype=v0.dtype)
            self.vectors[:p] = v0.T
            self.size = p
        self.capacity = self.vectors.shape[0]

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= self.size or key < -self.size:
                raise IndexError(f"index {key} is out of bounds for basis contaning {self.size} vectors.")
        elif isinstance(key, Sequence):
            for i in key or i < -self.size:
                if i >= self.size:
                    raise IndexError(f"index {i} is out of bounds for basis contaning {self.size} vectors.")

        elif isinstance(key, slice):
            start = key.start
            if start >= self.size or start < -self.size:
                raise IndexError(f"index {start} is out of bounds for basis contaning {self.size} vectors.")
            stop = key.stop
            if stop >= self.size or stop < -self.size:
                raise IndexError(f"index {stop} is out of bounds for basis contaning {self.size} vectors.")
        return self.vectors[key].T

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.vectors[: self.size].__iter__()

    def add(self, new_vectors):
        p = new_vectors.shape[1]
        if p + self.size > self.capacity:
            self.vectors = np.append(self.vectors, np.empty_like(self.vectors), axis=0)
            self.capacity = self.vectors.shape[0]
        self.vectors[self.size : self.size + p] = new_vectors.T
        self.size += p

    def calc_projection(self, v):
        return self.vectors[: self.size].T @ (np.conj(self.vectors[: self.size]) @ v)
