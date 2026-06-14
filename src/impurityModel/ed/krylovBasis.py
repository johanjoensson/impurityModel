import numpy as np

try:
    from collections.abc import Sequence
except:
    from collections import Sequence


class KrylovIterator:
    """
    Iterator for traversing Krylov basis vectors.

    Yields each vector in the basis reshaped as a column vector of shape (N, 1).
    """

    def __init__(self, vectors):
        """
        Initialize the Krylov iterator.

        Parameters
        ----------
        vectors : np.ndarray
            Array of shape (capacity, N) containing basis vectors.
        """
        self._vectors = vectors
        self._current = 0

    def __iter__(self):
        """
        Return the iterator instance itself.
        """
        return self

    def __next__(self):
        """
        Return the next vector in the basis as a column vector.

        Returns
        -------
        vector : np.ndarray
            Column vector of shape (N, 1).
        """
        if self._current >= self._vectors.shape[0]:
            raise StopIteration
        self._current += 1
        return self._vectors[self._current - 1].reshape((self._vectors.shape[1], 1))


class KrylovBasis:
    """
    Container for storing and manipulating Krylov basis vectors.

    Provides mechanisms to store vectors, add new ones with automatic capacity expansion,
    retrieve vectors using index/slice notation, and calculate projections.

    Examples
    --------
    >>> basis = KrylovBasis(N=3, dtype=float, capacity=2)
    >>> basis.add(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T)
    >>> len(basis)
    2
    >>> basis[0]
    array([[1.],
           [0.],
           [0.]])
    """

    def __init__(self, N, dtype, v0=None, capacity=100):
        """
        Initialize the Krylov basis.

        Parameters
        ----------
        N : int
            Dimension of each vector.
        dtype : data-type
            The data type of the vectors (e.g., float, complex).
        v0 : np.ndarray, optional
            Initial set of column vectors of shape (N, p).
        capacity : int, optional
            Initial capacity of the vector storage. Default is 100.
        """
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
        """
        Retrieve a vector or slice of vectors from the basis.

        Parameters
        ----------
        key : int, Sequence, or slice
            Index/indices of the vectors to retrieve.

        Returns
        -------
        vectors : np.ndarray
            Retrieved vector(s) as column vector(s) of shape (N, k).
        """
        if isinstance(key, int):
            if key >= self.size or key < -self.size:
                raise IndexError(f"index {key} is out of bounds for basis contaning {self.size} vectors.")
        elif isinstance(key, Sequence):
            for i in key:
                if i < -self.size or i >= self.size:
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
        """
        Return the number of vectors currently in the basis.
        """
        return self.size

    def __iter__(self):
        """
        Return an iterator over the underlying vectors array.
        """
        return self.vectors[: self.size].__iter__()

    def add(self, new_vectors):
        """
        Add new vectors to the basis. Doubles capacity if needed.

        Parameters
        ----------
        new_vectors : np.ndarray
            Column vectors to add, shape (N, p) where N is vector dimension and p is number of vectors.
        """
        p = new_vectors.shape[1]
        if p + self.size > self.capacity:
            self.vectors = np.append(self.vectors, np.empty_like(self.vectors), axis=0)
            self.capacity = self.vectors.shape[0]
        self.vectors[self.size : self.size + p] = new_vectors.T
        self.size += p

    def calc_projection(self, v):
        """
        Calculate the projection of a vector `v` onto the basis.

        Parameters
        ----------
        v : np.ndarray
            The target vector to project, shape (N, ...).

        Returns
        -------
        proj : np.ndarray
            The projected vector, shape (N, ...).
        """
        return self.vectors[: self.size].T @ (np.conj(self.vectors[: self.size]) @ v)
