import numpy as np
from typing import Optional


def vector_to_string(v: np.ndarray, n_prec: int = 15):
    """
    Pretty string representation of a (row) vector
    Arguments:
    =========
    v: np.ndarray - vector to print
    n_prec: int - number of decimal places to print (default=15)
    """
    real_format = f" .{n_prec}f"
    imag_format = f">+.{n_prec}f"
    if np.any(np.abs(v.imag)) > float(f"1e{-n_prec}"):
        return " ".join([f"{np.real(el):{real_format}} {np.imag(el):{imag_format}}j" for el in v])
    return " ".join([f"{np.real(el):{real_format}}" for el in v])


def matrix_to_string(m: np.ndarray, n_prec: int = 15):
    """
    Pretty string representation of matrix
    Arguments:
    =========
    m: np.ndarray - matrix to print
    n_prec: int - number of decimal places to print (default=15)
    """
    return "\n".join([vector_to_string(row, n_prec) for row in m])


def matrix_print(m: np.ndarray, label: Optional[str] = None, n_prec=15):
    """
    Pretty print the matrix m
    Arguments
    =========
    m: numpy ndarray - Matrix to print
    label: Optional[str] - Text to print above the matrix (default=None)
    n_prec: int - number of decimal places to print (default=15)
    """
    if label is not None:
        print(label)
    if len(m.shape) == 1:
        print(vector_to_string(m, n_prec))
        return
    print(matrix_to_string(m, n_prec))


def matrix_connectivity_print(m: np.ndarray, block_size: int = 1, label: Optional[str] = None):
    """
    Print the connections in matrix. "O" signifies a (block-) diagonal term, "X" represents an (block-) offdiagonal term
    Arguments
    =========
    m: numpy.ndarray - Matrix to print
    block_size: int - size of blocks (default=1)
    label: Optional[str] - label to print above the matrix (default=None)
    """

    def get_char(el: float | complex, i: int, j: int):
        if np.abs(el) <= np.finfo(float).eps:
            return " "
        if i == j:
            return "O"
        return "X"

    if label is not None:
        print(label)

    print(
        "\n".join(
            [
                " ".join([get_char(el, i // block_size, j // block_size) for j, el in enumerate(row)])
                for i, row in enumerate(m)
            ]
        )
    )
