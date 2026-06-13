import numpy as np
from typing import Optional, Iterable, Callable, Any, Tuple, List


def vector_to_string(v: np.ndarray, realvalue: Optional[bool] = None, n_prec: int = 15) -> str:
    """Pretty string representation of a (row) vector.

    Parameters
    ----------
    v : np.ndarray
        The vector to print.
    realvalue : bool, optional
        If True, only print the real parts. If None, it is automatically
        determined based on whether imaginary parts are close to zero.
    n_prec : int, default 15
        Number of decimal places to print.

    Returns
    -------
    str
        Formatted string representation of the vector.
    """
    assert v.ndim == 1, f"{v.shape=}"
    if realvalue is None:
        realvalue = not np.any(np.abs(v.imag) > float(f"1e-{n_prec}"))
    real_format = f">{n_prec+2}.{n_prec}f"
    if realvalue:
        return " ".join([f"{np.real(el):{real_format}}" for el in v])
    imag_format = f">+.{n_prec}f"
    return " ".join([f"{np.real(el):{real_format}} {np.imag(el):{imag_format}}j" for el in v])


def matrix_to_string(m: np.ndarray, n_prec: int = 15, offset: int = 0) -> str:
    """Pretty string representation of a matrix.

    Parameters
    ----------
    m : np.ndarray
        The matrix to print.
    n_prec : int, default 15
        Number of decimal places to print.
    offset : int, default 0
        Indentation offset (number of spaces) for each line.

    Returns
    -------
    str
        Formatted string representation of the matrix.
    """
    realvalue = not np.any(np.abs(m.imag) > float(f"1e-{n_prec}"))
    print(" " * offset, end="")
    return ("\n" + " " * offset).join([vector_to_string(row, realvalue, n_prec) for row in m])


def matrix_print(m: np.ndarray, label: Optional[str] = None, n_prec: int = 15, **kwargs) -> None:
    """Pretty print the matrix m.

    Parameters
    ----------
    m : np.ndarray
        Matrix to print.
    label : str, optional
        Text to print above the matrix.
    n_prec : int, default 15
        Number of decimal places to print.
    **kwargs : dict
        Additional keyword arguments passed to the print function.
    """
    if label is not None:
        print(label)
    if len(m.shape) == 1:
        print(vector_to_string(m, n_prec=n_prec), **kwargs)
        return
    print(matrix_to_string(m, n_prec, 4 + (len(label) - len(label.lstrip())) if label is not None else 0), **kwargs)


def matrix_connectivity_print(m: np.ndarray, block_size: int = 1, label: Optional[str] = None) -> None:
    """Print the connections in a matrix.

    "O" signifies a (block-) diagonal term, "X" represents a (block-) offdiagonal term.

    Parameters
    ----------
    m : np.ndarray
        Matrix to print.
    block_size : int, default 1
        Size of blocks.
    label : str, optional
        Label to print above the matrix.
    """

    def get_char(el: float | complex, i: int, j: int) -> str:
        """Get the character representation for a matrix element.

        Parameters
        ----------
        el : float or complex
            The matrix element value to represent.
        i : int
            The block row index of the element.
        j : int
            The block column index of the element.

        Returns
        -------
        str
            "O" if diagonal, "X" if off-diagonal, or " " if zero.
        """
        if np.abs(el) <= np.finfo(float).eps:
            return " "
        if i == j:
            return "O"
        return "X"

    offset = 4 + (len(label) - len(label.lstrip())) if label is not None else 0
    if label is not None:
        print(label)
        print(" " * offset, end="")

    print(
        ("\n" + " " * offset).join(
            [
                " ".join([get_char(el, i // block_size, j // block_size) for j, el in enumerate(row)])
                for i, row in enumerate(m)
            ]
        )
    )


def partition(l: Iterable[Any], predicate: Callable[[Any], bool] = lambda a: bool(a)) -> Tuple[List[Any], List[Any]]:
    """Partition elements of an iterable into two lists based on a predicate.

    Parameters
    ----------
    l : Iterable
        The collection of elements to partition.
    predicate : callable, optional
        A function that takes an element and returns a boolean value.
        Defaults to `bool(a)`.

    Returns
    -------
    passed : list
        Elements for which the predicate returned True.
    failed : list
        Elements for which the predicate returned False.
    """
    passed = []
    failed = []
    for item in l:
        if predicate(item):
            passed.append(item)
        else:
            failed.append(item)
    return passed, failed

