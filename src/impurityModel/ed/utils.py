from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np


def rotate_matrix(M, T):
    r"""
    Rotate the matrix, M, using the matrix T.
    Returns :math:`M' = T^{\dagger} M T`
    Parameters
    ==========
    M : NDArray - Matrix to rotate
    T : NDArray or dict - Rotation matrix to use, or dict of rotation matrices for blocks.
    Returns
    =======
    M' : NDArray - The rotated matrix
    """
    if isinstance(T, dict):
        from scipy.linalg import block_diag

        sorted_keys = sorted(T.keys())
        T_matrix = block_diag(*(T[k] for k in sorted_keys))
        return np.conj(T_matrix.T) @ M @ T_matrix
    return np.conj(T.T) @ M @ T


def _float_field_width(values: np.ndarray, n_prec: int, force_sign: bool = False) -> int:
    """Field width for printing ``values`` with ``n_prec`` decimals, columns aligned.

    The width reserves room for the integer digits of the largest-magnitude entry, the
    decimal point, ``n_prec`` decimals, and a leading sign column when any value is
    negative (or always, if ``force_sign`` — e.g. the ``+`` flag used for imaginary parts).
    """
    flat = np.real(np.asarray(values)).ravel()
    max_abs = round(float(np.max(np.abs(flat))), n_prec) if flat.size else 0.0
    int_digits = max(1, int(np.floor(np.log10(max_abs))) + 1) if max_abs >= 1 else 1
    sign = 1 if force_sign or (flat.size and np.any(flat < 0)) else 0
    return sign + int_digits + 1 + n_prec


def vector_to_string(
    v: np.ndarray,
    realvalue: Optional[bool] = None,
    n_prec: int = 15,
    real_width: Optional[int] = None,
    imag_width: Optional[int] = None,
) -> str:
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
    real_width, imag_width : int, optional
        Field widths for the real/imaginary parts. When None they are derived from ``v``;
        :func:`matrix_to_string` passes matrix-wide widths so every row lines up.

    Returns
    -------
    str
        Formatted string representation of the vector.
    """
    assert v.ndim == 1, f"{v.shape=}"
    if realvalue is None:
        realvalue = not np.any(np.abs(v.imag) > float(f"1e-{n_prec}"))
    if real_width is None:
        real_width = _float_field_width(v.real, n_prec)
    if realvalue:
        return " ".join(f"{np.real(el):>{real_width}.{n_prec}f}" for el in v)
    if imag_width is None:
        imag_width = _float_field_width(v.imag, n_prec, force_sign=True)
    return " ".join(f"{np.real(el):>{real_width}.{n_prec}f} {np.imag(el):>+{imag_width}.{n_prec}f}j" for el in v)


def matrix_to_string(m: np.ndarray, n_prec: int = 15, offset: int = 0) -> str:
    """Pretty string representation of a matrix.

    Columns are right-aligned to a common width derived from the whole matrix, so the
    entries form an aligned grid regardless of sign or magnitude.

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
    real_width = _float_field_width(m.real, n_prec)
    imag_width = None if realvalue else _float_field_width(m.imag, n_prec, force_sign=True)
    pad = " " * offset
    return "\n".join(
        pad + vector_to_string(row, realvalue, n_prec, real_width=real_width, imag_width=imag_width) for row in m
    )


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


REPORT_WIDTH = 80


def print_density_matrix_summary(m: np.ndarray, label: Optional[str] = None, offdiag_tol: float = 1e-4) -> None:
    """Compact summary of a (nearly diagonal) density matrix.

    Prints the diagonal occupations (wrapped, eight per row, with index ranges) and only
    the off-diagonal entries with ``|m_ij| > offdiag_tol`` (upper triangle, sparse
    ``(i, j) = value`` lines with a count summary), instead of the full matrix.

    Parameters
    ----------
    m : np.ndarray
        The (square) density matrix to summarize.
    label : str, optional
        Text to print above the summary.
    offdiag_tol : float, default 1e-4
        Magnitude threshold above which off-diagonal entries are listed.
    """
    if label is not None:
        print(label)
    diag = np.real(np.diag(m))
    per_row = 8
    print("  diagonal occupations:")
    for start in range(0, len(diag), per_row):
        chunk = diag[start : start + per_row]
        idx = f"[{start:>3d}-{min(start + per_row, len(diag)) - 1:>3d}]"
        print(f"    {idx}  " + "  ".join(f"{x:8.5f}" for x in chunk))
    iu, ju = np.triu_indices_from(m, k=1)
    big = np.abs(m[iu, ju]) > offdiag_tol
    n_big = int(np.count_nonzero(big))
    print(f"  off-diagonal entries with |value| > {offdiag_tol:g}: {n_big} of {iu.size}")
    order = np.argsort(np.abs(m[iu[big], ju[big]]))[::-1]
    for k in order:
        i, j = int(iu[big][k]), int(ju[big][k])
        val = m[i, j]
        val_str = f"{val.real: .5f}" if abs(val.imag) < 1e-12 else f"{val.real: .5f}{val.imag:+.5f}j"
        print(f"    ({i:>3d},{j:>3d}) = {val_str}")


def report_banner(title: str, file=None) -> None:
    """Print a top-level report section banner.

    Parameters
    ----------
    title : str
        Section title.
    file : file-like, optional
        Destination stream (defaults to stdout).
    """
    rule = "=" * REPORT_WIDTH
    print(f"\n{rule}\n  {title}\n{rule}", file=file)


def report_rule(title: str, file=None) -> None:
    """Print a light sub-section rule: ``-- title ------``.

    Parameters
    ----------
    title : str
        Sub-section title.
    file : file-like, optional
        Destination stream (defaults to stdout).
    """
    head = f"-- {title} "
    print(f"\n{head}{'-' * max(0, REPORT_WIDTH - len(head))}", file=file)


def partition(l: Iterable[Any], predicate: Callable[[Any], bool] = bool) -> Tuple[List[Any], List[Any]]:
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
