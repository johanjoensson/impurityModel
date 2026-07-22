"""
Algebra on second-quantized operators in dict form, and conversion between
spin-orbital labels ``(l, s, m)`` and flat indices.

An operator dict maps a process ``((sorb_1, 'c'), (sorb_2, 'a'), ...)`` to its
amplitude, where each spin-orbital ``sorb`` is either an ``(l, s, m)`` tuple or a
flat index (see :func:`c2i` / :func:`i2c` for the mapping).
"""

import numpy as np

from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


def daggerOp(op):
    """
    return op^dagger
    """
    opDagger = {}
    for process, value in op.items():
        processNew = []
        for e in process[::-1]:
            if e[1] == "a":
                processNew.append((e[0], "c"))
            elif e[1] == "c":
                processNew.append((e[0], "a"))
            else:
                raise Exception("Operator type unknown: {}".format(e[1]))
        processNew = tuple(processNew)
        opDagger[processNew] = value.conjugate()
    return opDagger


def assert_hermitian(op: dict[tuple, int | float | complex]) -> None:
    """Assert that the operator is Hermitian (equal to its adjoint).

    Parameters
    ----------
    op : dict
        The operator representing a mapping from excitation tuples to amplitudes.
    """
    assert daggerOp(op) == op


def addOps(ops):
    """
    Return one operator, represented as a dictonary.

    Parameters
    ----------
    ops : list
        Operators

    Returns
    -------
    opSum : dict

    """
    opSum = {}
    for op in ops:
        for sOp, value in op.items():
            opSum[sOp] = opSum.get(sOp, 0) + value
            # if np.abs(value) > 1e-12:
            #     if sOp in opSum:
            #         opSum[sOp] += value
            #     else:
            #         opSum[sOp] = value
    return opSum


def c2i(nBaths, spinOrb):
    """
    Return an index, representing a spin-orbital or a bath state.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    Returns
    -------
    i : int
        An index denoting a spin-orbital or a bath state.

    """
    # Counting index and return variable.
    i = 0
    # Check if spinOrb is an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            for sp in range(2):
                for mp in range(-lp, lp + 1):
                    if (lp, sp, mp) == spinOrb:
                        return i
                    i += 1
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                for sp in range(2):
                    for mp in range(-lp_int, lp_int + 1):
                        if (lp_int, sp, mp) == spinOrb:
                            return i
                        i += 1
    # If reach this point it means spinOrb is a bath state.
    # Need to figure out which one index is has.
    for lp, nBath in nBaths.items():
        for b in range(nBath):
            if (lp, b) == spinOrb:
                return i
            i += 1
    raise Exception("Can not find index corresponding to spin-orbital state")


def i2c(nBaths, i):
    """
    Return an coordinate tuple, representing a spin-orbital.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    i : int
        An index denoting a spin-orbital or a bath state.

    Returns
    -------
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    """
    # Counting index.
    k = 0
    # Check if index "i" belong to an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            # Check if index "i" belong to impurity spin-orbital having lp.
            if i - k < 2 * (2 * lp + 1):
                for sp in range(2):
                    for mp in range(-lp, lp + 1):
                        if k == i:
                            return (lp, sp, mp)
                        k += 1
            k += 2 * (2 * lp + 1)
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                # Check if index "i" belong to impurity spin-orbital having lp_int.
                if i - k < 2 * (2 * lp_int + 1):
                    for sp in range(2):
                        for mp in range(-lp_int, lp_int + 1):
                            if k == i:
                                return (lp_int, sp, mp)
                            k += 1
                k += 2 * (2 * lp_int + 1)
    # If reach this point it means index "i" belong to a bath state.
    # Need to figure out which one.
    for lp, nBath in nBaths.items():
        b = i - k
        # Check if bath state belong to bath states having lp.
        if b < nBath:
            # The index "b" will have a value between 0 and nBath-1
            return (lp, b)
        k += nBath
    raise Exception("Can not find spin-orbital state corresponding to index.")


def arrayOp2Dict(nbaths, opsArray):
    r"""
    Convert several label-keyed one-body projectors to index-keyed dicts.

    See :func:`op2Dict` for the accepted input forms.

    Parameters
    ----------
    nbaths : dict
        l : nb
    opsArray : iterable of dict
        ``[{((l, s, m), 'c'), ((l, s, m), 'a')): val, ...}, ...]``
    """
    res = []
    for ops in opsArray:
        res.append(op2Dict(nbaths, ops))
    return res


def _is_ladder_factor(element):
    """Whether ``element`` is a ``(spin_orbital_label, 'c'|'a')`` factor.

    Dispatching on the action string rather than on ``len(element)`` matters: a bath
    label is itself the 2-tuple ``(l, b)``, so a length test cannot tell
    ``((l, b), 'c')`` from a bare ``(l, b)``.
    """
    return len(element) == 2 and isinstance(element[1], str) and element[1] in ("c", "a")


def _term_to_index_key(nBaths, term):
    """One label-keyed process tuple -> the same process with integer orbital indices.

    The written operator order is preserved; normal ordering is left to
    :class:`ManyBodyOperator`.
    """
    if len(term) != 2:
        raise ValueError(
            f"Projector term {term} has {len(term)} factors; only one-body terms "
            "(c^dagger_i c_j) can be projected onto a transition operator."
        )
    first, second = term
    labelled = [_is_ladder_factor(element) for element in term]
    if all(labelled):
        return tuple((c2i(nBaths, orbital), action) for orbital, action in term)
    if any(labelled):
        raise ValueError(
            f"Projector term {term} mixes the labelled form ((l, s, m), 'c') with the bare "
            "form (l, s, m); write both factors the same way."
        )
    # Bare pair ((l, s, m), (l', s', m')), which means c^dagger_i c_j.
    return ((c2i(nBaths, first), "c"), (c2i(nBaths, second), "a"))


def op2Dict(nBaths, ops):
    r"""
    Convert a label-keyed one-body projector to an index-keyed operator dict.

    The XAS/RIXS ``*_projectors`` options take projectors written in spin-orbital
    labels; this maps those labels to the flat indices (:func:`c2i`) the rest of the
    package uses, so that :func:`combineOp` can project a transition operator onto an
    irrep and :func:`spectra.simulate_spectra` can report that irrep's contribution
    separately.

    Both spellings of a term are accepted:

    - ``(((l, s, m), 'c'), ((l', s', m'), 'a')): val`` — explicit ladder operators, in
      either order;
    - ``((l, s, m), (l', s', m')): val`` — a bare pair, meaning
      :math:`c^\dagger_i c_j`.

    Terms written anti-normal-ordered are normal-ordered by
    :class:`ManyBodyOperator`, using :math:`c_i c^\dagger_j = \delta_{ij} - c^\dagger_j c_i`,
    and terms that collide after reordering are summed.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    ops : dict
        The projector, ``{process: amplitude}``.

    Returns
    -------
    dict
        ``{((i, 'c'), (j, 'a')): val, ...}`` in canonical normal order.

    Raises
    ------
    ValueError
        If a term is not one-body, mixes the two spellings, or leaves an identity
        (constant) part after normal ordering -- the projected transition operator is
        built by a single-particle matrix product (:func:`combineOp`), which has no way
        to represent a multiple of the identity.
    """
    # Sum term by term through the operator algebra: it normal-orders each term and
    # accumulates ones that collide, which a plain dict assignment would silently drop.
    op = ManyBodyOperator()
    for term, val in ops.items():
        op += ManyBodyOperator({_term_to_index_key(nBaths, term): val})

    if op.constant != 0:
        raise ValueError(
            f"Projector normal-orders to an identity part {op.constant:g} (from an "
            "anti-normal-ordered diagonal term such as c_i c^dagger_i = 1 - n_i). A "
            "projector must be a pure one-body operator here, because it is applied to "
            "the transition operator as a single-particle matrix product. Write the "
            "complement explicitly instead, e.g. sum_{j != i} c^dagger_j c_j over the "
            "shell rather than c_i c^dagger_i."
        )
    return op.to_dict()


def combineOp(nBaths, op1, op2):
    r"""
    Return a dict of the form {(i, j) : val, ...} corresponding to the
    operator op1*op2

    This is the *single-particle matrix* product, i.e. the one-body operator
    :math:`\sum_{ij} (M_1 M_2)_{ij} c^\dagger_i c_j`. It is deliberately NOT
    ``ManyBodyOperator.__mul__``, which composes the many-body operators and so also
    produces the two-body terms of :math:`\hat O_1 \hat O_2`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    op1 : dict
        Operator dictionary {(i, j) : val}
    op2 : dict
        Operator dictionary {(i, j) : val}

    Returns
    -------
    newOp : dict
        Combined operator dictionary
    """
    mOp1 = iOpToMatrix(nBaths, op1)
    mOp2 = iOpToMatrix(nBaths, op2)

    newOp = np.matmul(mOp1, mOp2)

    return matrixToIOp(newOp)


def iOpToMatrix(nBaths, op):
    r"""
    Return the matrix representation of op

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    op : dict or ManyBodyOperator
        Operator dictionary {(i, j) : val}

    Returns
    -------
    m : numpy.ndarray
        Dense matrix representation of the operator
    """
    dsize = 0
    for l, nb in nBaths.items():
        dsize += nb + (2 * l + 1) * 2
    # Wrapping a plain dict normal-orders it first, so an anti-normal-ordered term
    # (c_i c^dag_j) is resolved by the operator algebra rather than by a second convention.
    if not isinstance(op, ManyBodyOperator):
        op = ManyBodyOperator(dict(op))
    return extract_tensors(op, n_orb=dsize, two_body=False)[0]


def matrixToIOp(mat):
    r"""
    Return a dict containing the non-zero elements of the matrix mat

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix representation of the operator

    Returns
    -------
    res : dict
        Operator dictionary {(i, j) : val}
    """
    rows, columns = mat.shape
    res = {}
    for i in range(rows):
        for j in range(columns):
            if abs(mat[i, j]) > 0:
                res[((i, "c"), (j, "a"))] = mat[i, j]
    return res
