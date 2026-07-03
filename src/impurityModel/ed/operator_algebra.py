"""
Algebra on second-quantized operators in dict form, and conversion between
spin-orbital labels ``(l, s, m)`` and flat indices.

An operator dict maps a process ``((sorb_1, 'c'), (sorb_2, 'a'), ...)`` to its
amplitude, where each spin-orbital ``sorb`` is either an ``(l, s, m)`` tuple or a
flat index (see :func:`c2i` / :func:`i2c` for the mapping).
"""

import numpy as np


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
    Return an array of dicts of the form {(i,j):val ...} corresponding to the
    operators c^dagger_i c_j, stored in the opsArray
    Parameters
    ----------
    nbaths : dict
        l : nb
    opsArray : [dict]
        [{(((l, m, s, b), 'c'), ((l, m, s, b), 'a'))}, ...]
    """
    res = []
    for ops in opsArray:
        res.append(op2Dict(nbaths, ops))
    return res


def op2Dict(nBaths, ops):
    r"""
    returns a dict of the form {(i,j):val,...} correspoding to the opeator c^dagger_i c_j
    where i, j are obtained from c2i
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    ops: dict
        Multi configurational state.
    """
    d = {}
    for t, val in ops.items():
        # Only one particle terms
        if len(t) == 2:
            # Accept both ((l,m,s,b),'c/a') and (l,m,s,b)
            if len(t[0]) == 2:
                if t[0][1] == "c" and t[1][1] == "a":
                    d[((c2i(nBaths, t[0][0]), "c"), (c2i(nBaths, t[1][0]), "a"))] = val
                elif t[0][1] == "a" and t[1][1] == "c":
                    if t[0][0] == t[1][0]:
                        d[((c2i(nBaths, t[1][0]), "c"), (c2i(nBaths, t[0][0]), "a"))] = 1.0 - val
                    else:
                        d[((c2i(nBaths, t[1][0]), "c"), (c2i(nBaths, t[0][0]), "a"))] = -val
            else:
                d[((c2i(nBaths, t[0]), "c"), (c2i(nBaths, t[1]), "a"))] = val
    return d


def combineOp(nBaths, op1, op2):
    r"""
    Return a dict of the form {(i, j) : val, ...} corresponding to the
    operator op1*op2

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
    op : dict
        Operator dictionary {(i, j) : val}

    Returns
    -------
    m : numpy.ndarray
        Dense matrix representation of the operator
    """
    dsize = 0
    for l, nb in nBaths.items():
        dsize += nb + (2 * l + 1) * 2
    m = np.zeros((dsize, dsize), dtype=complex)
    for ((i, opi), (j, opj)), val in op.items():
        if opi == "c" and opj == "a":
            m[i, j] = val
        elif opj == "c" and opi == "a":
            if i == j:
                m[i, j] = 1 - val
            else:
                m[i, j] = -val
    return m


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
