"""Self-energy extraction from the impurity Green's function.

The static (Hartree-Fock) and dynamic self-energies, the hybridization function, the
correlated/bath splitting of the one-body Hamiltonian, and the physicality check on a
computed Green's function -- everything downstream of having ``G`` in hand. The
double-counting search lives next door in :mod:`double_counting`; the orchestration and CLI
in :mod:`selfenergy`, which re-exports these so existing ``selfenergy.get_sigma`` etc.
callers are unchanged.
"""

import itertools

import numpy as np

from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


class UnphysicalGreensFunctionError(Exception):
    """
    Excpetion signalling an unphysical Greens function, i.e. the imaginary part is positive for some frequencies.
    """


def check_greens_function(G):
    """Verify that the Green's function makes physical sense.

    Raises an exception if the diagonal elements of the imaginary part are positive.

    Parameters
    ----------
    G : np.ndarray
        The Green's function matrix.

    Raises
    ------
    UnphysicalGreensFunctionError
        If the diagonal term has a positive imaginary part.
    """
    if np.any(np.diagonal(G, axis1=1, axis2=2).imag > 0):
        raise UnphysicalGreensFunctionError("Diagonal term has positive imaginary part.")


def get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states):
    """Extract the correlation Hamiltonian, hybridization, and bath Hamiltonian.

    The matrix form of h0op can be written as:
      [  hcorr  V^+    ]
      [  V      hbath  ]

    Parameters
    ----------
    h0op : dict or ManyBodyOperator
        The non-interacting Hamiltonian operator. Any identity (constant) term is dropped:
        it shifts every eigenvalue equally and carries no hybridization information.
    impurity_orbitals : dict
        Dictionary of impurity orbitals.
    sum_bath_states : dict
        Dictionary of total bath states.

    Returns
    -------
    hcorr : np.ndarray
        Hamiltonian for the correlated impurity orbitals.
    v : np.ndarray
        Hopping from impurity to bath orbitals.
    v_dagger : np.ndarray
        Hopping from bath to impurity orbitals.
    h_bath : np.ndarray
        Hamiltonian for the non-interacting bath orbitals.
    """

    num_spin_orbitals = sum(impurity_orbitals[i] + sum_bath_states[i] for i in impurity_orbitals)
    n_corr = sum(ni for ni in impurity_orbitals.values())
    # Wrapping a plain dict normal-orders it first, so a caller-supplied anti-normal-ordered
    # term (c_i c^dag_j) is handled by the operator algebra rather than by a second, subtly
    # different convention here.
    if not isinstance(h0op, ManyBodyOperator):
        h0op = ManyBodyOperator(dict(h0op))
    h0Matrix = extract_tensors(h0op, n_orb=num_spin_orbitals, two_body=False)[0]
    hcorr = h0Matrix[0:n_corr, 0:n_corr]
    v_dagger = h0Matrix[0:n_corr, n_corr:]
    v = h0Matrix[n_corr:, 0:n_corr]
    h_bath = h0Matrix[n_corr:, n_corr:]
    return hcorr, v, v_dagger, h_bath


def hyb(ws, v, hbath, delta):
    """Calculate hybridization function from hopping parameters and bath energies.

    Δ(w) = V^dag [(w + i*delta)I - hbath]^-1 V

    Parameters
    ----------
    ws : np.ndarray
        Frequency mesh.
    v : np.ndarray
        Hopping matrix V.
    hbath : np.ndarray
        Bath Hamiltonian matrix.
    delta : float
        Smearing parameter.

    Returns
    -------
    np.ndarray
        The hybridization function.
    """
    return np.conj(v.T) @ np.linalg.solve(
        (ws + 1j * delta)[:, None, None] * np.identity(hbath.shape[0], dtype=complex)[None, :, :] - hbath[None, :, :],
        v[None, :, :],
    )


def get_sigma(
    omega_mesh,
    impurity_orbitals,
    nBaths,
    gs,
    h0op,
    delta,
    blocks,
    clustername="",
):
    """Calculate self-energy from interacting Greens function and local hamiltonian.

    Parameters
    ----------
    omega_mesh : np.ndarray
        Frequency mesh.
    impurity_orbitals : dict
        Dictionary of impurity orbitals.
    nBaths : dict
        Dictionary of total bath states.
    gs : list of np.ndarray
        List of block Green's function matrices.
    h0op : dict or ManyBodyOperator
        The non-interacting Hamiltonian operator.
    delta : float
        Smearing parameter.
    blocks : list of list of int
        List of blocks.
    clustername : str, optional
        Label for the cluster.

    Returns
    -------
    list of np.ndarray
        The self-energy matrices for each block.
    """
    hcorr, v_full, _, h_bath = get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)

    res = []
    for block, g in zip(blocks, gs):
        block_ix = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_ix] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        res.append(g0_inv - np.linalg.inv(g))

    return res


def get_Sigma_static(U4, rho):
    r"""Calculate the static (Hartree-Fock) self-energy.

    ``U4`` is in RSPt's physicists'-notation convention,
    :math:`U4[i,j,k,l] = \langle ij|V|kl \rangle`, i.e. the operator
    :math:`\frac{1}{2}\sum U4[i,j,k,l] c^\dagger_i c^\dagger_j c_l c_k`
    (see :func:`impurityModel.ed.atomic_physics.getUop_from_rspt_u4`).

    Parameters
    ----------
    U4 : np.ndarray
        Coulomb interaction tensor (RSPt convention).
    rho : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        The static self-energy.
    """
    sigma_static = np.zeros_like(rho)
    for i, j in itertools.product(range(rho.shape[0]), range(rho.shape[1])):
        sigma_static += (U4[j, :, i, :] - U4[j, :, :, i]) * rho[i, j]

    return sigma_static


def get_Sigma_moments(M, hcorr, v, hbath):
    r"""High-frequency self-energy moments from the interacting Green's-function moments.

    Given the spectral moments ``M[n]`` of the impurity Green's function
    (:math:`G(z) = \sum_n M_n / z^{n+1}`, ``M[0] = I``; see
    :func:`impurityModel.ed.greens_function.get_greens_function_moments`) and the
    correlated/hybridization/bath blocks of the non-interacting Hamiltonian
    (:func:`get_hcorr_v_hbath`), return the coefficients of

    .. math::

        \Sigma(z) = \Sigma_\infty + \Sigma_1 / z + \Sigma_2 / z^2 + \dots

    The non-interacting inverse Green's function is
    :math:`G_0^{-1}(z) = z - h_{corr} - \Delta(z)` with the hybridization moments
    :math:`\Delta(z) = V^\dagger V / z + V^\dagger h_{bath} V / z^2 + \dots`, so with
    :math:`\Sigma = G_0^{-1} - G^{-1}`:

    .. math::

        \Sigma_\infty &= M_1 - h_{corr}, \\
        \Sigma_1 &= M_2 - M_1^2 - V^\dagger V, \\
        \Sigma_2 &= M_3 - M_1 M_2 - M_2 M_1 + M_1^3 - V^\dagger h_{bath} V.

    :math:`\Sigma_\infty` equals the static (Hartree-Fock) self-energy
    :func:`get_Sigma_static` and is returned as a consistency handle.

    Parameters
    ----------
    M : np.ndarray
        ``(>=4, n_corr, n_corr)`` Green's-function moments ``M[0..3]`` (solver basis).
    hcorr : np.ndarray
        ``(n_corr, n_corr)`` correlated one-body block.
    v : np.ndarray
        ``(n_bath, n_corr)`` impurity-to-bath hopping ``V``.
    hbath : np.ndarray
        ``(n_bath, n_bath)`` bath Hamiltonian.

    Returns
    -------
    sigma_inf : np.ndarray
        The static moment :math:`\Sigma_\infty` (``= M_1 - h_{corr}``).
    sigma_1 : np.ndarray
        The first dynamic moment :math:`\Sigma_1`.
    sigma_2 : np.ndarray
        The second dynamic moment :math:`\Sigma_2`.
    """
    m1, m2, m3 = M[1], M[2], M[3]
    vtv = v.conj().T @ v
    vt_hbath_v = v.conj().T @ hbath @ v
    sigma_inf = m1 - hcorr
    sigma_1 = m2 - m1 @ m1 - vtv
    sigma_2 = m3 - m1 @ m2 - m2 @ m1 + m1 @ m1 @ m1 - vt_hbath_v
    return sigma_inf, sigma_1, sigma_2
