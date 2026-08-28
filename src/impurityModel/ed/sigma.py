"""Self-energy extraction from the impurity Green's function.

The static (Hartree-Fock) and dynamic self-energies, the hybridization function, the
correlated/bath splitting of the one-body Hamiltonian, and the physicality check on a
computed Green's function -- everything downstream of having ``G`` in hand. The
double-counting criteria live next door in :mod:`dc_criteria`/:mod:`dc_static`; the
orchestration and CLI in :mod:`selfenergy`, which re-exports these so existing
``selfenergy.get_sigma`` etc. callers are unchanged.
"""

import itertools

import numpy as np

from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


class UnphysicalGreensFunctionError(Exception):
    """
    Excpetion signalling an unphysical Greens function, i.e. the imaginary part is positive for some frequencies.
    """


def _localize_causality_violation(diag_imag, omega_mesh=None, g0_inv_diag=None, ginv_diag=None):
    r"""Localize a positive-imaginary-part (non-causal) violation on a diagonal.

    ``diag_imag`` is the real ``(n_omega, n_orb)`` array of the diagonal's imaginary part
    (``np.diagonal(G, axis1=1, axis2=2).imag``). Returns ``None`` if every entry is causal
    (``<= 0``), else a dict describing the violation:

    * ``n_viol``/``n_total``/``frac`` -- violating (frequency, orbital) point count.
    * ``windows`` -- contiguous violating frequency windows (index pairs, or ``omega_mesh``
      values if given), unioned over orbitals.
    * ``worst_val``, ``worst_freq_index``, ``worst_orbital`` -- the single worst point.
    * ``worst_ratio`` -- the causality-relevant normalizer: **per diagonal element**, compare
      :math:`\max_\omega \mathrm{Im}(G_{ii})` against :math:`\max_\omega |\mathrm{Im}(G_{ii})|`
      for that same ``i``, and take the worst ratio over ``i``. Not a whole-block
      ``max|Im G|`` -- that lets a wide, high-weight orbital mask a violation on a nearly-empty
      one, and the two denominators diverge with block width.
    * ``g0_inv_at_worst``/``ginv_at_worst`` -- (only when ``g0_inv_diag``/``ginv_diag`` are
      given) the two terms of :math:`\Sigma = G_0^{-1} - G^{-1}` at the worst point, so the
      mechanism (a narrower interacting ``G`` than the bare hybridized ``G_0``) is visible
      without reconstructing it from saved files afterwards.
    """
    viol = diag_imag > 0
    if not np.any(viol):
        return None
    n_omega, _n_orb = diag_imag.shape
    iw, io = np.unravel_index(np.argmax(diag_imag), diag_imag.shape)

    max_im = np.max(diag_imag, axis=0)  # (n_orb,) -- causality-violating direction per orbital
    max_abs_im = np.max(np.abs(diag_imag), axis=0)
    has_weight = max_abs_im > 0
    ratio_per_orb = np.where(has_weight, max_im / np.where(has_weight, max_abs_im, 1.0), -np.inf)
    worst_orb = int(np.argmax(ratio_per_orb))

    any_viol_w = np.any(viol, axis=1)
    windows = []
    start = None
    for i, v in enumerate(any_viol_w):
        if v and start is None:
            start = i
        elif not v and start is not None:
            windows.append((start, i - 1))
            start = None
    if start is not None:
        windows.append((start, n_omega - 1))
    if omega_mesh is not None:
        # Kept in the mesh's own dtype (real for w, complex for iw -- Matsubara) rather than
        # cast to float, which would silently discard the imaginary part of a Matsubara mesh.
        windows = [(omega_mesh[a], omega_mesh[b]) for a, b in windows]

    report = {
        "n_viol": int(np.sum(viol)),
        "n_total": int(diag_imag.size),
        "frac": float(np.sum(viol)) / diag_imag.size,
        "worst_val": float(diag_imag[iw, io]),
        "worst_freq_index": int(iw),
        "worst_orbital": int(io),
        "worst_orbital_ratio": worst_orb,
        "worst_ratio": float(ratio_per_orb[worst_orb]),
        "windows": windows,
    }
    if g0_inv_diag is not None and ginv_diag is not None:
        report["g0_inv_at_worst"] = complex(g0_inv_diag[iw, io])
        report["ginv_at_worst"] = complex(ginv_diag[iw, io])
    return report


def _fmt_freq(v):
    """Format one mesh point -- real (omega) or complex (i*omega_n, Matsubara) -- for a message.

    Always goes through ``complex()`` first: formatting a ``numpy.complex128`` with a ``.4g``
    spec raises ``TypeError`` even when its imaginary part is exactly zero, which would replace
    a real causality-failure message with an unrelated formatting crash.
    """
    v = complex(v)
    return f"{v.real:.4g}" if v.imag == 0 else f"{v.real:.4g}{v.imag:+.4g}j"


def _format_causality_report(report, label):
    """Render a :func:`_localize_causality_violation` report as a human-readable message."""
    lines = [
        f"{report['n_viol']} of {report['n_total']} ({label} diagonal, frequency x orbital) "
        f"points violate causality ({100 * report['frac']:.2f}%)",
        f"  worst Im = {report['worst_val']:.3e} at orbital {report['worst_orbital']}, "
        f"frequency index {report['worst_freq_index']}",
        f"  worst per-orbital ratio max(Im)/max(|Im|) = {report['worst_ratio']:.3e} "
        f"(orbital {report['worst_orbital_ratio']})",
    ]
    if report["windows"]:
        windows_str = ", ".join(f"[{_fmt_freq(a)}, {_fmt_freq(b)}]" for a, b in report["windows"])
        lines.append(f"  violating window(s): {windows_str}")
    if "g0_inv_at_worst" in report:
        g0i, gi = report["g0_inv_at_worst"], report["ginv_at_worst"]
        lines.append(
            f"  at the worst point: Im(G0^-1) = {g0i.imag:.3e}, Im(G^-1) = {gi.imag:.3e} "
            f"(Im Sigma = Im(G0^-1) - Im(G^-1) = {(g0i - gi).imag:.3e})"
        )
    return "\n".join(lines)


def check_greens_function(G, tol: float = 0.0, omega_mesh=None, label="G", g0_inv=None, ginv=None):
    r"""Verify that the Green's function (or self-energy) makes physical sense.

    Causality (retarded convention) requires every diagonal element's imaginary part to stay
    ``<= 0``. Below ``tol`` -- a **relative**, per-diagonal-element tolerance, see
    :func:`_localize_causality_violation`'s ``worst_ratio`` -- a violation is reported (printed)
    as a warning and the call returns normally; above it, raises with the same report in the
    message. ``tol=0`` (the default, used for the Green's function itself) preserves the
    original always-raise-on-any-violation behaviour.

    Parameters
    ----------
    G : np.ndarray
        The Green's function (or self-energy) matrix, shape ``(n_omega, n_orb, n_orb)``.
    tol : float
        Relative tolerance (see above). ``0`` raises on any violation, however small.
    omega_mesh : np.ndarray, optional
        Frequency mesh matching ``G``'s first axis, used only to report violating windows in
        physical units rather than mesh indices.
    label : str
        Name used in the message (e.g. ``"G"`` or ``"Sigma"``).
    g0_inv, ginv : np.ndarray, optional
        Same shape as ``G``. When both are given (the self-energy call site, which already has
        them from :func:`get_sigma`), the report includes the :math:`G_0^{-1}` vs. :math:`G^{-1}`
        breakdown at the worst point.

    Raises
    ------
    UnphysicalGreensFunctionError
        If the worst-case relative violation exceeds ``tol``.
    """
    diag_imag = np.diagonal(G, axis1=1, axis2=2).imag
    g0_inv_diag = np.diagonal(g0_inv, axis1=1, axis2=2) if g0_inv is not None else None
    ginv_diag = np.diagonal(ginv, axis1=1, axis2=2) if ginv is not None else None
    report = _localize_causality_violation(diag_imag, omega_mesh, g0_inv_diag, ginv_diag)
    if report is None:
        return
    text = _format_causality_report(report, label)
    if report["worst_ratio"] > tol:
        raise UnphysicalGreensFunctionError(f"Diagonal term has positive imaginary part.\n{text}")
    print(f"warning: {label} causality violated within tolerance (tol={tol:.1e}):\n{text}", flush=True)


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
    return_components=False,
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
    return_components : bool, optional
        If True, also return the two terms of :math:`\\Sigma = G_0^{-1} - G^{-1}` per block --
        lets a causality check (:func:`check_greens_function`) report which of the two moved,
        rather than only the difference.

    Returns
    -------
    list of np.ndarray
        The self-energy matrices for each block.
    list of tuple[np.ndarray, np.ndarray], optional
        Only when ``return_components=True``: ``(g0_inv, ginv)`` per block, same shape as the
        corresponding self-energy matrix.
    """
    hcorr, v_full, _, h_bath = get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)

    res = []
    components = []
    for block, g in zip(blocks, gs):
        block_ix = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_ix] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        ginv = np.linalg.inv(g)
        res.append(g0_inv - ginv)
        if return_components:
            components.append((g0_inv, ginv))

    if return_components:
        return res, components
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
