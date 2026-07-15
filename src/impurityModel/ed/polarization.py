r"""
Polarization vectors and contractions of spectroscopy tensors.

The solvers (:mod:`impurityModel.ed.spectra`) compute polarization-*tensor* quantities:
the XAS spectral tensor :math:`\chi_{\alpha\beta}(\omega)` and the rank-4 RIXS
Kramers-Heisenberg tensor :math:`C_{\alpha\beta\alpha'\beta'}(\omega_\text{in},
\omega_\text{loss})` over the Cartesian dipole components. This module holds the
post-processing stage: contracting those tensors with concrete polarization vectors and
deriving plot-ready quantities (intensities, isotropic averages, dichroism).

It is a numpy-only leaf module -- no MPI, no solver imports -- so plot scripts can use it
without pulling in the compute stack.
"""

import numpy as np

# Named polarization vectors (propagation along z for the circular ones):
# cl/cr are the left/right circularly polarized unit vectors (1, +-i, 0)/sqrt(2).
_NAMED_POLARIZATIONS = {
    "x": np.array([1.0, 0.0, 0.0], dtype=complex),
    "y": np.array([0.0, 1.0, 0.0], dtype=complex),
    "z": np.array([0.0, 0.0, 1.0], dtype=complex),
    "cl": np.array([1.0, 1.0j, 0.0], dtype=complex) / np.sqrt(2),
    "cr": np.array([1.0, -1.0j, 0.0], dtype=complex) / np.sqrt(2),
}


def polarization_vector(spec):
    """
    Resolve a polarization specification to a complex vector.

    Parameters
    ----------
    spec : str or array_like
        Either a named polarization -- ``"x"``, ``"y"``, ``"z"``, or the circular
        ``"cl"``/``"cr"`` (left/right, propagation along z: ``(1, +-i, 0)/sqrt(2)``) --
        a comma-separated component string like ``"0,0.707,0.707j"`` (each component is
        parsed by :class:`complex`), or a sequence of numbers. Component specifications
        are used as given (not normalized).

    Returns
    -------
    ndarray
        The complex polarization vector.
    """
    if isinstance(spec, str):
        name = spec.strip().lower()
        if name in _NAMED_POLARIZATIONS:
            return _NAMED_POLARIZATIONS[name].copy()
        try:
            return np.array([complex(tok.strip().replace("i", "j")) for tok in spec.split(",")], dtype=complex)
        except ValueError:
            raise ValueError(
                f"Malformed polarization {spec!r} (expected one of {sorted(_NAMED_POLARIZATIONS)} "
                'or comma-separated components like "0,0.707,0.707j")'
            ) from None
    return np.asarray(spec, dtype=complex)


def contract_spectra_tensor(chi, polarizations):
    r"""
    Contract the XAS/NIXS-style spectral tensor with polarization vectors.

    .. math:: G_\varepsilon(\omega) = \sum_{\alpha\beta} \varepsilon_\alpha^*
              \chi_{\alpha\beta}(\omega) \varepsilon_\beta .

    Parameters
    ----------
    chi : ndarray
        Spectral tensor of shape ``(n_w, m, m)`` over the ``m`` Cartesian transition
        components (from :func:`spectra.calc_spectra_tensor`).
    polarizations : sequence
        Polarization specifications (anything :func:`polarization_vector` accepts),
        each of length ``m``.

    Returns
    -------
    ndarray
        Complex spectra of shape ``(n_w, n_pol)``.
    """
    eps = np.array([polarization_vector(p) for p in polarizations])  # (n_pol, m)
    return np.einsum("pa,wab,pb->wp", eps.conj(), chi, eps, optimize=True)


def contract_rixs_tensor(C, polarizations_in, polarizations_out):
    r"""
    Contract the rank-4 RIXS Kramers-Heisenberg tensor with in/out polarization vectors.

    ``C[a, b, a', b', wIn, wLoss]`` is the resolvent matrix over the flattened out-seed
    block :math:`s_{\alpha\beta} = T^\text{out}_\beta \psi^{(2)}_\alpha` (from
    :func:`spectra.calc_tensor_map`). The out-component operators are *daggered*
    dipole operators, :math:`T_\text{out}(\varepsilon) = \sum_\beta \varepsilon_\beta^*
    T^\text{out}_\beta`, so the out polarization enters unconjugated on the R2-ket seed
    index (:math:`\beta`) and conjugated on the bra index (:math:`\beta'`); the in
    operators carry no dagger (conjugated on the bra index :math:`\alpha`):

    .. math:: G_{pq}(\omega_\text{in}, \omega_\text{loss}) = \sum_{\alpha\beta\alpha'\beta'}
              \varepsilon^{\text{in}*}_{p\alpha} \varepsilon^\text{out}_{q\beta}
              \varepsilon^\text{in}_{p\alpha'} \varepsilon^{\text{out}*}_{q\beta'}
              C_{\alpha\beta\alpha'\beta'} .

    Parameters
    ----------
    C : ndarray
        RIXS tensor of shape ``(n_in, n_out, n_in, n_out, n_wIn, n_wLoss)``.
    polarizations_in, polarizations_out : sequence
        In/out polarization specifications (anything :func:`polarization_vector`
        accepts), of length ``n_in`` / ``n_out`` each.

    Returns
    -------
    ndarray
        Complex RIXS maps of shape ``(n_pin, n_pout, n_wIn, n_wLoss)``.
    """
    eps_in = np.array([polarization_vector(p) for p in polarizations_in])  # (n_pin, n_in)
    eps_out = np.array([polarization_vector(p) for p in polarizations_out])  # (n_pout, n_out)
    return np.einsum(
        "pa,qb,pc,qd,abcdwl->pqwl",
        eps_in.conj(),
        eps_out,
        eps_in,
        eps_out.conj(),
        C,
        optimize=True,
    )


def intensity(g):
    """Spectral intensity ``-Im g`` of a (contracted) complex Green's function/map."""
    return -np.imag(g)


def isotropic(chi):
    r"""
    Isotropic (polarization-averaged) spectrum ``Tr chi / m`` of a spectral tensor.

    Averaging :math:`\varepsilon^\dagger \chi \varepsilon` over all directions gives the
    normalized trace, so this equals the mean over any complete orthonormal polarization
    basis.

    Parameters
    ----------
    chi : ndarray
        Spectral tensor of shape ``(n_w, m, m)``.

    Returns
    -------
    ndarray
        Complex isotropic spectrum of shape ``(n_w,)``.
    """
    return np.trace(chi, axis1=1, axis2=2) / chi.shape[-1]


def circular_dichroism(chi):
    """
    XMCD spectrum: intensity difference ``I_cl - I_cr`` for propagation along z.

    Parameters
    ----------
    chi : ndarray
        Spectral tensor of shape ``(n_w, 3, 3)``.

    Returns
    -------
    ndarray
        Real dichroic spectrum of shape ``(n_w,)``.
    """
    g = contract_spectra_tensor(chi, ["cl", "cr"])
    return intensity(g[:, 0]) - intensity(g[:, 1])


def linear_dichroism(chi, pol_a="z", pol_b="x"):
    """
    XLD spectrum: intensity difference between two linear polarizations (default z - x).

    Parameters
    ----------
    chi : ndarray
        Spectral tensor of shape ``(n_w, 3, 3)``.
    pol_a, pol_b : str or array_like
        The two polarizations (anything :func:`polarization_vector` accepts).

    Returns
    -------
    ndarray
        Real dichroic spectrum of shape ``(n_w,)``.
    """
    g = contract_spectra_tensor(chi, [pol_a, pol_b])
    return intensity(g[:, 0]) - intensity(g[:, 1])
