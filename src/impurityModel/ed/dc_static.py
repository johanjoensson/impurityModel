r"""The static double-counting schemes: FLL, AMF and Held's :math:`\Sigma(\infty)`.

Unlike the two searches in :mod:`dc_criteria`, these are closed-form one-body computations with
no ED solve and no MPI collective -- deterministic NumPy, identical on every rank. Each evaluates
its formula at the DFT impurity occupation or density matrix from :mod:`dc_reference` (and so
inherits its saturation check) unless the caller supplies the filling explicitly.

:func:`impurityModel.ed.atomic_physics.uj_from_u4` derives the average Coulomb repulsion and
exchange that FLL and AMF need from ``model.u4``; :func:`impurityModel.ed.sigma.get_Sigma_static`
is the static Hartree-Fock self-energy :func:`sigma_inf_dc` *is*.
"""

import numpy as np

from impurityModel.ed import dc_record
from impurityModel.ed.atomic_physics import uj_from_u4
from impurityModel.ed.dc_reference import (
    _SATURATION_ADVICE,
    _noninteracting_impurity_rho,
    _reference_impurity_occupation,
    _warn_if_reference_saturated,
)
from impurityModel.ed.lie_algebra import extract_tensors, rotate_two_body
from impurityModel.ed.sigma import get_Sigma_static


def _model_u4_dense(model):
    r"""Recover the dense, impurity-local RSPt Coulomb tensor from ``model.u4``.

    ``model.u4`` is the *raw* (never canonicalized) operator dict built by
    :func:`impurityModel.ed.atomic_physics.getUop_from_rspt_u4`: one term per index quadruple,
    amplitude ``u4[i,j,k,l] / 2``, with no folding of equivalent terms. Wrapping it in a
    :class:`ManyBodyOperator` first (which canonicalizes/folds terms together) would lose the
    direct/exchange split this relies on -- do not do that before calling this function.
    :func:`extract_tensors` on the raw dict gives ``V[i,j,k,l] = u4[i,j,k,l] / 2`` entry-for-entry
    (no two raw keys ever map to the same ``V`` cell), so ``2 * V`` already recovers ``u4``
    exactly. ``V + V.transpose(1, 0, 3, 2)`` is used instead of a bare ``2 * V`` because it is
    numerically identical for a raw dict (via ``u4``'s own exchange symmetry, ``u4[i,j,k,l] =
    u4[j,i,l,k]``, which any RSPt-convention tensor satisfies by construction) while also being
    the natural, symmetric way to state "recover the tensor from the operator".

    Parameters
    ----------
    model : ImpurityModel

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp, n_imp, n_imp)
        The Coulomb tensor over the impurity spin-orbitals, in the model's input basis.

    Raises
    ------
    ValueError
        If ``model.u4`` is ``None``.
    """
    if model.u4 is None:
        raise ValueError("model.u4 is None; pass an explicit u=/j= (or n=/rho=) instead of deriving them from u4.")
    n_imp = len(model.impurity_indices)
    _, V, _ = extract_tensors(model.u4, n_orb=n_imp)
    return V + V.transpose(1, 0, 3, 2)


def _model_uj(model):
    r"""Average Coulomb repulsion and exchange (:func:`uj_from_u4`) derived from ``model.u4``.

    Rotates the dense impurity Coulomb tensor (:func:`_model_u4_dense`) into the spherical
    basis with :func:`impurityModel.ed.lie_algebra.rotate_two_body` (the same transformation as
    :func:`impurityModel.ed.greens_function.rotate_4index_U`, avoiding an import of the
    heavyweight solver module) using ``model.rot_to_spherical``, then reads off ``(U, J)``.

    Parameters
    ----------
    model : ImpurityModel

    Returns
    -------
    (U, J) : tuple of float

    Raises
    ------
    ValueError
        If ``model.u4`` is ``None``, or ``model.rot_to_spherical`` is a multi-group dict (per-
        group rotations are not supported here, matching :func:`fixed_peak_dc`'s restriction).
    """
    if isinstance(model.rot_to_spherical, dict):
        raise ValueError("_model_uj does not support a multi-group model.rot_to_spherical; pass explicit u=/j=.")
    u4_dense = _model_u4_dense(model)
    rotation = np.asarray(model.rot_to_spherical, dtype=complex)
    u4_spherical = rotate_two_body(u4_dense, rotation)
    return uj_from_u4(u4_spherical)


def fll_dc(model, *, tau=0.002, n=None, u=None, j=None):
    r"""Fully Localized Limit double counting, ``dc = [U(N - 1/2) - (J/2)(N - 1)] I``.

    ``U``, ``J`` default to :func:`_model_uj`'s spherical average (needs ``model.u4``); either
    may be overridden explicitly (e.g. from tabulated values), independently of the other. ``N``
    defaults to the DFT impurity occupation, the Fermi filling of the raw ``h0``
    (:func:`_noninteracting_impurity_rho`); an explicit ``N`` (together with both ``u`` and
    ``j``) needs no ``model.u4`` at all.

    .. note:: **Paramagnetic (spin-blind), like** :func:`amf_dc`. The literature FLL potential is
       spin-resolved, ``V_sigma = U(N - 1/2) - J(N_sigma - 1/2)``; setting ``N_sigma = N/2``
       gives the scalar form implemented here, so the spin splitting
       ``V_up - V_down = -J(N_up - N_down)`` is discarded. For a spin-polarized ground state that
       is not small -- Ni(2+) in NiO has ``N_up - N_down ~ 1.4`` and ``J ~ 0.9 eV``, so ~1.3 eV.
       Nothing downstream forces this: RSPt's ``sig_dc`` is a full spin-orbital matrix.

    .. note:: ``N`` is the *non-interacting* occupation. In DFT+U/DFT+DMFT the FLL ``N`` is
       conventionally the correlated impurity occupation, updated self-consistently, which makes
       this a non-self-consistent scheme. Combined with the bath-discretization sensitivity of
       the reference filling (see the module docstring) that is the dominant uncertainty here;
       :func:`nominal_dc` avoids both by using the nominal integer occupation.

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the DFT occupation (ignored if ``n`` is given).
    n : float, optional
        Impurity occupation ``N``. ``None`` uses the DFT impurity occupation.
    u, j : float, optional
        Average Coulomb repulsion and exchange. ``None`` derives them from ``model.u4`` via
        :func:`_model_uj`.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("fll") as record:
        return _fll(model, record, tau=tau, n=n, u=u, j=j)


def _fll(model, record, *, n, u, j, tau=None):
    """:func:`fll_dc`'s formula, with the record it fills handed in.

    Split out so :func:`nominal_dc` -- which *is* FLL, evaluated at the nominal occupation -- can
    reuse the formula without emitting a second record naming the wrong criterion. Every scheme
    emits exactly one record per public call.
    """
    n_imp = len(model.impurity_indices)
    if u is None or j is None:
        u_auto, j_auto = _model_uj(model)
        u = u_auto if u is None else u
        j = j_auto if j is None else j
    identity = np.identity(n_imp, dtype=complex)
    if n is None:
        n = _reference_impurity_occupation(model, tau)
    dc = (u * (n - 0.5) - 0.5 * j * (n - 1.0)) * identity
    record.update(status="closed_form", n_ref=n, u=u, j=j)
    record["dc_trace"], record["dc_level"] = dc_record.dc_levels(dc)
    record["dc_spread"] = dc_record.dc_spread(dc)
    return dc


def amf_dc(model, *, tau=0.002, n=None):
    r"""Around Mean Field double counting, ``dc = Σ_static(u4, (N / n_imp) I)``.

    The static Hartree-Fock self-energy (:func:`impurityModel.ed.sigma.get_Sigma_static`)
    evaluated at a *uniform* trial density matrix, ``N`` spread evenly over every impurity
    spin-orbital -- the defining assumption of AMF, that the impurity has no orbital *or spin*
    polarization -- as opposed to :func:`sigma_inf_dc`, which uses the actual (possibly
    anisotropic) density matrix. For a spin-polarized ground state this is the paramagnetic
    (spin-blind) AMF potential, not a per-spin-channel one. ``N`` defaults to the DFT impurity
    occupation, the Fermi filling of the raw ``h0`` (:func:`_noninteracting_impurity_rho`).

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the DFT occupation (ignored if ``n`` is given).
    n : float, optional
        Impurity occupation ``N``. ``None`` uses the DFT impurity occupation.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("amf") as record:
        n_imp = len(model.impurity_indices)
        u4_dense = _model_u4_dense(model)
        identity = np.identity(n_imp, dtype=complex)
        if n is None:
            n = _reference_impurity_occupation(model, tau)
        dc = get_Sigma_static(u4_dense, (n / n_imp) * identity)
        record.update(status="closed_form", n_ref=n)
        record["dc_trace"], record["dc_level"] = dc_record.dc_levels(dc)
        record["dc_spread"] = dc_record.dc_spread(dc)
        return dc


def nominal_dc(model, nominal_occupation, *, u=None, j=None):
    r"""FLL evaluated at the **nominal** (integer) impurity occupation, not the DFT reference.

    :func:`fll_dc`, :func:`amf_dc` and :func:`sigma_inf_dc` all default to the DFT reference
    filling (:mod:`dc_reference`'s Fermi fill of the raw ``h0``), which is a property of the
    *discretized* bath and can saturate at the full or empty shell for a coarse valence-only fit
    (measured: NiO's reference swings by 1.37 electrons between 1 and 15 bath states). This scheme
    needs no reference filling at all -- ``N`` is the charge state the model was set up for
    (``sum(BasisOptions.nominal_occ.values())``), an integer by construction -- so it cannot
    saturate and cannot inherit that fit sensitivity. Haule, *PRL* **115**, 196403 (2015) finds
    FLL at the nominal occupation close to the exact double counting for transition-metal oxides,
    which makes this the natural ``dc_guess`` for the first iteration of a CSC loop, and a
    reference to compare a converged :func:`dc_criteria.fixed_occupation_dc` answer against.

    Parameters
    ----------
    model : ImpurityModel
    nominal_occupation : float
        The nominal impurity occupation (``sum(BasisOptions.nominal_occ.values())``).
    u, j : float, optional
        Average Coulomb repulsion and exchange. ``None`` derives them from ``model.u4`` via
        :func:`_model_uj` (see :func:`fll_dc`).

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("nominal") as record:
        # Same key, different provenance: `n_ref` here is the integer charge state the model was
        # set up for, not a filling read off the bath fit. The two are not comparable across
        # records -- that is the whole reason this scheme exists -- so the record says which it is.
        record["n_ref_kind"] = "nominal integer occupation"
        return _fll(model, record, n=float(nominal_occupation), u=u, j=j)


def sigma_inf_dc(model, *, tau=0.002, rho=None):
    r"""Static Hartree-Fock double counting at the DFT density matrix,
    ``dc = Σ_static(u4, rho_imp)`` -- K. Held's :math:`\Sigma(\infty)` *evaluated on the
    uncorrelated* :math:`\rho`.

    Unlike :func:`amf_dc`, uses the actual (possibly anisotropic) non-interacting impurity
    density matrix rather than a uniform trial -- the two agree exactly when that density matrix
    happens to be uniform (e.g. a single, orbitally-degenerate shell), and differ whenever the
    impurity levels split. ``rho`` defaults to the DFT impurity density matrix, the Fermi
    filling of the raw ``h0`` (:func:`_noninteracting_impurity_rho`).

    .. warning:: **This is not** :math:`\lim_{\omega \to \infty} \Sigma(\omega)` **of the
       interacting impurity problem.** That limit is the Hartree-Fock contraction with the
       *correlated* density matrix; what is computed here is Hartree-Fock at the DFT density.
       For a uniform ``rho`` the two functions coincide with :func:`amf_dc` algebraically
       (asserted by ``test_static_dc.py``), so on a degenerate shell this scheme *is* AMF and
       carries none of the correlation Held's criterion is meant to capture. The genuine object
       is available downstream and is not used here: :func:`impurityModel.ed.sigma.
       get_Sigma_moments` returns :math:`\Sigma_\infty = M_1 - h_{corr}` from the interacting
       Green's-function moments, and ``calc_selfenergy`` returns ``thermal_rho``. Passing that
       ``thermal_rho`` as ``rho`` turns this into the self-consistent scheme.

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the DFT density matrix (ignored if ``rho`` is given).
    rho : numpy.ndarray, shape (n_imp, n_imp), optional
        Impurity density matrix. ``None`` uses the DFT impurity density matrix.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("sigma_inf") as record:
        u4_dense = _model_u4_dense(model)
        if rho is None:
            rho = _noninteracting_impurity_rho(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
            _warn_if_reference_saturated(
                float(np.real(np.trace(rho))), len(model.impurity_indices), _SATURATION_ADVICE["static"]
            )
        rho = np.asarray(rho, dtype=complex)
        dc = get_Sigma_static(u4_dense, rho)
        # The only scheme here whose dc need not be uniform: `dc_level` is then an average over an
        # orbital-dependent potential, and DC_DIAGNOSTICS is what shows the matrix itself.
        record.update(status="closed_form", n_ref=float(np.real(np.trace(rho))))
        record["dc_trace"], record["dc_level"] = dc_record.dc_levels(dc)
        record["dc_spread"] = dc_record.dc_spread(dc)
        return dc
