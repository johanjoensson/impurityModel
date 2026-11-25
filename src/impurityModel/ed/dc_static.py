r"""The static double-counting schemes: FLL, AMF and Held's :math:`\Sigma(\infty)`.

Unlike the two searches in :mod:`dc_criteria`, these are closed-form one-body computations with
no ED solve and no MPI collective -- deterministic NumPy, identical on every rank. Each evaluates
its formula at the DFT impurity occupation or density matrix from :mod:`dc_reference` (and so
inherits its saturation check) unless the caller supplies the filling explicitly.

:func:`impurityModel.ed.atomic_physics.uj_from_u4` derives the average Coulomb repulsion and
exchange that FLL and AMF need from ``model.u4``; :func:`impurityModel.ed.sigma.get_Sigma_static`
is the static Hartree-Fock self-energy :func:`sigma_inf_dc` *is*.

**Several shells** (``model.shell_ls`` with more than one entry). FLL, AMF and nominal are evaluated
per shell: each shell ``a`` has its own occupation ``N_a`` and its own ``(U_a, J_a)``, read from its
own block of ``u4``. Each shell also feels the Hartree and averaged exchange of the other shells'
charge, taken from the inter-shell elements of ``u4`` at the uniform density
``⊕_a (N_a / n_a) I_a`` (:func:`_fll_multishell_matrix`). The result is block-diagonal by shell. An
occupation, ``u`` or ``j`` given explicitly must then have one value per shell; a single number is
refused rather than split. :func:`with_bathless_shells_at_nominal_fll` gives the ED criteria the
double counting of a shell they cannot shift.
"""

import dataclasses

import numpy as np

from impurityModel.ed import dc_record
from impurityModel.ed.atomic_physics import uj_from_u4
from impurityModel.ed.dc_reference import (
    _SATURATION_ADVICE,
    _noninteracting_impurity_rho,
    _reference_impurity_occupation,
    _warn_if_reference_saturated,
)
from impurityModel.ed.lie_algebra import extract_tensors, rotate_two_body, tensors_to_operator
from impurityModel.ed.observables import shell_column_slices, shell_index_of_orbitals
from impurityModel.ed.sigma import get_Sigma_static
from impurityModel.ed.symmetries import bath_anchors


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
    basis with :func:`impurityModel.ed.lie_algebra.rotate_two_body` using
    ``model.rot_to_spherical``, then reads off ``(U, J)``.

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


def _is_multishell(model):
    """Whether ``model`` declares more than one shell. An inferred single shell is the legacy path."""
    return model.shell_ls is not None and len(model.shell_ls) > 1


def _shell_rows(model):
    """Per shell, the positions (into ``model.impurity_indices``) of the impurity orbitals it holds."""
    if isinstance(model.rot_to_spherical, dict):
        raise ValueError("a multi-shell double counting needs an ndarray rot_to_spherical with shell_ls, not a dict.")
    index = shell_index_of_orbitals(model.rot_to_spherical, model.shell_ls)
    return [np.flatnonzero(index == a) for a in range(len(model.shell_ls))]


def shells_with_bath(model):
    """Per shell of a multi-shell ``model``, whether any bath orbital belongs with it.

    The rule :func:`solver_basis.prepare_solver_basis` uses for ``shell_has_bath``: a bath orbital belongs with
    the impurity orbital :func:`symmetries.bath_anchors` gives it.
    """
    rows = _shell_rows(model)
    impurity = list(model.impurity_indices)
    bath = [o for o in range(model.n_spin_orbitals) if o not in set(impurity)]
    h = extract_tensors(model.h0, n_orb=model.n_spin_orbitals, two_body=False)[0]
    shell_of = {impurity[k]: a for a, r in enumerate(rows) for k in r}
    reached = {shell_of[anchor] for anchor in bath_anchors(h, impurity, bath).values()}
    return [a in reached for a in range(len(rows))]


def _per_shell(value, n_shells, name):
    """``value`` as a list with one entry per shell (``None`` entries: derive it).

    A dict must be keyed by the shell indices ``0 .. n_shells-1``. A single number is refused: splitting one
    occupation or one ``(U, J)`` over several shells is exactly what a per-shell double counting must not do.
    """
    if value is None:
        return [None] * n_shells
    if isinstance(value, dict):
        if set(value) != set(range(n_shells)):
            raise ValueError(f"{name} must be keyed by the shell indices {list(range(n_shells))}, got {sorted(value)}.")
        return [None if value[a] is None else float(value[a]) for a in range(n_shells)]
    if np.ndim(value) == 0:
        raise ValueError(
            f"{name}={value} is one number, but this impurity has {n_shells} shells: give one value per shell "
            "(a sequence, or a dict keyed by shell index)."
        )
    values = list(value)
    if len(values) != n_shells:
        raise ValueError(f"{name} has {len(values)} entries, but this impurity has {n_shells} shells.")
    return [None if v is None else float(v) for v in values]


def _spin_major_basis(rot_rows, l, tol=1e-6):
    """A unitary on one shell's rows that orders them by spin: the first half one spin, the second the other.

    ``rot_rows`` is the shell's block of ``rot_to_spherical`` (its rows, its ``2(2l+1)`` columns, the first
    ``2l+1`` of which are one spin). The projector onto that spin, compressed onto the rows, has eigenvalues 0
    and 1 exactly when the rows span a spin-collinear space; its eigenvectors are then the spin-major basis.
    This works for a partial set (t2g only) as well as a whole shell: :func:`uj_from_u4` averages over the
    orbitals present, never over spherical columns the set does not hold.
    """
    half = 2 * l + 1
    spin = rot_rows[:, :half]
    projector = spin @ spin.conj().T
    eigenvalues, vectors = np.linalg.eigh(0.5 * (projector + projector.conj().T))
    if np.any(np.minimum(np.abs(eigenvalues), np.abs(eigenvalues - 1.0)) > tol):
        raise ValueError(
            f"the orbitals of the l={l} shell are not spin-collinear (e.g. a spin-orbit-coupled basis), so no "
            "(U, J) can be read from u4; pass explicit u=/j= per shell."
        )
    first, second = vectors[:, eigenvalues > 0.5], vectors[:, eigenvalues <= 0.5]
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            f"the l={l} set holds {first.shape[1]} orbitals of one spin and {second.shape[1]} of the other; "
            "(U, J) needs both spins of every orbital. Pass explicit u=/j= per shell."
        )
    return np.hstack([first, second])


def _shell_uj(model, u4_dense, rows, wanted=None):
    """``[(U_a, J_a), ...]``: :func:`uj_from_u4` on each shell's own block of ``u4``.

    ``wanted`` (shell indices) limits the work to those shells; the others are ``None``. A shell whose
    ``(U, J)`` is not needed is never asked to be spin-collinear.
    """
    rot = np.asarray(model.rot_to_spherical, dtype=complex)
    out = []
    for a, (r, l, cols) in enumerate(zip(rows, model.shell_ls, shell_column_slices(model.shell_ls))):
        if wanted is not None and a not in wanted:
            out.append(None)
            continue
        w = _spin_major_basis(rot[np.ix_(r, np.arange(rot.shape[1])[cols])], int(l))
        out.append(uj_from_u4(rotate_two_body(u4_dense[np.ix_(r, r, r, r)], w)))
    return out


def _shell_reference_occupations(model, tau, rows, warn=True):
    """Per shell, its DFT reference occupation: the trace of its block of the Fermi-filled ``h0``.

    The saturation warning is per shell and only for a shell with a bath: a deep, bath-less core is meant
    to be full.
    """
    rho = _noninteracting_impurity_rho(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
    occupations = [float(np.real(np.trace(rho[np.ix_(r, r)]))) for r in rows]
    if warn:
        for n_a, r, has_bath in zip(occupations, rows, shells_with_bath(model)):
            if has_bath:
                _warn_if_reference_saturated(n_a, len(r), _SATURATION_ADVICE["static"])
    return occupations


def _shell_block_diagonal(matrix, rows):
    """``matrix`` with every block between two different shells set to zero."""
    out = np.zeros_like(matrix)
    for r in rows:
        out[np.ix_(r, r)] = matrix[np.ix_(r, r)]
    return out


def _uniform_shell_density(rows, occupations, n_imp):
    r"""``\bar\rho = \oplus_a (N_a / n_a) I_a``: each shell's charge spread evenly over its own orbitals."""
    rho = np.zeros((n_imp, n_imp), dtype=complex)
    for r, n_a in zip(rows, occupations):
        rho[r, r] = n_a / len(r)
    return rho


def _intra_shell_part(u4_dense, rows):
    """The elements of ``u4`` whose four indices all lie in one shell."""
    shell = np.empty(u4_dense.shape[0], dtype=int)
    for a, r in enumerate(rows):
        shell[r] = a
    s = shell[:, None, None, None]
    same = (s == shell[None, :, None, None]) & (s == shell[None, None, :, None]) & (s == shell[None, None, None, :])
    return np.where(same, u4_dense, 0.0)


def _shell_summary(model, occupations, uj=None, levels=None):
    """The record's ``shells`` field: one ``l=.., N=.., U=.., J=..`` entry per shell."""
    parts = []
    for a, l in enumerate(model.shell_ls):
        entry = f"l={l}: N={occupations[a]:.4f}"
        if uj is not None:
            entry += f" U={uj[a][0]:.4f} J={uj[a][1]:.4f}"
        if levels is not None:
            entry += f" dc_level={levels[a]:.6f}"
        parts.append(entry)
    return "; ".join(parts)


def _shell_levels(dc, rows):
    """Per shell, the mean diagonal of its block of ``dc``."""
    return [float(np.real(np.trace(dc[np.ix_(r, r)]))) / len(r) for r in rows]


def _fll_multishell_matrix(model, *, n, u, j, tau=None):
    r"""Per-shell FLL on a multi-shell model. Returns ``(dc, occupations, uj, rows)``.

    .. math::
        V = \bigoplus_a \Big[v_a I_a + \big(\Sigma_{static}(u_4, \bar\rho)
            - \Sigma_{static}(u_4^{intra}, \bar\rho)\big)_{aa}\Big],\quad
        v_a = U_a (N_a - \tfrac12) - \tfrac12 J_a (N_a - 1).

    FLL and AMF differ only in the self-interaction, which is inside a shell; between shells both are the
    Hartree and averaged exchange of the other shells' charge at the uniform density :math:`\bar\rho`. So
    each shell gets its own FLL potential plus the inter-shell part of AMF. With no inter-shell elements in
    ``u4`` the result is block-diagonal copies of the single-shell FLL.
    """
    n_shells = len(model.shell_ls)
    rows = _shell_rows(model)
    n_list = _per_shell(n, n_shells, "n")
    u_list = _per_shell(u, n_shells, "u")
    j_list = _per_shell(j, n_shells, "j")
    n_imp = len(model.impurity_indices)
    needs_u4 = any(v is None for v in u_list + j_list)
    u4_dense = _model_u4_dense(model) if (needs_u4 or model.u4 is not None) else None
    if any(v is None for v in n_list):
        reference = _shell_reference_occupations(model, tau, rows)
        n_list = [ref if v is None else v for v, ref in zip(n_list, reference)]
    if needs_u4:
        derived = _shell_uj(
            model, u4_dense, rows, wanted={a for a in range(n_shells) if u_list[a] is None or j_list[a] is None}
        )
        u_list = [d[0] if v is None else v for v, d in zip(u_list, derived)]
        j_list = [d[1] if v is None else v for v, d in zip(j_list, derived)]
    dc = np.zeros((n_imp, n_imp), dtype=complex)
    if u4_dense is not None:
        rho = _uniform_shell_density(rows, n_list, n_imp)
        dc += get_Sigma_static(u4_dense, rho) - get_Sigma_static(_intra_shell_part(u4_dense, rows), rho)
        dc = _shell_block_diagonal(dc, rows)
    for r, n_a, u_a, j_a in zip(rows, n_list, u_list, j_list):
        dc[r, r] += u_a * (n_a - 0.5) - 0.5 * j_a * (n_a - 1.0)
    return dc, n_list, list(zip(u_list, j_list)), rows


def _fll_multishell(model, record, *, n, u, j, tau=None):
    dc, occupations, uj, rows = _fll_multishell_matrix(model, n=n, u=u, j=j, tau=tau)
    record.update(status="closed_form", n_ref=float(sum(occupations)))
    record["shells"] = _shell_summary(model, occupations, uj, _shell_levels(dc, rows))
    record["dc_trace"], record["dc_level"] = dc_record.dc_levels(dc)
    record["dc_spread"] = dc_record.dc_spread(dc)
    return dc


def _nominal_per_shell(model, nominal_occupation):
    """The nominal occupation of each shell, from a per-shell dict or sequence (``BasisOptions.nominal_occ``)."""
    values = _per_shell(nominal_occupation, len(model.shell_ls), "nominal_occupation")
    if any(v is None for v in values):
        raise ValueError("nominal_occupation needs a value for every shell.")
    return values


def with_bathless_shells_at_nominal_fll(model, nominal_occ, *, rank=0):
    r"""``model`` with the double counting of every bath-less shell set to FLL at the nominal occupations.

    The ED criteria shift only the shells a bath hops to (``dc_guess + mu P``): a deep shell's occupation is
    pinned by its energy, so no criterion measured on the impurity can fix its double counting. It would
    otherwise keep whatever guess the caller sent. Here it gets the per-shell FLL potential of
    :func:`_fll_multishell_matrix` evaluated at ``nominal_occ`` (one entry per shell), which includes the
    Hartree and exchange of the other shells' nominal charge. The blocks between a bath-less shell and any
    other shell are zero. The shells with a bath keep the guess, and the search shifts them from there.

    Must run before the solver basis is built: ``H = h0 - DC + U`` is assembled from ``model.dc``, and the
    double counting searched for has to be the one the self-energy then solves with.

    Single shell, every shell with a bath, or none with a bath (the search refuses that case itself):
    ``model`` unchanged.
    """
    if not _is_multishell(model):
        return model
    has_bath = shells_with_bath(model)
    if all(has_bath) or not any(has_bath):
        return model
    # Only the bath-less blocks are kept, so only their (U, J) is derived: the intra-shell part of a shell with a
    # bath is discarded (placeholder 0), and its basis need not be spin-collinear.
    placeholder = [0.0 if has else None for has in has_bath]
    fll, _, _, rows = _fll_multishell_matrix(
        model, n=_nominal_per_shell(model, nominal_occ), u=placeholder, j=placeholder
    )
    n_imp = len(model.impurity_indices)
    dc = np.array(extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0], dtype=complex)
    for a, r in enumerate(rows):
        if has_bath[a]:
            continue
        dc[r, :] = 0.0
        dc[:, r] = 0.0
        dc[np.ix_(r, r)] = fll[np.ix_(r, r)]
        if rank == 0:
            level = float(np.real(np.trace(fll[np.ix_(r, r)]))) / len(r)
            print(
                f"DC of the bath-less shell l={model.shell_ls[a]}: FLL at the nominal occupations, "
                f"level {level:.6f} (the search does not shift it).",
                flush=True,
            )
    return dataclasses.replace(model, dc=tensors_to_operator(dc).to_dict())


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
        On several shells: one value per shell (sequence, or dict keyed by shell index; ``None``
        entries are derived), see the module docstring.
    u, j : float, optional
        Average Coulomb repulsion and exchange. ``None`` derives them from ``model.u4`` via
        :func:`_model_uj`.
        On several shells: one value per shell (sequence, or dict keyed by shell index; ``None``
        entries are derived), see the module docstring.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("fll") as record:
        if _is_multishell(model):
            return _fll_multishell(model, record, tau=tau, n=n, u=u, j=j)
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
        On several shells: one value per shell (sequence, or dict keyed by shell index; ``None``
        entries are derived), see the module docstring.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("amf") as record:
        n_imp = len(model.impurity_indices)
        if _is_multishell(model):
            rows = _shell_rows(model)
            occupations = _per_shell(n, len(rows), "n")
            if any(v is None for v in occupations):
                reference = _shell_reference_occupations(model, tau, rows)
                occupations = [ref if v is None else v for v, ref in zip(occupations, reference)]
            rho = _uniform_shell_density(rows, occupations, n_imp)
            dc = _shell_block_diagonal(get_Sigma_static(_model_u4_dense(model), rho), rows)
            record.update(status="closed_form", n_ref=float(sum(occupations)))
            record["shells"] = _shell_summary(model, occupations, levels=_shell_levels(dc, rows))
            record["dc_trace"], record["dc_level"] = dc_record.dc_levels(dc)
            record["dc_spread"] = dc_record.dc_spread(dc)
            return dc
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
    reference to compare a converged :func:`dc_occupation.fixed_occupation_dc` answer against.

    Parameters
    ----------
    model : ImpurityModel
    nominal_occupation : float or dict
        The nominal impurity occupation. A dict (``BasisOptions.nominal_occ``) is summed on one shell;
        on several shells it gives each shell's nominal occupation and must be keyed by shell index.
    u, j : float, optional
        Average Coulomb repulsion and exchange. ``None`` derives them from ``model.u4`` via
        :func:`_model_uj` (see :func:`fll_dc`).
        On several shells: one value per shell (sequence, or dict keyed by shell index; ``None``
        entries are derived), see the module docstring.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    with dc_record.recording("nominal") as record:
        # Same key, different provenance: `n_ref` here is the integer charge state the model was
        # set up for, not a filling read off the bath fit. The two are not comparable across
        # records -- that is the whole reason this scheme exists -- so the record says which it is.
        record["n_ref_kind"] = "nominal integer occupation"
        if _is_multishell(model):
            return _fll_multishell(model, record, n=_nominal_per_shell(model, nominal_occupation), u=u, j=j)
        if isinstance(nominal_occupation, dict):
            nominal_occupation = sum(nominal_occupation.values())
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
