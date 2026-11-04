"""``calc_selfenergy`` under ``mixed_valence`` auto/on: the Green's function of the charge-sector ensemble (MV 7).

The model is the minimal SIAM of ``test_selfenergy_end_to_end`` (two impurity spin orbitals, one bath level per spin;
16 many-body states), small enough that the exact grand-canonical Green's function is one dense diagonalisation.
At ``tau = 0.2`` the N = 0 and N = 2 sectors sit inside the thermal window beside the winner N = 1, so the
single-sector answer is off by 4e-3 and the ensemble one (weights ``c_s ~ Z_s exp(-(e0_s - e0)/tau)``) is exact to the
window's truncation.
"""

import contextlib
import io

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.model import BasisOptions, SolverOptions
from impurityModel.ed.selfenergy import calc_selfenergy
from impurityModel.test.gf.test_selfenergy_end_to_end import _meshes, _siam_model


def _dense_siam(eps_d=-0.7, eps_b=1.5, v=0.3, U=3.0):
    n = 4
    dim = 2**n

    def creation(i):
        m = np.zeros((dim, dim))
        for s in range(dim):
            if not (s >> i) & 1:
                m[s | (1 << i), s] = (-1) ** bin(s & ((1 << i) - 1)).count("1")
        return m

    cd = [creation(i) for i in range(n)]
    c = [m.T for m in cd]
    h = sum(e * cd[i] @ c[i] for i, e in enumerate((eps_d, eps_d, eps_b, eps_b)))
    for a, b in ((0, 2), (1, 3)):
        h = h + v * (cd[a] @ c[b] + cd[b] @ c[a])
    h = h + U * cd[0] @ c[0] @ cd[1] @ c[1]
    return h, cd


def _exact_gc_green(tau, z, orbital=0, dropped=()):
    """Exact grand-canonical impurity Green's function G_oo(z) over every electron count.

    ``dropped`` are excitation energies (E - E0) of initial states left out of the ensemble, as ``gf_min_weight`` does.
    """
    h, cd = _dense_siam()
    e, v = np.linalg.eigh(h)
    p = np.exp(-(e - e.min()) / tau)
    for de in dropped:
        p[np.abs(e - e.min() - de) < 1e-3] = 0.0
    p /= p.sum()
    a = v.T @ cd[orbital] @ v  # <m|c^dag|n>
    g = np.zeros(len(z), complex)
    for n in range(len(e)):
        for m in range(len(e)):
            g += p[n] * a[m, n] ** 2 / (z - (e[m] - e[n]))  # addition
            g += p[n] * a[n, m] ** 2 / (z + (e[m] - e[n]))  # removal
    return g


def _run(comm, mixed_valence, tau, gf_min_weight=None, gf_settle_retry=None):
    basis = BasisOptions(
        nominal_occ={0: 1},
        mixed_valence=mixed_valence,
        dN=None,
        truncation_threshold=np.inf,
        chain_restrict=False,
        occ_cutoff=1e-10,
        slater_weight_min=1e-10,
        tau=tau,
    )
    solver = SolverOptions(
        dense_cutoff=500,
        sparse_green=True,
        gf_method="lanczos",
        gf_min_weight=gf_min_weight,
        gf_settle_retry=gf_settle_retry,
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=comm, verbosity=0)
    res["stdout"] = out.getvalue()
    return res


def _g00(res):
    return res["gs_matsubara"][:, 0, 0]


@pytest.mark.mpi
def test_the_ensemble_green_function_is_the_exact_grand_canonical_one_and_the_winner_alone_is_not():
    comm = MPI.COMM_WORLD
    tau = 0.2
    on = _run(comm, "on", tau)
    off = _run(comm, False, tau)
    members = on["gs_ensemble"]["members"]  # replicated
    assert sorted(m["total"] for m in members) == [0, 1, 2]
    assert off["gs_ensemble"] is None
    if comm.rank == 0:
        exact = _exact_gc_green(tau, _meshes().iw)
        assert np.abs(_g00(on) - exact).max() < 1e-4
        # mutation: the winner alone (mixed_valence off) must NOT reproduce it
        assert np.abs(_g00(off) - exact).max() > 1e-3
        assert np.all(np.diagonal(on["sigma_real"], axis1=1, axis2=2).imag <= 1e-9)


@pytest.mark.mpi
def test_one_member_is_bit_identical_to_the_single_sector_run():
    """At low temperature the window holds one sector: ``on`` forms a one-member ensemble and changes nothing."""
    comm = MPI.COMM_WORLD
    tau = 0.01
    on = _run(comm, "on", tau)
    off = _run(comm, False, tau)
    assert len(on["gs_ensemble"]["members"]) == 1
    if comm.rank == 0:
        for key in ("sigma", "sigma_real", "sigma_static", "sigma_moment_1", "sigma_moment_2", "gs_matsubara"):
            np.testing.assert_array_equal(on[key], off[key], err_msg=key)


@pytest.mark.mpi
def test_gf_min_weight_drops_a_whole_minority_member_but_never_the_global_ground_manifold():
    """The rule normalises by the running sum over members. At 0.05 the 1.2% (N = 0) and 0.14% (N = 2) members have no
    state above it and are skipped whole; the winner's ground manifold is kept regardless, so G is the winner's."""
    comm = MPI.COMM_WORLD
    tau = 0.2
    cut = _run(comm, "on", tau, gf_min_weight=0.05)
    mild = _run(comm, "on", tau, gf_min_weight=1e-9)  # nothing falls below this: the full ensemble
    full = _run(comm, "on", tau)
    off = _run(comm, False, tau)
    if comm.rank == 0:
        assert cut["stdout"].count("member is skipped") == 2
        assert "member is skipped" not in mild["stdout"]
        np.testing.assert_allclose(_g00(mild), _g00(full), atol=1e-12)
        np.testing.assert_allclose(_g00(cut), _g00(off), atol=1e-6)  # only the winner's states remain


@pytest.mark.mpi
@pytest.mark.parametrize("mode", ["auto", "on"])
@pytest.mark.parametrize("gf_min_weight", [None, 1e-12])
def test_one_member_ensemble_reproduces_every_array_of_the_single_sector_run(mode, gf_min_weight):
    """Not only Sigma: every array the run returns (G, moments, rho, energies ...) is bit-identical to ``off``."""
    comm = MPI.COMM_WORLD
    tau = 0.01
    ens = _run(comm, mode, tau, gf_min_weight=gf_min_weight)
    off = _run(comm, False, tau, gf_min_weight=gf_min_weight)
    assert len(ens["gs_ensemble"]["members"]) == 1
    if comm.rank == 0:
        arrays = [k for k, v in off.items() if isinstance(v, np.ndarray)]
        assert len(arrays) >= 8, arrays
        for key in arrays:
            np.testing.assert_array_equal(ens[key], off[key], err_msg=key)


def test_member_coefficients_weigh_each_member_by_the_share_its_kept_states_carry():
    from impurityModel.ed.selfenergy import _member_coefficients

    # a member that kept half its weight counts for half; one that kept nothing for nothing; the rest renormalise
    assert _member_coefficients([0.6, 0.3, 0.1], [1.0, 0.5, 0.0]) == pytest.approx([0.6 / 0.75, 0.15 / 0.75, 0.0])
    assert _member_coefficients([0.7, 0.3], [1.0, 1.0]) == pytest.approx([0.7, 0.3])  # nothing dropped: unchanged


@pytest.mark.mpi
def test_a_partly_dropped_member_ensemble_stays_close_to_the_exact_one_without_those_states():
    """tau = 0.5, gf_min_weight 0.02: seven initial states dropped across two members; G stays within the thermal
    window's own truncation (2.8e-4, as the undropped ensemble) of the exact G without them."""
    comm = MPI.COMM_WORLD
    tau = 0.5
    res = _run(comm, "on", tau, gf_min_weight=0.02)
    if comm.rank == 0:
        assert "dropped 2 of 4" in res["stdout"] and "dropped 5 of 6" in res["stdout"]
        exact = _exact_gc_green(tau, _meshes().iw, dropped=(2.2804, 2.2804, 1.5402, 1.5402, 1.5402, 2.5026, 3.8234))
        assert np.abs(_g00(res) - exact).max() < 5e-4


def test_a_later_member_inherits_the_pinned_gf_caps_except_those_below_its_own_support():
    """One cap per calculation, but never a cap below a member's ground-state basis (units would freeze at seeds)."""
    import types

    from impurityModel.ed.selfenergy import _inherit_gf_caps

    shared = {"small": (100, 1), "large": (10_000, 2)}
    member = types.SimpleNamespace(size=500)
    _inherit_gf_caps(member, shared)
    assert member._auto_gf_caps == {"large": (10_000, 2)}
    _inherit_gf_caps(member, {"other": (1, 1)})  # already pinned: left alone
    assert member._auto_gf_caps == {"large": (10_000, 2)}
    first = types.SimpleNamespace(size=500)
    _inherit_gf_caps(first, None)  # the dominant member sizes its own
    assert not hasattr(first, "_auto_gf_caps")


def test_a_later_member_that_realised_the_lowest_energy_keeps_its_ground_manifold():
    from impurityModel.ed.selfenergy import _holds_the_global_ground

    assert _holds_the_global_ground(True, [0.0, 1.0], 0.0)
    assert not _holds_the_global_ground(False, [0.5, 1.0], 0.0)  # an ordinary minority member
    assert _holds_the_global_ground(False, [-0.5, 1.0], 0.0)  # a mis-ranked walk: it is the global minimum


@pytest.mark.mpi
@pytest.mark.parametrize("retry", [True, None])
def test_the_settle_retry_runs_once_per_member_on_that_members_own_run(monkeypatch, retry):
    """Each member's Green's function is attributed and (when asked) re-solved on its own ``GFRun``: one
    ``_settle_retry`` call per member, each with a live run and the user's ``gf_settle_retry``. Skipping later members
    fails the count, dropping the run fails the live-run check; default off hands no run at all."""
    from impurityModel.ed import selfenergy

    calls = []
    real = selfenergy._settle_retry

    def spy(**kw):
        calls.append((kw["retry"], id(kw["run"]) if kw["run"] is not None else None))
        return real(**kw)

    monkeypatch.setattr(selfenergy, "_settle_retry", spy)
    res = _run(MPI.COMM_WORLD, "on", 0.2, gf_settle_retry=retry)
    n_members = len(res["gs_ensemble"]["members"])
    assert n_members == 3 and len(calls) == n_members, calls
    assert all(flag is bool(retry) for flag, _ in calls), calls
    runs = [r for _, r in calls]
    if retry:
        assert all(r is not None for r in runs)
    else:
        assert all(r is None for r in runs)
