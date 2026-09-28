"""The Green's-function units' measured memory guard (``_CappedBasisProxy(memory_budget=...)``).

The byte model behind an auto GF cap has never been validated at scale, so, like the ground state,
the GF recurrence gets a guard on *measured* RSS: set per color by ``run_units_distributed``, it
freezes an auto unit's support where it stands once the color's MAX resident set reaches the
budget, and only warns for a cap the user set (which is final).
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.gf_primitives import _CappedBasisProxy
from impurityModel.ed.greens_function import get_Greens_function
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.memory_estimate import CapPolicy, reset_memory_warnings

# Six impurity orbitals with nearest-neighbour hopping: the removal recurrence provably grows past
# its one-determinant seed (the same fixture shape as test_gf_unit_basis_report).
_TERMS = {((i, "c"), (i, "a")): 0.1 * (i + 1) for i in range(6)}
for _i in range(5):
    _TERMS[((_i, "c"), (_i + 1, "a"))] = 0.4
    _TERMS[((_i + 1, "c"), (_i, "a"))] = 0.4
HOP = ManyBodyOperator(_TERMS)
GROUND = b"\xe0"
BLOCKS = [[0], [1]]
CAP = 100_000  # never reached by this fixture: any freeze below it is the memory guard's


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("GS_MEMORY_BUDGET_SAFETY", raising=False)
    reset_memory_warnings()
    yield
    reset_memory_warnings()


def _force_budget(monkeypatch, budget_bytes):
    """A GF guard budget every RSS sample exceeds. Patched at its source rather than driven by a
    tiny GS_MEMORY_BUDGET_SAFETY: the budget is resident-at-entry plus a floored headroom, and the
    heap trimmed between units can leave the current RSS *below* that entry figure."""
    from impurityModel.ed import gf_units as gu

    monkeypatch.setattr(gu, "_gf_memory_budget", lambda available, resident: budget_bytes)


def _basis(comm, policy, split_threshold=None):
    kwargs = {} if split_threshold is None else {"split_threshold": split_threshold}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[GROUND],
        comm=comm,
        truncation_threshold=CAP,
        **kwargs,
    )
    basis.cap_policy = policy
    return basis


def _gf(comm, policy, split_threshold=None, gf_method="lanczos"):
    rank0 = comm is None or comm.rank == 0
    psi = ManyBodyState({SlaterDeterminant.from_bytes(GROUND): 1.0} if rank0 else {}, width=1)
    return get_Greens_function(
        matsubara_mesh=None,
        omega_mesh=np.linspace(-3.0, 3.0, 21),
        psis=[psi],
        es=[0.0],
        tau=1.0,
        basis=_basis(comm, policy, split_threshold),
        hOp=HOP,
        delta=0.1,
        blocks=BLOCKS,
        verbose=False,
        verbose_extra=False,
        reort=None,
        dN=3,
        occ_cutoff=1e-9,
        slaterWeightMin=0.0,
        sparse=True,
        gf_method=gf_method,
    )


def _basis_cap_messages(report):
    return [d.message for d in report.diagnostics if d.name == "basis_cap"]


def test_the_proxy_guard_is_off_by_default():
    proxy = _CappedBasisProxy(_basis(None, None), CAP)
    assert proxy.memory_budget is None and not proxy._over_memory_budget()


def test_an_auto_gf_unit_freezes_when_measured_memory_reaches_the_budget(monkeypatch):
    """A budget every sample exceeds: the auto unit's support freezes at its seed, and the
    report attributes it to the memory guard, not to the cap."""
    _force_budget(monkeypatch, 1)
    _, _, report = _gf(None, CapPolicy(gs=CAP, gf=CAP, from_memory=True))
    messages = _basis_cap_messages(report)
    assert any("memory guard" in m for m in messages), messages


def test_a_user_gf_cap_is_not_frozen_by_memory_only_warned_about(monkeypatch, capfd):
    _force_budget(monkeypatch, 1)
    _, _, report = _gf(None, CapPolicy(gs=CAP, gf=CAP, from_memory=False))
    messages = _basis_cap_messages(report)
    assert not any("memory guard" in m for m in messages), messages
    _out, err = capfd.readouterr()
    assert err.count("WARNING determinant cap: a Green's-function unit") == 1


def test_the_guard_changes_nothing_under_budget(monkeypatch):
    """Default budget on a toy problem: bit-identical to the guard switched off."""
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", "0")
    reference = _gf(None, CapPolicy(gs=CAP, gf=CAP, from_memory=True))
    monkeypatch.delenv("GS_MEMORY_BUDGET_SAFETY")
    guarded = _gf(None, CapPolicy(gs=CAP, gf=CAP, from_memory=True))
    np.testing.assert_array_equal(reference[1][0], guarded[1][0])
    np.testing.assert_array_equal(reference[1][1], guarded[1][1])


@pytest.mark.mpi
def test_the_gf_guard_is_collective_safe_mpi(monkeypatch):
    """Run at -n 2 and -n 3 (an empty rank): the guard's RSS reduction sits on the pre-freeze path
    beside the admission count's, so a freeze must be reached on every rank of a color together
    -- a mismatch deadlocks here instead of passing."""
    _force_budget(monkeypatch, 1)
    comm = MPI.COMM_WORLD
    # split_threshold=0 forces one color spanning every rank, so the proxy's RSS reduction really
    # runs across ranks (and across an empty one at -n 3) instead of on one-rank colors.
    _, _, report = _gf(comm, CapPolicy(gs=CAP, gf=CAP, from_memory=True), split_threshold=0)
    if comm.rank == 0:
        assert any("memory guard" in m for m in _basis_cap_messages(report))
    _gf(comm, CapPolicy(gs=CAP, gf=CAP, from_memory=False), split_threshold=0)
    comm.Barrier()


def test_the_guard_budget_has_the_sizing_floor():
    """After a heavy ground state (resident >= available) the guard must still leave the headroom
    the auto cap was sized with; a floorless `s * (A + R)` sits at the resident set and freezes
    every unit at its seeds."""
    from impurityModel.ed import memory_estimate as me
    from impurityModel.ed.gf_units import _gf_memory_budget

    available, resident = 1 * 2**30, 2 * 2**30
    budget = _gf_memory_budget(available, resident)
    assert budget - resident == pytest.approx(me._resident_adjusted_budget(0.5, available, resident), rel=1e-9)
    assert budget - resident > 0.2 * available


def test_an_unlimited_gf_unit_is_still_memory_guarded(monkeypatch):
    """`unlimited` sets no cap but keeps the guard: the GF recurrence is proxied for the guard's
    sake alone, and at a budget every sample exceeds it freezes."""
    _force_budget(monkeypatch, 1)
    _, _, report = _gf(None, CapPolicy(gs=float("inf"), gf=float("inf"), from_memory=True))
    messages = _basis_cap_messages(report)
    assert any("memory guard" in m for m in messages), messages


def test_auto_gf_caps_are_pinned_per_kernel_and_clamped(monkeypatch):
    """One pinned number per (reort, method) for a calculation; a later, wider stage that cannot
    afford it on all ranks is clamped instead of trusted to a guard it may not have."""
    from impurityModel.ed import gf_units as gu
    from impurityModel.ed import memory_estimate as me

    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 4 * 2**20)
    basis = _basis(None, CapPolicy(gs=CAP, gf=None, from_memory=True))
    narrow = gu._pinned_auto_gf_cap(basis, [1], 1, None, "lanczos", 0)
    assert gu._pinned_auto_gf_cap(basis, [1], 1, None, "lanczos", 10**6) == narrow, "pinned"
    wide = gu._pinned_auto_gf_cap(basis, [1], 40, None, "lanczos", 0)
    assert wide < narrow, "a wider stage is clamped to what it can afford"
    other = gu._pinned_auto_gf_cap(basis, [1], 1, None, "bicgstab", 0)
    assert set(basis._auto_gf_caps) == {("None", "lanczos"), ("None", "bicgstab")}
    assert other != narrow


def test_clones_carry_the_cap_policy_and_the_gf_guard():
    """Kernels build their solve bases by cloning; the guard must reach them whichever basis
    they were handed, not only the split basis `run_units_distributed` configured."""
    policy = CapPolicy(gs=CAP, gf=None, from_memory=True)
    basis = _basis(None, policy)
    basis.gf_memory_budget, basis.gf_memory_policy = 123, "warn"
    for derived in (basis.clone(initial_basis=[]), basis.copy()):
        assert derived.cap_policy is policy
        assert (derived.gf_memory_budget, derived.gf_memory_policy) == (123, "warn")
    assert not hasattr(_basis(None, None).clone(initial_basis=[]), "gf_memory_budget")


def test_guarded_proxy_proxies_for_a_cap_or_a_budget_only():
    from impurityModel.ed.gf_primitives import _CappedBasisProxy, guarded_proxy

    bare = _basis(None, None)
    assert guarded_proxy(bare, np.inf) is bare
    assert isinstance(guarded_proxy(bare, 50), _CappedBasisProxy)
    bare.gf_memory_budget, bare.gf_memory_policy = 10**12, "warn"
    unlimited = guarded_proxy(bare, np.inf)
    assert isinstance(unlimited, _CappedBasisProxy)
    assert unlimited.memory_budget == 10**12 and unlimited.memory_policy == "warn"


def test_the_bicgstab_gf_kernel_is_guarded_too(monkeypatch, capfd):
    """The per-frequency driver builds its own proxies; under a user cap and a budget every
    sample exceeds, it must warn exactly as the Lanczos kernel does."""
    _force_budget(monkeypatch, 1)
    _gf(None, CapPolicy(gs=CAP, gf=CAP, from_memory=False), gf_method="bicgstab")
    _out, err = capfd.readouterr()
    assert err.count("WARNING determinant cap: a Green's-function unit") == 1
