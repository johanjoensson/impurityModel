"""The frozen-CSR decision's own memory budget (``GF_CSR_MEMORY_SAFETY``).

The auto GF cap fills the guard's share of memory (``GS_MEMORY_BUDGET_SAFETY``, 0.5) on the sparse
path, so judged against that same share the frozen ``P H P`` never fit and every capped unit stayed
on the ~45x slower sparse kernel. The decision gets its own, looser fraction of the rank's share;
the guard and the cap keep theirs.
"""

import numpy as np
import pytest

from impurityModel.ed import gf_solvers, gf_units
from impurityModel.ed.memory_estimate import _resident_adjusted_budget
from impurityModel.test.gf.test_gf_truncation import _excited_basis, _siam_6


@pytest.fixture(autouse=True)
def _knobs_unset(monkeypatch):
    monkeypatch.delenv("GF_CSR_MEMORY_SAFETY", raising=False)
    monkeypatch.delenv("GS_MEMORY_BUDGET_SAFETY", raising=False)


def test_the_csr_budget_is_the_guard_formula_at_its_own_fraction():
    available, resident = 3 * 2**30, 2**30
    guard = gf_units._gf_memory_budget(available, resident)
    csr = gf_units._gf_csr_memory_budget(available, resident)
    assert csr == int(resident + _resident_adjusted_budget(0.75, available, resident))
    assert csr > guard


def test_the_csr_budget_is_never_below_the_guard_and_off_with_it(monkeypatch):
    available, resident = 3 * 2**30, 2**30
    monkeypatch.setenv("GF_CSR_MEMORY_SAFETY", "0.1")
    assert gf_units._gf_csr_memory_budget(available, resident) == gf_units._gf_memory_budget(available, resident)
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", "0")
    assert gf_units._gf_csr_memory_budget(available, resident) is None


def test_the_decision_uses_the_csr_budget_the_gf_stage_put_on_the_basis(monkeypatch):
    """A proxy at the guard's budget would decline; the looser budget on the basis accepts."""
    basis = _excited_basis(np.inf)
    monkeypatch.setattr(gf_solvers, "current_rss_bytes", lambda: 10**9)
    monkeypatch.setattr(type(basis), "clone_from_keys", lambda self, mask: self, raising=False)

    class _Proxy:
        memory_frozen = False
        retained_mask = None
        memory_budget = 10**9 + 1

    assert gf_solvers._frozen_csr_basis(_Proxy(), basis, _siam_6(), False) is None, "premise: the guard declines"
    basis.gf_csr_memory_budget = 10**10
    assert gf_solvers._frozen_csr_basis(_Proxy(), basis, _siam_6(), False) is not None


def test_clones_and_copies_carry_the_csr_budget():
    """The kernels run on clones of the colour's basis (the excited basis, the frozen retained set)."""
    basis = _excited_basis(np.inf)
    basis.gf_csr_memory_budget = 12345
    assert basis.clone().gf_csr_memory_budget == 12345
    assert basis.copy().gf_csr_memory_budget == 12345
