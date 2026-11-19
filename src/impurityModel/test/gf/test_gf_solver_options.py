"""The Green's-function options a ``get_Greens_function`` call hands to the one Lanczos kernel.

These tests pin the contract at the layers the options pass through:

* ``SolverOptions`` no longer carries the retired ``gf_method`` / ``gf_admission`` / ``gf_admit_tol``;
* an unset prune tolerance reaches the kernel as the default, an explicit one as given;
* the refinement spec reaches the kernel unchanged.
"""

import pytest

from impurityModel.ed.model import SolverOptions
from impurityModel.test.support.gf_oracles import _run_driver


@pytest.mark.parametrize("name", ["gf_method", "gf_admission", "gf_admit_tol"])
def test_a_retired_option_is_refused(name):
    with pytest.raises(TypeError, match=name):
        SolverOptions(**{name: "lanczos" if name == "gf_method" else None})


def test_the_prune_options_reach_the_kernel_and_beat_the_default(monkeypatch):
    from impurityModel.ed import gf_engine

    seen = []
    real = gf_engine.block_Green_sparse

    def spy(*args, **kwargs):
        seen.append((kwargs.get("prune_tol"), kwargs.get("first_shell_tol")))
        return real(*args, **kwargs)

    monkeypatch.setattr(gf_engine, "block_Green_sparse", spy)
    _run_driver(None)
    assert set(seen) == {(1e-3, None)}, "an unset tolerance reaches the kernel as the default"
    seen.clear()
    _run_driver(None, extra={"gf_prune_tol": 1e-3, "gf_first_shell_tol": 1e-5})
    assert set(seen) == {(1e-3, 1e-5)}


def test_the_refine_spec_reaches_the_kernel(monkeypatch):
    from impurityModel.ed import gf_engine
    from impurityModel.ed.gf_refinement import RefineSpec

    seen = []
    real = gf_engine.block_Green_sparse

    def spy(*args, **kwargs):
        seen.append(kwargs.get("refine"))
        return real(*args, **kwargs)

    monkeypatch.setattr(gf_engine, "block_Green_sparse", spy)
    _run_driver(None)
    assert set(seen) == {None}
    seen.clear()
    spec = RefineSpec(rounds=2, tol=1e-4, swap=0.2)
    _run_driver(None, extra={"gf_refine": spec})
    assert set(seen) == {spec}
