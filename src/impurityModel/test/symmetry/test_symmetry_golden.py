"""Phase 0 golden baseline: freeze the spherical-basis self-energy physics.

Later phases must not change the physics:

* **P1** (correct GF block structure from the full Hamiltonian) and **P2** (solving in
  the symmetry-adapted single-particle basis and rotating outputs back to spherical) must
  reproduce these numbers to tolerance.
* **P3** (physics-derived occupation restrictions) must not shift the ground state.

The heavy solve (~2.5 min for the 10-bath NiO d-shell) only runs when opted in via
``RUN_SYMMETRY_GOLDEN=1``; regenerate the stored fingerprint with
``REGEN_SYMMETRY_GOLDEN=1``. The compare helpers (:func:`fingerprint`,
:func:`assert_matches_golden`) are imported by the P1/P2/P3 regression tests.

Run / regenerate::

    RUN_SYMMETRY_GOLDEN=1 pytest -s src/impurityModel/test/symmetry/test_symmetry_golden.py
    REGEN_SYMMETRY_GOLDEN=1 RUN_SYMMETRY_GOLDEN=1 \
        pytest -s src/impurityModel/test/symmetry/test_symmetry_golden.py
"""

import json
import os

import numpy as np
import pytest

RUN = os.environ.get("RUN_SYMMETRY_GOLDEN") == "1"
REGEN = os.environ.get("REGEN_SYMMETRY_GOLDEN") == "1"

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_SYMMETRY_GOLDEN=1 to run the symmetry golden baseline."),
]

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "symmetry_golden.json")

# Frequency sample points (as fractions of the mesh) captured in the fingerprint.
_FREQ_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _c2list(arr):
    """Serialize a complex ndarray as {re, im, shape} JSON-safe lists."""
    arr = np.asarray(arr, dtype=complex)
    return {"re": arr.real.tolist(), "im": arr.imag.tolist(), "shape": list(arr.shape)}


def _sample_indices(n):
    return sorted({round(f * (n - 1)) for f in _FREQ_FRACTIONS})


def fingerprint(result, n_imp):
    """Reduce a ``calc_selfenergy`` result dict to a basis-invariant physics fingerprint.

    Captures the impurity occupations (diagonal of the thermal density matrix), the static
    (Hartree-Fock) self-energy, and the full real-axis self-energy matrix at sampled
    frequencies. All are reported in the **spherical** basis (``calc_selfenergy`` already
    rotates its outputs back through ``rot_to_spherical``), so a symmetry-adapted-basis run
    produces the same fingerprint iff it preserves the physics.
    """
    thermal_rho = np.asarray(result["thermal_rho"], dtype=complex)
    imp_occ = np.real(np.diag(thermal_rho))[:n_imp]

    sigma_static = np.asarray(result["sigma_static"], dtype=complex)

    # Real-axis self-energy: calc_selfenergy returns the full (n_omega, n_imp, n_imp) matrix
    # (already reassembled from the blocks and rotated back to the spherical basis), so it is
    # independent of how the blocks were derived (P1) or which basis was solved in (P2).
    full = np.asarray(result["sigma_real"], dtype=complex)
    n_omega = full.shape[0]
    idx = _sample_indices(n_omega)
    full = full[idx]

    return {
        "n_imp": int(n_imp),
        "imp_occ": imp_occ.tolist(),
        "n_imp_total": float(np.sum(imp_occ)),
        "sigma_static": _c2list(sigma_static),
        "sigma_real_freq_idx": idx,
        "sigma_real_full": _c2list(full),
    }


def _cmp_c(name, golden_entry, current_entry, atol):
    g = np.asarray(golden_entry["re"]) + 1j * np.asarray(golden_entry["im"])
    c = np.asarray(current_entry["re"]) + 1j * np.asarray(current_entry["im"])
    assert g.shape == c.shape, f"{name} shape {c.shape} != golden {g.shape}"
    max_dev = np.max(np.abs(g - c)) if g.size else 0.0
    assert max_dev <= atol, f"{name} max deviation {max_dev:.3e} > atol {atol:.1e}"


def assert_matches_golden(fp, atol=1e-6):
    """Assert a fingerprint matches the stored golden baseline within ``atol``."""
    assert os.path.exists(GOLDEN_PATH), f"No golden baseline at {GOLDEN_PATH}; regenerate with REGEN_SYMMETRY_GOLDEN=1."
    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)
    assert fp["n_imp"] == golden["n_imp"]
    np.testing.assert_allclose(fp["imp_occ"], golden["imp_occ"], atol=atol, err_msg="impurity occupations drifted")
    assert abs(fp["n_imp_total"] - golden["n_imp_total"]) <= atol, "total impurity occupation drifted"
    _cmp_c("sigma_static", golden["sigma_static"], fp["sigma_static"], atol)
    assert fp["sigma_real_freq_idx"] == golden["sigma_real_freq_idx"], "frequency sample indices changed"
    _cmp_c("sigma_real_full", golden["sigma_real_full"], fp["sigma_real_full"], atol)


def _run_spherical_baseline():
    from mpi4py import MPI

    from impurityModel.ed import selfenergy
    from impurityModel.test.support._nio_workload import as_calc_selfenergy_args, build_selfenergy_inputs

    comm = MPI.COMM_WORLD
    n_omega = int(os.environ.get("SYMMETRY_GOLDEN_NW", "41"))
    # Physical NiO Ni(2+): d-only MLFT double counting (chargeTransferCorrection) so the ground
    # state is the genuine d8 energetic minimum (not a window-confined artifact), plus the valence
    # 3d spin-orbit coupling. These are golden-specific physics knobs; the shared builder defaults
    # to the SOC-free, DC-free d-shell used by the perf / driver-glue anchors.
    kwargs = build_selfenergy_inputs(
        nBaths=10, nValBaths=10, n0imp=8, n_omega=n_omega, rank=comm.rank, xi=0.096, chargeTransferCorrection=1.5
    )
    n_imp = 10
    result = selfenergy.calc_selfenergy(**as_calc_selfenergy_args(kwargs), comm=comm)
    return result, n_imp


def test_capture_or_compare_golden():
    """Regenerate (REGEN=1) or compare the spherical-basis self-energy fingerprint."""
    from mpi4py import MPI

    result, n_imp = _run_spherical_baseline()
    if MPI.COMM_WORLD.rank != 0:
        return
    fp = fingerprint(result, n_imp)
    if REGEN or not os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "w") as fh:
            json.dump(fp, fh, indent=1)
        print(f"Wrote golden baseline to {GOLDEN_PATH}")
        return
    assert_matches_golden(fp)
