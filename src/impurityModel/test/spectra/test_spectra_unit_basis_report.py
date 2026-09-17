"""The per-unit basis report a completed spectra calculation prints.

``calc_spectra``'s work units are transition operators: the photoemission and
inverse-photoemission call sites are the annihilate and create blocks of a ``get_spectra`` run,
and NIXS/XAS reach the same driver with their own operators. The report is opt-in
(``unit_report_label``) because a caller that runs this driver several times for one result
wants one report for the result, not one per call.

The size itself comes from ``_block_green_group``'s ``cap_stats``; what is pinned here is the
per-operator row shape, the symmetry-reduced labelling, and the silence without a label. See
``test/gf/test_gf_unit_basis_report.py`` for the seed-vs-support regression the number rests on.
"""

import numpy as np

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.spectra import calc_spectra

from ..gf.test_gf_unit_basis_report import (
    GROUND,
    HOP,
    NON_BINDING_CAP,
    _report_lines,
    _seed_support_size,
    _sizes,
)


def _basis(cap):
    return Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[GROUND],
        comm=None,
        truncation_threshold=cap,
    )


def _run_spectra(unit_report_label, cap=NON_BINDING_CAP, equivalence_groups=None):
    return calc_spectra(
        HOP,
        [ManyBodyOperator({((0, "a"),): 1.0}), ManyBodyOperator({((1, "a"),): 1.0})],
        [ManyBodyState({SlaterDeterminant.from_bytes(GROUND): 1.0})],
        [0.0],
        1.0,
        np.linspace(-3.0, 3.0, 21),
        _basis(cap),
        0.1,
        0.0,
        False,
        1e-9,
        {0: (3, 3)},
        {0: (0, 0)},
        {0: (0, 0)},
        equivalence_groups=equivalence_groups,
        unit_report_label=unit_report_label,
    )


def test_spectra_driver_reports_one_row_per_transition_operator(capsys):
    """``calc_spectra``'s units are transition operators; the photoemission and
    inverse-photoemission call sites are the annihilate and create blocks of a spectra run."""
    _run_spectra("photoemission, l=2 (annihilate)")
    out = capsys.readouterr().out
    assert "Maximum excited basis size per unit -- photoemission, l=2 (annihilate):" in out
    rows, summary = _report_lines(out)
    assert [row.split()[1] for row in rows] == ["0", "1"]
    assert all(size > _seed_support_size() for size in _sizes(rows).values())
    assert summary is not None


def test_spectra_rows_name_representatives_when_operators_were_symmetry_reduced(capsys):
    """With ``equivalence_groups`` the recursion reduces ``tOps`` to one representative per
    class, so the row index is into the representatives -- not an orbital. Say so, or a reader
    maps "operator 1" to spin-orbital 1."""
    _run_spectra("photoemission (annihilate)", equivalence_groups=["g", "g", "g"])
    rows, _ = _report_lines(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0].startswith("operator 0"), rows

    # Second class first: the surviving row must name operator 1, not position 0.
    _run_spectra("photoemission (annihilate)", equivalence_groups=["a", "b", "b"])
    rows, _ = _report_lines(capsys.readouterr().out)
    assert [row.split()[1] for row in rows] == ["0", "1"], rows


def test_spectra_driver_is_silent_without_a_label(capsys):
    """A caller that runs ``calc_spectra`` several times for one result (the susceptibility
    driver, once per spin sector) passes no label and gets no report."""
    _run_spectra(None)
    assert "Maximum excited basis size per unit" not in capsys.readouterr().out
