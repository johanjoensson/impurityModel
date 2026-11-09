"""The compact ground-state summary renderer."""

from impurityModel.ed.gs_report_model import assemble_report
from impurityModel.ed.gs_report_render import fmt, render_summary


def _report(**kw):
    return assemble_report(
        [-1.0, -1.0, -1.0, -0.9],
        0.01,
        120,
        two_sz=[-2, 0, 2, 0],
        s_values=[1.0, 1.0, 1.0, 0.0],
        l_values=[0.0] * 4,
        thermal={"chi_spin_zz": 3.21, "sisb": -0.0000001},
        correlation={"docc_open_shell_total": 0.135},
        magnetic={"sz": -0.0, "lz": 0.0, "s2": 2.0, "l2": 0.0, "term": "3F", "mu_spin_only": 2.83},
        convergence={"converged": True, "residual_pt2": 1e-9, "e_pt2_tol": 1e-8},
        warnings=["something degraded"],
        **kw,
    )


def test_fmt_never_prints_negative_zero():
    assert fmt(-1e-9) == "0.000"
    assert fmt(-1e-9, 4, signed=True) == "+0.0000"
    assert fmt(None) == "-"


def test_summary_is_one_screen_and_deterministic():
    lines = render_summary(_report(), unit="Ni")
    assert lines == render_summary(_report(), unit="Ni")
    assert len(lines) <= 25
    text = "\n".join(lines)
    assert "-0.000" not in text and "\n\n\n" not in text
    assert "Ground state: Ni" in text
    assert "2 manifolds" in text and "-2, 0, +2" in text  # the triplet is one row, tagged by 2Sz
    assert "chi_spin_zz (Curie) = 3.2" in text
    assert "WARNING: something degraded" in text
    assert "converged" in text
