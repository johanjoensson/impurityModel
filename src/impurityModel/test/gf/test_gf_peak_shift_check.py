"""The per-unit peak-shift check: how far G at the Sigma-critical shifts still moves over the recurrence's tail."""

import numpy as np

from impurityModel.ed.gf_recurrence import peak_shift_check
from impurityModel.ed.gf_units import format_unit_timing


def _chain(n, seed=0):
    rng = np.random.default_rng(seed)
    a = [np.array([[x]], dtype=complex) for x in rng.normal(size=n)]
    b = [np.array([[x]], dtype=complex) for x in 0.5 + 0.1 * rng.random(n)]
    return a, b


def _dense_g(a, b, z, n):
    t = np.zeros((n, n), dtype=complex)
    for i in range(n):
        t[i, i] = a[i][0, 0]
        if i + 1 < n:
            t[i, i + 1] = t[i + 1, i] = b[i][0, 0]
    return np.linalg.inv(z * np.eye(n) - t)[0, 0]


def test_g_max_and_tail_change_match_a_dense_resolvent():
    n = 40
    a, b = _chain(n)
    zs = [0.3 + 0.05j, -0.2 + 0.05j]
    out = peak_shift_check(a, b, np.eye(1, dtype=complex), zs)
    keep = int(0.9 * n)
    g_all = np.array([_dense_g(a, b, z, n) for z in zs])
    g_cut = np.array([_dense_g(a, b, z, keep) for z in zs])
    assert np.isclose(out["tail_change"], (np.abs(g_all - g_cut) / np.abs(g_all)).max(), rtol=1e-9)
    assert out["n_shifts"] == 2 and out["n_dropped"] == n - keep


def test_a_recurrence_that_still_moves_reports_a_larger_tail_change_than_a_settled_one():
    # injected defect: a tail that keeps a large coupling must not look settled
    a, b = _chain(40)
    r = np.eye(1, dtype=complex)
    settled = peak_shift_check(a, [x * (0.01 if i > 30 else 1) for i, x in enumerate(b)], r, [0.1 + 0.05j])
    moving = peak_shift_check(a, [x * (4.0 if i > 30 else 1) for i, x in enumerate(b)], r, [0.1 + 0.05j])
    assert moving["tail_change"] > 10 * settled["tail_change"]


def test_too_short_no_shifts_or_complete_report_nan_not_an_error():
    a, b = _chain(5)  # below PEAK_SHIFT_MIN_BLOCKS: the last block is not a tail
    out = peak_shift_check(a, b, np.eye(1, dtype=complex), [0.1j])
    assert np.isnan(out["tail_change"]) and out["n_dropped"] == 0
    a, b = _chain(40)
    assert peak_shift_check(a, b, np.eye(1, dtype=complex), [])["n_shifts"] == 0
    out = peak_shift_check(a, b, np.eye(1, dtype=complex), [0.1j], complete=True)
    assert out["complete"] and np.isnan(out["tail_change"])


def test_nonfinite_coefficients_give_nan_and_never_raise():
    a, b = _chain(40)
    b[30] = b[30] * np.nan
    assert np.isnan(peak_shift_check(a, b, np.eye(1, dtype=complex), [0.1 + 0.05j])["tail_change"])


def test_shrinking_blocks_with_rectangular_betas_match_a_dense_resolvent():
    # widths 2,2,...,2,1,1: deflation leaves rectangular off-diagonal blocks inside the kept run
    rng = np.random.default_rng(3)
    widths = [2] * 24 + [1] * 6
    n = len(widths)
    a = []
    for w in widths:
        m = rng.normal(size=(w, w)) + 1j * rng.normal(size=(w, w))
        a.append(m + m.conj().T)
    b = [
        0.4 * (rng.normal(size=(widths[i + 1] if i + 1 < n else 1, w)) + 0j)
        + 0.6 * np.eye(widths[i + 1] if i + 1 < n else 1, w)
        for i, w in enumerate(widths)
    ]
    r = np.eye(2, dtype=complex)
    z = np.array([0.2 + 0.05j])

    def dense(k):
        off = np.cumsum([0] + widths[:k])
        t = np.zeros((off[-1], off[-1]), dtype=complex)
        for i in range(k):
            sl = slice(off[i], off[i + 1])
            t[sl, sl] = a[i]
            if i + 1 < k:
                nxt = slice(off[i + 1], off[i + 2])
                t[nxt, sl] = b[i]
                t[sl, nxt] = b[i].conj().T
        return np.linalg.inv(z[0] * np.eye(off[-1]) - t)[:2, :2]

    keep = int(0.9 * n)
    out = peak_shift_check(a, b, r, z)
    g_all, g_cut = dense(n), dense(keep)
    assert np.isclose(out["tail_change"], np.abs(g_all - g_cut).max() / np.abs(g_all).max(), rtol=1e-6)


def test_the_unit_line_names_the_check_in_words():
    passes = dict(
        growth_blocks=10,
        growth_seconds=1.0,
        restart_blocks=9,
        restart_seconds=0.5,
        peak_shifts={"n_shifts": 7, "n_dropped": 4, "tail_change": 4.5e-6, "complete": False},
    )
    _, line = format_unit_timing([(0, 0.0, 2.0, passes)], [1])
    assert "changes by at most 4.5e-06 of its size without the last 4 Lanczos steps" in line
    assert "not the determinant cap" in line
    passes["peak_shifts"]["complete"] = True
    assert "Krylov space complete" in format_unit_timing([(0, 0.0, 2.0, passes)], [1])[1]
    passes = dict(passes, peak_shifts=None, peak_shifts_note="stacked thermal states")
    assert "not checked (stacked thermal states)" in format_unit_timing([(0, 0.0, 2.0, passes)], [1])[1]
