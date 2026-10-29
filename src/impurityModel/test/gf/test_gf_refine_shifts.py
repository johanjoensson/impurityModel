"""``g0_zero_shifts``: the scoring shifts of a block, chosen from ``G_0^-1`` before the recurrence runs."""

import numpy as np

from impurityModel.ed.gf_refinement import g0_zero_shifts

EPS = np.array([-0.3, 0.4])
V = np.array([0.2, 0.3])
DELTA = 0.02
IW = np.pi * 0.01 * (2 * np.arange(5) + 1)


def _g0_inv(z):
    z = np.asarray(z)
    return (z - 0.1 - np.sum(V[None] ** 2 / (z[:, None] - EPS[None]), axis=1))[:, None, None]


def _shifts(**kwargs):
    return g0_zero_shifts(_g0_inv, np.linspace(-1, 1, 401), IW, DELTA, **kwargs)


def test_the_shifts_sit_on_the_bath_poles_the_quasiparticle_point_and_the_lowest_matsubara_points():
    zs, weights = _shifts()
    real = [z for z in zs if z.imag == DELTA]
    assert any(abs(z.real - 0.4) < 0.02 for z in real) and any(abs(z.real + 0.3) < 0.02 for z in real)
    assert any(abs(z.real) < 0.01 for z in real)  # omega = 0
    assert [z for z in zs if z.real == 0.0 and z.imag > DELTA] == [1j * IW[0], 1j * IW[1]]
    assert len(zs) == len(weights) <= 8


def test_the_heavier_pole_is_weighted_one_and_the_rest_below():
    zs, weights = _shifts()
    assert max(weights) == 1.0
    heavy = zs[int(np.argmax(weights))]
    assert abs(heavy.real - 0.4) < 0.02  # |V|^2 = 0.09 against 0.04
    assert all(0.0 <= w <= 1.0 for w in weights)


def test_the_count_is_capped_and_a_mesh_without_a_real_axis_gives_matsubara_only():
    zs, _ = _shifts(max_shifts=4)
    assert len(zs) <= 4
    zs, _ = g0_zero_shifts(_g0_inv, None, IW, DELTA)
    assert zs == [1j * IW[0], 1j * IW[1]]
    assert g0_zero_shifts(_g0_inv, None, None, DELTA) == ([], [])
