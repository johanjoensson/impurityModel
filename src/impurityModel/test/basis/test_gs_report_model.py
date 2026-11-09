"""``assemble_report``: manifolds, members and the JSON, from replicated per-state arrays alone."""

import json

import numpy as np

from impurityModel.ed.gs_report_model import assemble_report
from impurityModel.ed.gs_statistics import _NumpyEncoder


def test_a_triplet_and_a_singlet_become_two_manifolds_with_member_rows():
    es = np.array([1.0, 0.0, 0.0, 0.0])  # unsorted on purpose: the report orders by energy
    rep = assemble_report(
        es,
        0.5,
        100,
        two_sz=[0, -2, 0, 2],
        s_values=[0.0, 1.0, 1.0, 1.0],
        sisb_values=[0.1, 0.2, 0.3, 0.4],
    )
    assert [m.g for m in rep.manifolds] == [3, 1]
    triplet, singlet = rep.manifolds
    assert triplet.states == [0, 1, 2] and triplet.s_sharp is True
    assert sorted(triplet.two_sz) == [-2, 0, 2]
    assert np.isclose(triplet.weight + singlet.weight, 1.0) and triplet.weight > singlet.weight
    assert rep.e0 == 0.0 and [m.manifold for m in rep.members] == [0, 0, 0, 1]
    assert rep.members[3].e_minus_e0 == 1.0 and rep.members[3].sisb == 0.1


def test_a_manifold_mixing_labels_is_not_sharp_and_the_report_serialises():
    rep = assemble_report(np.zeros(2), 0.1, 5, s_values=[0.0, 1.0], l_values=[2.0, 2.0])
    (m,) = rep.manifolds
    assert m.s_sharp is False and m.l_sharp is True and m.two_sz is None
    data = json.loads(json.dumps(rep.to_json(), cls=_NumpyEncoder))
    assert data["schema_version"] == 4 and data["manifolds"][0]["g"] == 2 and data["naming"] is None


def test_manifold_entropy_is_attached_to_the_manifold_it_belongs_to():
    es = np.array([1.0, 0.0, 0.0])  # the degenerate pair is at positions 1, 2 of ``es``
    rep = assemble_report(
        es, 0.5, 5, manifold_entropy=[{"states": [1, 2], "entropy": 0.69}, {"states": [0], "entropy": 0.1}]
    )
    assert [m.entropy for m in rep.manifolds] == [0.69, 0.1]
