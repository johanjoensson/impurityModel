"""GS statistics JSON v3: the version is stamped and ``truncation`` carries ``cap_bound``."""

import json

from impurityModel.ed.gs_statistics import GS_STATISTICS_SCHEMA_VERSION, save_gs_statistics


def test_version_three_is_stamped_and_the_cap_bound_key_round_trips(tmp_path):
    assert GS_STATISTICS_SCHEMA_VERSION == 3
    path = tmp_path / "gs.json"
    save_gs_statistics({"truncation": {"cap_hit": True, "cap_bound": False}}, path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == 3
    assert data["truncation"]["cap_bound"] is False
