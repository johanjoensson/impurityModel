"""Tests for the central knob registry (:mod:`impurityModel.ed.config`)."""

import pytest

from impurityModel.ed import config


def test_defaults_when_unset(monkeypatch):
    """Every knob returns its declared default when its variable is absent."""
    for knob in config.KNOBS.values():
        monkeypatch.delenv(knob.name, raising=False)
    for knob in config.KNOBS.values():
        assert knob.get() == knob.default, knob.name


def test_env_override_is_read_lazily(monkeypatch):
    """A variable set after import is picked up by the next read.

    The whole point of the registry: import-time constants silently voided a slicing test
    once, because a caller that had already imported the module could not change them.
    """
    monkeypatch.delenv("GF_SLICES", raising=False)
    assert config.GF_SLICES.get() == 8
    monkeypatch.setenv("GF_SLICES", "3")
    assert config.GF_SLICES.get() == 3


def test_parsers_and_clamps(monkeypatch):
    """Each kind parses, and the declared minimum clamps."""
    monkeypatch.setenv("GF_BICGSTAB_ATOL", "1e-6")
    assert config.GF_BICGSTAB_ATOL.get() == pytest.approx(1e-6)

    monkeypatch.setenv("GF_SLICES", "0")  # minimum=1
    assert config.GF_SLICES.get() == 1

    monkeypatch.setenv("GF_SLICE_TOL", "-1.0")  # minimum=0.0
    assert config.GF_SLICE_TOL.get() == 0.0

    monkeypatch.setenv("GF_SECTOR_CACHE_DIR", "/tmp/sectors")
    assert config.GF_SECTOR_CACHE_DIR.get() == "/tmp/sectors"


@pytest.mark.parametrize("raw,expected", [("1", True), ("yes", True), ("0", False), ("false", False), ("", False)])
def test_bool_truthiness(monkeypatch, raw, expected):
    """Only the explicit falsehoods (and unset/empty) are false -- the historical convention."""
    monkeypatch.setenv("GF_OPERATOR_SPLIT", raw)
    assert config.GF_OPERATOR_SPLIT.get() is expected


def test_derived_knobs_return_none_when_unset(monkeypatch):
    """A knob with no static default is an override only; the call site derives otherwise."""
    derived = [k for k in config.KNOBS.values() if k.default is None]
    assert derived, "expected at least one derived knob"
    for knob in derived:
        monkeypatch.delenv(knob.name, raising=False)
        assert knob.get() is None, knob.name


def test_empty_string_counts_as_unset_for_non_str(monkeypatch):
    """`GF_RIXS_ADAPTIVE_TOL=` disables the sampler rather than raising on float("")."""
    monkeypatch.setenv("GF_RIXS_ADAPTIVE_TOL", "")
    assert config.GF_RIXS_ADAPTIVE_TOL.get() is None


def test_registry_is_keyed_by_name_and_grouped():
    """Every knob is registered under its own name and in a group `dump` renders."""
    for name, knob in config.KNOBS.items():
        assert name == knob.name
        assert knob.group in config.GROUP_TITLES, f"{name} has an unrendered group {knob.group!r}"


def test_dump_covers_every_knob():
    """The generated configuration table names every declared knob.

    doc/configuration.md is generated from `dump()`, so this is what keeps a newly declared
    knob from being undocumented.
    """
    table = config.dump()
    for name in config.KNOBS:
        assert f"`{name}`" in table, f"{name} missing from dump()"
