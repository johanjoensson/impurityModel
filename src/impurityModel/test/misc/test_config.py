"""Tests for the central knob registry (:mod:`impurityModel.ed.config`)."""

from pathlib import Path

import pytest

from impurityModel.ed import config

#: Repo root, for the generated-documentation check. Same walk as
#: ``test/spectra/test_bath_layout.py``'s ``h0`` lookup: test file -> its directory -> test ->
#: impurityModel -> src -> root.
_REPO_ROOT = Path(__file__).resolve().parents[4]


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


def test_every_declared_knob_is_registered():
    """A ``Knob`` assigned at module scope but left out of ``KNOBS`` is invisible to everything.

    ``test_dump_covers_every_knob`` below iterates ``KNOBS``, so it cannot see this: an
    unregistered knob is absent from the registry *and* from the table, and the two agree with
    each other while the knob is silently undocumented -- even though it is fully functional at
    its call site, because ``Knob.get`` reads the environment directly and never consults
    ``KNOBS``. That is exactly what happened to ``DC_CAP_STRATEGY``: declared, wired into
    ``dc_criteria._calibrate_cap``, working, and in no document. This scans the module's own
    globals instead, which is the only place the omission is visible, and makes ``dump``'s
    docstring claim -- "a knob declared here is documented by construction" -- actually true.
    """
    declared = {value.name for value in vars(config).values() if isinstance(value, config.Knob)}
    missing = sorted(declared - set(config.KNOBS))
    assert not missing, f"declared but not in KNOBS (so undocumented and undumpable): {missing}"


def test_dump_covers_every_knob():
    """The generated configuration table names every declared knob.

    doc/configuration.md is generated from `dump()`, so this is what keeps a newly declared
    knob from being undocumented.
    """
    table = config.dump()
    for name in config.KNOBS:
        assert f"`{name}`" in table, f"{name} missing from dump()"


def test_the_generated_configuration_doc_is_in_sync_with_the_registry():
    """``doc/configuration.md`` is generated from ``dump()``, so it must still equal it.

    ``test_dump_covers_every_knob`` above checks the *generator*, not the file, which is why the
    file could and did drift: ``GS_MAX_BLOCK_WIDTH`` and ``SIGMA_CAUSALITY_TOL`` were declared and
    registered but missing from the document until someone noticed and resynced it by hand. A
    knob is only documented by construction if something asserts the construction was run.

    Regenerate with ``python -m impurityModel.ed.config > /tmp/tables.md`` and splice the tables
    in under the document's preamble, which is hand-written and deliberately not checked here.
    """
    doc = _REPO_ROOT / "doc" / "configuration.md"
    if not doc.is_file():
        pytest.skip("running against an installed package without the source tree")
    assert config.dump().strip() in doc.read_text(), (
        "doc/configuration.md no longer matches config.dump(); regenerate the tables from the "
        "registry rather than editing the document."
    )
