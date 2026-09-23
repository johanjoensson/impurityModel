"""The stable external surface: ``impurityModel.api`` and ``python -m impurityModel``.

``api`` is what RSPt's ``impurityModel_interface`` imports, and it had no test at all -- a name
dropped or renamed behind it would surface first as an ImportError in someone else's DMFT run.
"""

import runpy
import sys

import pytest

from impurityModel.ed.get_spectra import _resolve_shell_roles


def test_every_name_the_api_exports_resolves():
    from impurityModel import api

    assert len(api.__all__) == len(set(api.__all__)), "duplicate names in api.__all__"
    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert not missing, f"api.__all__ names that do not resolve: {missing}"
    assert isinstance(api.__version__, str) and api.__version__


def test_python_dash_m_reaches_the_umbrella_cli(monkeypatch, capsys):
    """``python -m impurityModel schema`` -- run in-process, so it works under mpiexec too."""
    # runpy warns (an error under filterwarnings = error) if the module is already imported.
    monkeypatch.delitem(sys.modules, "impurityModel.__main__", raising=False)
    monkeypatch.setattr(sys, "argv", ["impurityModel", "schema"])
    runpy.run_module("impurityModel", run_name="__main__")
    assert "[many_body_basis]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "n_baths, roles",
    [
        ({2: 10}, (2, None)),  # a single shell is the valence shell, bath or not
        ({2: 0}, (2, None)),
        ({1: 0, 2: 10}, (2, 1)),  # the core shell is the one without a bath
        ({2: 10, 1: 0}, (2, 1)),  # ... whatever order the shells come in
    ],
)
def test_shell_roles_are_read_from_the_bath_counts(n_baths, roles):
    assert _resolve_shell_roles(n_baths) == roles


@pytest.mark.parametrize(
    "n_baths",
    [
        {1: 0, 2: 0},  # Hubbard-I valence + core: the case the rule would get backwards
        {1: 6, 2: 10},  # two bathed shells
        {0: 0, 1: 0, 2: 10},  # two candidate core shells
    ],
)
def test_an_ambiguous_shell_layout_is_refused(n_baths):
    with pytest.raises(ValueError, match="Cannot tell which shell is the core one"):
        _resolve_shell_roles(n_baths)
