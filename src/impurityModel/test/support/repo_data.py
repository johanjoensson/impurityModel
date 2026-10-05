"""Where the repository's data files (``h0/``) are, and what to do when there is no repository.

The tests ship inside the package, but ``h0/`` lives at the repository root and is not installed.
From a source checkout ``src/impurityModel/test/<dir>/<file>`` is four levels below the root; from
site-packages the same walk lands in ``lib/pythonX.Y``, where there is no ``h0/``. A test that needs
those files skips there instead of failing on a missing path.

The skip is keyed on *being a checkout* (a ``pyproject.toml`` at the root), not on the file
existing: in a checkout a missing Hamiltonian stays a hard failure, so deleting ``h0/`` cannot turn
the CI run silently green.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def in_source_checkout():
    """True when the tests run from a repository checkout rather than an installed package."""
    return (REPO_ROOT / "pyproject.toml").is_file()


def repo_h0_dir():
    """The repository's ``h0/`` directory; skips the calling test when run from an installed package."""
    if not in_source_checkout():
        pytest.skip("the repository h0/ data is not part of an installed package")
    return REPO_ROOT / "h0"
