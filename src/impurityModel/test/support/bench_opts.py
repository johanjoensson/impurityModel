"""Options of the opt-in benchmarks and golden baselines, given on the pytest command line.

``--bench-opt KEY=VALUE`` (repeatable) sets a benchmark parameter, read with :func:`bench_opt`;
``--regen-golden`` rewrites the stored golden baselines. The benchmarks themselves are the tests marked
``benchmark``: select them with ``-m benchmark`` (and ``-k`` for one), they are deselected by default.
Registered in ``test/conftest.py``.
"""

_OPTS: dict = {}
_FLAGS = {"regen": False}


def configure(pairs, regen):
    """Called once by ``pytest_configure``; ``pairs`` is the list of ``KEY=VALUE`` strings."""
    _OPTS.clear()
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--bench-opt expects KEY=VALUE, got {pair!r}")
        _OPTS[key.strip()] = value
    _FLAGS["regen"] = bool(regen)


def bench_opt(key, default=None):
    """The value given as ``--bench-opt KEY=VALUE`` (a string), else ``default``."""
    return _OPTS.get(key, default)


def regen_golden():
    """Whether ``--regen-golden`` was given."""
    return _FLAGS["regen"]
