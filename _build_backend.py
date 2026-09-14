import os

from setuptools.build_meta import *
from setuptools.build_meta import (
    get_requires_for_build_editable as _original_get_requires_for_build_editable,
    get_requires_for_build_sdist as _original_get_requires_for_build_sdist,
    get_requires_for_build_wheel as _original_get_requires_for_build_wheel,
)


def is_boost_available():
    """Check if Boost is provided by the system/environment."""
    boost_dir = os.environ.get("BOOST_ROOT") or os.environ.get("BOOST_DIR")
    if boost_dir:
        return True

    system_paths = [
        "/usr/include/boost",
        "/usr/local/include/boost",
        "/opt/homebrew/include/boost",
        "/opt/local/include/boost",
    ]
    return any(os.path.exists(path) for path in system_paths)


def get_requires_for_build_wheel(config_settings=None):
    reqs = _original_get_requires_for_build_wheel(config_settings)
    if not is_boost_available():
        reqs.append("boost-headers")
    return reqs


def get_requires_for_build_sdist(config_settings=None):
    reqs = _original_get_requires_for_build_sdist(config_settings)
    if not is_boost_available():
        reqs.append("boost-headers")
    return reqs


def get_requires_for_build_editable(config_settings=None):
    reqs = _original_get_requires_for_build_editable(config_settings)
    if not is_boost_available():
        reqs.append("boost-headers")
    return reqs
