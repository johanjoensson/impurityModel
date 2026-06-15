import setuptools
from setuptools import Extension
import os

include_dirs = []

boost_dir = os.environ.get("BOOST_ROOT") or os.environ.get("BOOST_DIR")
if boost_dir:
    include_dirs.extend([os.path.join(boost_dir, "include"), boost_dir])


try:
    import numpy as np
    include_dirs.append(np.get_include())
except ImportError:
    pass

extra_compile_args = []
cxxflags = os.environ.get("CXXFLAGS", "")
if "-std=c++" not in cxxflags and "-std=gnu++" not in cxxflags:
    extra_compile_args.append("-std=c++17")

ext_modules = [
    Extension(
        name="impurityModel.ed.ManyBodyUtils",
        sources=["src/cython/ManyBodyUtils.pyx", "src/cython/MpiUtils.cpp"],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
]

setuptools.setup(
    ext_modules=ext_modules
)
