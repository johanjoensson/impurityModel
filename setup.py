import os

import setuptools
from setuptools import Extension

include_dirs = []

boost_dir = os.environ.get("BOOST_ROOT") or os.environ.get("BOOST_DIR")
if boost_dir:
    include_dirs.extend([os.path.join(boost_dir, "include"), boost_dir])


try:
    import numpy as np

    include_dirs.append(np.get_include())
except ImportError:
    pass

try:
    import mpi4py

    include_dirs.append(mpi4py.get_include())
except ImportError:
    pass

extra_compile_args = []
cxxflags = os.environ.get("CXXFLAGS", "")
if "-std=c++" not in cxxflags and "-std=gnu++" not in cxxflags:
    extra_compile_args.append("-std=c++17")

# Opt-in multithreaded ManyBodyOperator::apply (off by default). Enable with
# IMPURITYMODEL_PARALLEL=1 pip install -e . --no-build-isolation
# Intended for single-process / few-rank-many-core runs; do NOT combine with one MPI
# rank per core, which oversubscribes the node (each rank spawns its own threads).
extra_link_args = []
if os.environ.get("IMPURITYMODEL_PARALLEL", "").lower() in ("1", "true", "yes", "on"):
    extra_compile_args += ["-DPARALLEL", "-pthread"]
    extra_link_args += ["-pthread"]

_cython_src_dir = "src/cython"
_mpi_utils_src = os.path.join(_cython_src_dir, "MpiUtils.cpp")

ext_modules = [
    Extension(
        name="impurityModel.ed.ManyBodyUtils",
        sources=[
            os.path.join(_cython_src_dir, "ManyBodyUtils.pyx"),
            _mpi_utils_src,
        ],
        depends=[
            os.path.join(_cython_src_dir, "_slater_state.pxi"),
            os.path.join(_cython_src_dir, "_operator.pxi"),
            os.path.join(_cython_src_dir, "_mpi_pack.pxi"),
            os.path.join(_cython_src_dir, "_krylov_store.pxi"),
            os.path.join(_cython_src_dir, "_block_state.pxi"),
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.BlockLanczos",
        sources=[
            os.path.join(_cython_src_dir, "BlockLanczos.pyx"),
        ],
        depends=[
            os.path.join(_cython_src_dir, "_lanczos_step.pxi"),
            os.path.join(_cython_src_dir, "_trlm.pxi"),
            os.path.join(_cython_src_dir, "_irlm.pxi"),
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.BlockLanczosArray",
        sources=[
            os.path.join(_cython_src_dir, "BlockLanczosArray.pyx"),
        ],
        depends=[
            os.path.join(_cython_src_dir, "_reort.pxi"),
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.BiCGSTAB",
        sources=[
            os.path.join(_cython_src_dir, "BiCGSTAB.pyx"),
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.GMRES",
        sources=[
            os.path.join(_cython_src_dir, "GMRES.pyx"),
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.ChebyshevFilter",
        sources=[
            os.path.join(_cython_src_dir, "ChebyshevFilter.pyx"),
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setuptools.setup(ext_modules=ext_modules)
