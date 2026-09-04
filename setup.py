import os

import setuptools
from setuptools import Extension

from Cython.Build import cythonize

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

# ---------------------------------------------------------------------------
# Build mode: IMPURITYMODEL_BUILD = release (default) | debug | safe
# ---------------------------------------------------------------------------
# release  every optimization that is actually worth having: -O3, -march=native and
#          fast-math's reassociation, but NOT -ffinite-math-only (see below -- it
#          deletes NaN guards and measured no faster). Fastest of the three, and the
#          deliberate default.
# debug    Cython's runtime checks on (bounds, initialized memoryviews) at -O1 -g.
#          The only build that catches an out-of-bounds kernel access: with
#          boundscheck=False such a read returns whatever is there -- measured
#          1.429631004289123e-309, a subnormal that looks like a plausible small
#          number rather than an error.
# safe     optimized but numerically conservative: no -march=native, no
#          -ffast-math. Use for cluster deployment across heterogeneous nodes,
#          or to reproduce pre-2026-09 numerics.
_BUILD_MODE = os.environ.get("IMPURITYMODEL_BUILD", "release").strip().lower()
if _BUILD_MODE not in ("release", "debug", "safe"):
    raise SystemExit(f"IMPURITYMODEL_BUILD must be release, debug or safe (got {_BUILD_MODE!r})")
_CHECKED = _BUILD_MODE == "debug"

extra_compile_args = []
cxxflags = os.environ.get("CXXFLAGS", "")
if "-std=c++" not in cxxflags and "-std=gnu++" not in cxxflags:
    extra_compile_args.append("-std=c++17")

# Set the optimization level explicitly rather than inheriting it. Without this the
# extensions are compiled at whatever level the *interpreter* was built with (-O3 on
# this Fedora build, commonly -O2 on manylinux and CI images), so the same source
# produced differently optimized binaries depending on where it was installed.
if _CHECKED:
    extra_compile_args += ["-O1", "-g", "-fno-omit-frame-pointer"]
elif _BUILD_MODE == "safe":
    extra_compile_args += ["-O3", "-funroll-loops", "-fno-semantic-interposition"]
else:
    # -march/-mtune=native: the binary is then valid only for this CPU. Building per
    # node on a heterogeneous cluster gives different instruction selection (FMA,
    # AVX-512) and therefore different rounding, which breaks the bitwise-identical-
    # across-ranks contract cipsi_solver.py:1129 relies on -- ranks that disagree do
    # not return different answers, they enter an Allreduce with different shapes.
    # Within one build on one machine every rank runs identical code, so this is safe
    # for a single node and for CI, and is why `safe` exists for anything else.
    #
    # This is `-ffast-math` minus `-ffinite-math-only`, and the omission is deliberate and
    # measured. Plain -ffast-math tells the compiler NaN cannot occur, so it deletes NaN
    # tests -- including TSQR's corruption guard. Measured with -ffast-math:
    # test_tsqr_mpi_breakdown_and_corruption_agree poisons one rank's rows with np.nan and
    # asserts every rank reports -1; it returned 3, a healthy rank-3 factorization. A NaN
    # entering the Krylov basis would then propagate silently, in the very component whose
    # bitwise-identical R the rank-invariance argument rests on. It also stopped BiCGSTAB
    # converging to atol=1e-9 (test_block_bicgstab_info_cold_solve, deterministic 5/5).
    #
    # And it bought nothing. On a full solver+GF workload (the calc_selfenergy oracle,
    # two reps each): -O3 alone 24.2 s, full -ffast-math 22.9 s, this subset 22.2 s. The
    # subset is at worst equal and here slightly faster, so -ffinite-math-only is a
    # strictly dominated option -- it removes a safety check for no speed.
    #
    # Everything else fast-math implies is kept: -fno-math-errno, -fno-trapping-math,
    # -fassociative-math (the reassociation that lets the vectorizer work) and
    # -freciprocal-math. Not added: -fcx-limited-range, which changes complex
    # multiplication and division -- this code is complex-valued throughout.
    extra_compile_args += [
        "-O3",
        "-march=native",
        "-mtune=native",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-fassociative-math",
        "-freciprocal-math",
        "-funroll-loops",
        "-fno-semantic-interposition",
    ]

# Opt-in multithreaded ManyBodyOperator::apply (off by default). Enable with
# IMPURITYMODEL_PARALLEL=1 pip install -e . --no-build-isolation
# Intended for single-process / few-rank-many-core runs; do NOT combine with one MPI
# rank per core, which oversubscribes the node (each rank spawns its own threads).
#
# Note what is deliberately absent: -ffast-math is a *compile* flag only and must never
# be added here. Passing it at link time pulls in crtfastmath.o, which sets FTZ/DAZ for
# the whole process -- silently changing numpy, scipy and LAPACK behaviour, not just
# ours. That is the long-standing numpy footgun with fast-math extension modules.
extra_link_args = []
if os.environ.get("IMPURITYMODEL_PARALLEL", "").lower() in ("1", "true", "yes", "on"):
    extra_compile_args += ["-DPARALLEL", "-pthread"]
    extra_link_args += ["-pthread"]

_cython_src_dir = "src/cython"
_mpi_utils_src = os.path.join(_cython_src_dir, "MpiUtils.cpp")

# ---------------------------------------------------------------------------
# Cython directives -- centralized here, deliberately
# ---------------------------------------------------------------------------
# These used to live in a `# cython:` header in each .pyx. A file-level header
# *overrides* whatever cythonize() is given, so with them there no build mode could
# turn the checks back on; that is why they moved. If you need to know what a kernel
# is compiled with, this dict is the answer for all of them.
_DIRECTIVES = {
    "language_level": "3",
    "freethreading_compatible": True,
    # -- checks: the only entries `debug` flips --
    "boundscheck": _CHECKED,
    "initializedcheck": _CHECKED,
    # -- semantics: identical in every mode and every module, on purpose --
    # wraparound=False everywhere. It is not merely an optimization: under it Cython
    # does not add the length to a negative index, so `x[-1]` reads out of bounds --
    # measured, a one-line extension doing `xs[-1]` on a 3-element *list* segfaults
    # (exit 139) when built this way. Spell it `x[len(x) - 1]`, the idiom the kernels
    # already use; CLAUDE.md records this having landed twice.
    # cdivision stays True: turning it off would change negative-integer division from
    # C to Python semantics, which is a behaviour change and not a check. Division by
    # zero is better caught by UBSan, which does not alter results.
    "wraparound": False,
    "cdivision": True,
    # -- optimizations, now applied uniformly --
    # cpow and always_allow_keywords were set on only two of the eight modules, with
    # no reason recorded for the split; both are wins everywhere.
    "cpow": True,
    "always_allow_keywords": False,
}
# Not set, and each for a reason: `binding=False` would be faster but Sphinx autodoc
# introspects these modules (doc/sphinx/_doc_build/impurityModel.ed.rst), and
# `infer_types=True` can silently change an inferred integer's width and overflow
# behaviour.


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
        ],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.BlockLanczosCore",
        sources=[
            os.path.join(_cython_src_dir, "BlockLanczosCore.pyx"),
        ],
        depends=[
            os.path.join(_cython_src_dir, "_block_ops.pxi"),
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
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="impurityModel.ed.TSQR",
        sources=[
            os.path.join(_cython_src_dir, "TSQR.pyx"),
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

# cythonize decides staleness from source timestamps and does **not** notice that the
# directives changed, so switching IMPURITYMODEL_BUILD reuses the previously generated
# C++ and silently gives you the old mode. Verified: after a debug build, a release build
# left TSQR.cpp untouched and still carrying its bounds checks. Record the mode and force
# a re-cythonize when it changes -- only then, so the ordinary incremental build (same
# mode, one edited .pyx) stays fast.
_MODE_STAMP = os.path.join(_cython_src_dir, ".build-mode")
try:
    with open(_MODE_STAMP) as _fh:
        _previous_mode = _fh.read().strip()
except OSError:
    _previous_mode = None

setuptools.setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives=_DIRECTIVES,
        force=_previous_mode != _BUILD_MODE,
    )
)

with open(_MODE_STAMP, "w") as _fh:
    _fh.write(_BUILD_MODE + "\n")
