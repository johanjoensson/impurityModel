import subprocess
import os
import tempfile
import pytest

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CYTHON_DIR = os.path.join(DIR_PATH, "../../cython")


def test_cpp_compilation_and_execution():
    # Helper to run shell commands in CYTHON_DIR
    def run_cmd(args):
        subprocess.run(args, cwd=CYTHON_DIR, check=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Compile and test Default (no flags)
        default_bin = os.path.join(tmpdir, "test_cpp_default")
        run_cmd([
            "g++", "-O3", "-std=c++17",
            "test_cpp_compile.cpp", "ManyBodyState.cpp", "ManyBodyOperator.cpp",
            "-o", default_bin
        ])
        res = subprocess.run([default_bin], capture_output=True, text=True, check=True)
        assert "All C++ unit tests passed successfully!" in res.stdout

        # 2. Compile and test with -DBOOST
        boost_bin = os.path.join(tmpdir, "test_cpp_boost")
        run_cmd([
            "g++", "-O3", "-std=c++17", "-DBOOST",
            "test_cpp_compile.cpp", "ManyBodyState.cpp", "ManyBodyOperator.cpp",
            "-o", boost_bin
        ])
        res = subprocess.run([boost_bin], capture_output=True, text=True, check=True)
        assert "All C++ unit tests passed successfully!" in res.stdout

        # 3. Compile and test with -DPARALLEL
        parallel_bin = os.path.join(tmpdir, "test_cpp_parallel")
        run_cmd([
            "g++", "-O3", "-std=c++17", "-DPARALLEL", "-pthread",
            "test_cpp_compile.cpp", "ManyBodyState.cpp", "ManyBodyOperator.cpp",
            "-o", parallel_bin
        ])
        res = subprocess.run([parallel_bin], capture_output=True, text=True, check=True)
        assert "All C++ unit tests passed successfully!" in res.stdout

        # 4. Compile and test with both -DBOOST and -DPARALLEL
        both_bin = os.path.join(tmpdir, "test_cpp_both")
        run_cmd([
            "g++", "-O3", "-std=c++17", "-DBOOST", "-DPARALLEL", "-pthread",
            "test_cpp_compile.cpp", "ManyBodyState.cpp", "ManyBodyOperator.cpp",
            "-o", both_bin
        ])
        res = subprocess.run([both_bin], capture_output=True, text=True, check=True)
        assert "All C++ unit tests passed successfully!" in res.stdout
