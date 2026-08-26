#!/bin/bash -e
# Ni in NiO, fitted bath.
#
# The calculation is now described by an input file; this script only launches it. Everything
# that used to be a command-line flag lives in examples/NiO_10bath_spectra.toml, where it is named, documented
# and validated -- run `impurityModel run examples/NiO_10bath_spectra.toml --show-resolved` to see every
# value the solver will actually use.

ranks=${1:-1}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mpirun -n "$ranks" impurityModel run "${DIR}/../examples/NiO_10bath_spectra.toml"

# This script used to take the bath count as its first argument. The count is now two
# numbers in the input file (n_bath / n_valence_bath) alongside the h0 file they must
# match, so changing it is an edit to one file rather than an argument that has to agree
# with a filename. Format 1.0 has no sweep mechanism; that is deliberate.
