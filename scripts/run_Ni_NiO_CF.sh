#!/bin/bash -e
# Ni in NiO, crystal-field parametrisation.
#
# The calculation is now described by an input file; this script only launches it. Everything
# that used to be a command-line flag lives in examples/NiO_CF_spectra.toml, where it is named, documented
# and validated -- run `impurityModel run examples/NiO_CF_spectra.toml --show-resolved` to see every
# value the solver will actually use.

ranks=${1:-1}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mpirun -n "$ranks" impurityModel run "${DIR}/../examples/NiO_CF_spectra.toml"
