#!/bin/bash -e
# Fe in FeO, crystal-field parametrisation.
#
# The calculation is now described by an input file; this script only launches it. Everything
# that used to be a command-line flag lives in examples/FeO_CF_spectra.toml, where it is named, documented
# and validated -- run `impurityModel run examples/FeO_CF_spectra.toml --show-resolved` to see every
# value the solver will actually use.

ranks=${1:-6}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mpirun -n "$ranks" impurityModel run "${DIR}/../examples/FeO_CF_spectra.toml"
