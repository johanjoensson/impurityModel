#!/bin/bash -e
# Ni in NiO, 50 valence + 10 conduction bath states.
#
# The calculation is now described by an input file; this script only launches it. Everything
# that used to be a command-line flag lives in examples/NiO_50p10bath_spectra.toml, where it is named, documented
# and validated -- run `impurityModel run examples/NiO_50p10bath_spectra.toml --show-resolved` to see every
# value the solver will actually use.

ranks=${1:-1}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# KNOWN BROKEN: this workload does not currently solve -- the ground state comes back as the
# whole 45-fold-degenerate d8 manifold at E0 = 0 and the spectra stage is OOM-killed. The same
# happens from the old command line, so it is not the input file. See the header of the .toml,
# and use examples/NiO_10bath_spectra.toml for a working fitted-bath run.

mpirun -n "$ranks" impurityModel run "${DIR}/../examples/NiO_50p10bath_spectra.toml"
