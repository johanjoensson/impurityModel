#!/bin/bash -e

# Number of MPI ranks to use.
# Check if the first input parameter is empty.
if [[ -z "$1" ]]; then
    ranks=6
else
    ranks=$1
fi

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Mno in MnO
# Non-interacting Hamiltonian constructed from CF-parameters.

# Filename of the non-relativistic non-interacting Hamiltonian
h0_filename="${DIR}/../h0/h0_MnO_CF.json"
# Filename of the radial part of the correlated orbitals.
radial_filename=${DIR}"/../radialOrbitals/Mn3d.dat"

echo "H0 filename: $h0_filename"
echo "Radial wavefunction filename: $radial_filename"

mpirun -n $ranks python -m impurityModel.ed.get_spectra $h0_filename $radial_filename \
    --n0imps 6 5 --Fdd 6 0 9.0 0 6.1 --Fpd 7.5 0 5.6 --Gpd 0 4 0 2.3 --xi_2p 6.936 --xi_3d 0.051 --nPsiMax 7
