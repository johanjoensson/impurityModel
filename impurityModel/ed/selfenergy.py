import numpy as np
from collections import OrderedDict
from mpi4py import MPI
import pickle
import time
import argparse
import h5py

from impurityModel.ed.get_spectra.py import get_hamiltonian_operator, get_h0_operator
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.average import k_B

def get_selfenergy():
    omega = np.linspace(-25, 25, 100001)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank == 0:
        t0 = time.time()
    # -- System information --
    nBaths = OrderedDict(zip(ls, nBaths))
    nValBaths = OrderedDict(zip(ls, nValBaths))

    # -- Basis occupation information --
    n0imps = OrderedDict(zip(ls, n0imps))
    dnTols = OrderedDict(zip(ls, dnTols))
    dnValBaths = OrderedDict(zip(ls, dnValBaths))
    dnConBaths = OrderedDict(zip(ls, dnConBaths))

    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B * T

    # -- Occupation restrictions for excited states --
    l = 2
    restrictions = {}
    # Restriction on impurity orbitals
    indices = frozenset(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
    restrictions[indices] = (n0imps[l] - 1, n0imps[l] + dnTols[l] + 1)
    # Restriction on valence bath orbitals
    indices = []
    for b in range(nValBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (nValBaths[l] - dnValBaths[l], nValBaths[l])
    # Restriction on conduction bath orbitals
    indices = []
    for b in range(nValBaths[l], nBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (0, dnConBaths[l])
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0:
        print("#spin-orbitals:", n_spin_orbitals)

    # Hamiltonian
    if rank == 0:
        print("Construct the Hamiltonian operator...")
    hOp = get_hamiltonian_operator(
        nBaths,
        nValBaths,
        [Fdd, Fpp, Fpd, Gpd],
        [0, xi_3d],
        [n0imps, chargeTransferCorrection],
        hField,
        h0_filename,
    )
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0:
        print("{:d} processes in the Hamiltonian.".format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0:
        print("Create basis...")
    basis = finite.get_basis(nBaths, nValBaths, dnValBaths, dnConBaths, dnTols, n0imps)
    if rank == 0:
        print("#basis states = {:d}".format(len(basis)))
    # Diagonalization of restricted active space Hamiltonian
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax)

    if rank == 0:
        print("time(ground_state) = {:.2f} seconds \n".format(time.time() - t0))
        t0 = time.time()

    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0:
        print("#spin-orbitals:", n_spin_orbitals)

    # Print Slater determinants and weights
    if rank == 0:
        print("Slater determinants/product states and correspoinding weights")
        weights = []
        for i, psi in enumerate(psis):
            print("Eigenstate {:d}.".format(i))
            print("Consists of {:d} product states.".format(len(psi)))
            ws = np.array([abs(a) ** 2 for a in psi.values()])
            s = np.array(list(psi.keys()))
            j = np.argsort(ws)
            ws = ws[j[-1::-1]]
            s = s[j[-1::-1]]
            weights.append(ws)
            if nPrintSlaterWeights > 0:
                print("Highest (product state) weights:")
                print(ws[:nPrintSlaterWeights])
                print("Corresponding product states:")
                print(s[:nPrintSlaterWeights])
                print("")

    # Calculate density matrix
    if rank == 0:
        print("Density matrix (in cubic harmonics basis):")
        for i, psi in enumerate(psis):
            print("Eigenstate {:d}".format(i))
            n = finite.getDensityMatrixCubic(nBaths, psi)
            print("#density matrix elements: {:d}".format(len(n)))
            for e, ne in n.items():
                if abs(ne) > tolPrintOccupation:
                    if e[0] == e[1]:
                        print(
                            "Diagonal: (i,s) =",
                            e[0],
                            ", occupation = {:7.2f}".format(ne),
                        )
                    else:
                        print("Off-diagonal: (i,si), (j,sj) =", e, ", {:7.2f}".format(ne))
            print("")
    # Save some information to disk
    if rank == 0:
        # Most of the input parameters. Dictonaries can be stored in this file format.
        np.savez_compressed(
            "data",
            ls=ls,
            nBaths=nBaths,
            nValBaths=nValBaths,
            n0imps=n0imps,
            dnTols=dnTols,
            dnValBaths=dnValBaths,
            dnConBaths=dnConBaths,
            Fdd=Fdd,
            Fpp=Fpp,
            xi_3d=xi_3d,
            chargeTransferCorrection=chargeTransferCorrection,
            hField=hField,
            h0_filename=h0_filename,
            nPsiMax=nPsiMax,
            T=T,
            energy_cut=energy_cut,
            delta=delta,
            restrictions=restrictions,
            n_spin_orbitals=n_spin_orbitals,
            hOp=hOp,
        )
        # Save some of the arrays.
        # HDF5-format does not directly support dictonaries.
        h5f = h5py.File("spectra.h5", "w")
        h5f.create_dataset("E", data=es)
        h5f.create_dataset("w", data=omega)
    else:
        h5f = None
    if rank == 0:
        print("time(expectation values) = {:.2f} seconds \n".format(time.time() - t0))

    # Consider from now on only eigenstates with low energy
    es = tuple(e for e in es if e - es[0] < energy_cut)
    psis = tuple(psis[i] for i in range(len(es)))
    if rank == 0:
        print("Consider {:d} eigenstates for the spectra \n".format(len(es)))

    gs = np.zeros((len(es), len(w)), dtype=np.complex)
    h = {}
    for e_index, e in enumerate(es):
        gs[e_index, :] = getGreen(n_spin_orbitals = n_spin_orbitals,
                                  e = e,
                                  psi = psis(e_index),
                                  h0p = h0p,
                                  omega = omega,
                                  delta = delta,
                                  krylovSize = krylovsize,
                                  slaterWeightMin = slaterWeightMin,
                                  restrictions = restrictions,
                                  h = h,
                                  parallelization_mode = 'H_build'
                                 )
    h0 = get_h0_operator(h0_filename, nBaths)

    return 
    
