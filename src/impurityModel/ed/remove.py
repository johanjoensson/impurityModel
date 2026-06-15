"""
This module contains functions to remove/annihilate an electron to a product state.
Depending on the representation type of the product state, different functions should be used.
Supported types are: tuple, str, int, bitarray and bytes.

The ordering convention is such that the normal ordering of a product state is
`|psi> = c2 c5 |0>`, (and not `c5 c2 |0>`).

"""

# Local imports
from impurityModel.ed import product_state_representation as psr


def ubitarray(i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    i : int
        Spin-orbital index
    state : bitarray(N)
        Product state.

    Returns
    -------
    amp : int
        Amplitude. 0, -1 or 1.

    """
    if not state[i]:
        return 0
    # Modify the product state by removing an electron
    state[i] = False
    # Amplitude
    return 1 if state[:i].count() % 2 == 0 else -1


def ubytes(n_spin_orbitals, i, state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    state : bytes
        Product state.

    Returns
    -------
    state_new : bytes
        Product state.
    amp : int
        Amplitude. 0, -1 or 1.

    """
    # bitarray representation of product state.
    bits = psr.bytes2bitarray(state, n_spin_orbitals)
    # remove an electron at spin-orbital index i.
    amp = ubitarray(i, bits)
    # Convert back the updated product state to bytes representation.
    state_new = psr.bitarray2bytes(bits)
    return state_new, amp

