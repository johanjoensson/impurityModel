from impurityModel.ed import create
from impurityModel.ed import product_state_representation as psr


def test_create_simple_state():
    # Start with vaccum state (no electrons)
    integer = 0

    n_spin_orbitals = 7

    # Create the other representations of the state
    bits = psr.int2bitarray(integer, n_spin_orbitals)
    b = psr.int2bytes(integer, n_spin_orbitals)

    # Create two electrons and get state: |psi> = c2 c5 |0>
    bits_new = bits.copy()
    amp = create.ubitarray(5, bits_new)
    assert amp == 1
    amp = create.ubitarray(2, bits_new)
    assert amp == 1
    assert psr.bitarray2tuple(bits_new) == (2, 5)

    b_new, amp = create.ubytes(n_spin_orbitals, 5, b)
    assert amp == 1
    b_new, amp = create.ubytes(n_spin_orbitals, 2, b_new)
    assert amp == 1
    assert psr.bytes2tuple(b_new, n_spin_orbitals) == (2, 5)

    # Check that get a negative sign when create electron at spin-orbital 3,
    # since there is an odd number of eletrons with lower spin-orbital index (one electron in spin-orbital 2)
    amp = create.ubitarray(3, bits_new)
    assert amp == -1
    assert psr.bitarray2tuple(bits_new) == (2, 3, 5)

    b_new, amp = create.ubytes(n_spin_orbitals, 3, b_new)
    assert amp == -1
    assert psr.bytes2tuple(b_new, n_spin_orbitals) == (2, 3, 5)

    # Check that get zero amplitude when create electron at spin-orbital 3
    amp = create.ubitarray(3, bits_new)
    assert amp == 0
    assert psr.bitarray2tuple(bits_new) == (2, 3, 5)

    b_new, amp = create.ubytes(n_spin_orbitals, 3, b_new)
    assert amp == 0
    assert psr.bytes2tuple(b_new, n_spin_orbitals) == (2, 3, 5)

