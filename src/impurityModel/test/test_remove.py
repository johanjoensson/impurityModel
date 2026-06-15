from impurityModel.ed import create, remove
from impurityModel.ed import product_state_representation as psr


def test_remove_simple_state():
    # Start with vaccum state (no electrons)
    n_spin_orbitals = 7
    b = psr.int2bytes(0, n_spin_orbitals)

    # Create three electrons and get state: |psi> = c2 c3 c5 |0er>
    b, amp = create.ubytes(n_spin_orbitals, 5, b)
    b, amp = create.ubytes(n_spin_orbitals, 3, b)
    b, amp = create.ubytes(n_spin_orbitals, 2, b)
    assert psr.bytes2tuple(b, n_spin_orbitals) == (2, 3, 5)

    bits = psr.bytes2bitarray(b, n_spin_orbitals)

    # Remove an electron in spin-orbital 3, and then in 2, and then 4
    amp = remove.ubitarray(3, bits)
    assert amp == -1
    assert psr.bitarray2tuple(bits) == (2, 5)
    amp = remove.ubitarray(2, bits)
    assert amp == 1
    assert psr.bitarray2tuple(bits) == (5,)
    amp = remove.ubitarray(4, bits)
    assert amp == 0
    assert psr.bitarray2tuple(bits) == (5,)

    b, amp = remove.ubytes(n_spin_orbitals, 3, b)
    assert amp == -1
    assert psr.bytes2tuple(b, n_spin_orbitals) == (2, 5)
    b, amp = remove.ubytes(n_spin_orbitals, 2, b)
    assert amp == 1
    assert psr.bytes2tuple(b, n_spin_orbitals) == (5,)
    b, amp = remove.ubytes(n_spin_orbitals, 4, b)
    assert amp == 0
    assert psr.bytes2tuple(b, n_spin_orbitals) == (5,)

