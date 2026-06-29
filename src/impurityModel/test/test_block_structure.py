import numpy as np
import pytest

from impurityModel.ed.block_structure import (
    BlockStructure,
    build_block_structure,
    build_greens_function,
    get_equivalent_blocks,
    get_equivalent_orbs,
    build_matrix,
    print_block_structure,
)


def test_build_block_structure_matrix():
    mat = np.zeros((7, 7))
    b1 = np.array([[1.0, 0.5], [0.5, 2.0]])
    mat[0:2, 0:2] = b1
    # Identical
    mat[2:4, 2:4] = b1

    b3 = np.array([[3.0]])
    mat[4:5, 4:5] = b3

    # Negative identical (particle-hole without transpose)
    b4 = -b1
    mat[5:7, 5:7] = b4

    bs = build_block_structure(G=None, mat=mat)

    assert len(bs.blocks) == 4
    assert bs.blocks[0] == [0, 1]
    assert bs.blocks[1] == [2, 3]
    assert bs.blocks[2] == [4]
    assert bs.blocks[3] == [5, 6]

    assert bs.identical_blocks[0] == [0, 1]
    assert bs.inequivalent_blocks == [0, 2]


def test_get_equivalent_blocks():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3], [4]],
        identical_blocks=[[0, 1], [], [2]],
        transposed_blocks=[[], [], []],
        particle_hole_blocks=[[], [], []],
        particle_hole_transposed_blocks=[[], [], []],
        inequivalent_blocks=[0, 2],
    )

    eq_blocks = get_equivalent_blocks(bs)
    assert eq_blocks == [[0, 1], [2]]

    eq_orbs = get_equivalent_orbs(bs)
    assert eq_orbs == [[0, 1, 2, 3], [4]]


def test_build_matrix():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[1.0, 0.5], [0.5, 2.0]])
    mat = build_matrix([b1], bs)
    assert np.allclose(mat[0:2, 0:2], b1)
    assert np.allclose(mat[2:4, 2:4], b1)


def test_print_block_structure(capsys):
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    print_block_structure(bs)
    captured = capsys.readouterr()
    assert "0   0   +   +" in captured.out
    assert "+   +   1   1" in captured.out


def test_build_block_structure_G():
    G = np.zeros((1, 7, 7), dtype=complex)
    b1 = np.array([[[1.0, 0.5], [0.5, 2.0]]])
    G[:, 0:2, 0:2] = b1
    # Identical
    G[:, 2:4, 2:4] = b1

    b3 = np.array([[[3.0]]])
    G[:, 4:5, 4:5] = b3

    # Particle-hole (minus and reverse freq, here shape is 1)
    b4 = -b1
    G[:, 5:7, 5:7] = b4

    bs = build_block_structure(G=G)
    assert len(bs.blocks) == 4
    assert bs.identical_blocks[0] == [0, 1]
    # Check that particle-hole blocks for block 0 include block 3
    assert 3 in bs.particle_hole_blocks[0] or 3 in bs.particle_hole_transposed_blocks[0]


def test_build_greens_function():
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0, 1], []],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[[1.0, 0.5], [0.5, 2.0]]])
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], b1)


def test_build_greens_function_transposed():
    # Block 1 is the transpose of block 0 (swap last two orbital axes).
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[1], [0]],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    # Shape: (n_freq=3, n_orb_block=2, n_orb_block=2) — asymmetric so transpose is visible.
    b1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], b1.swapaxes(-2, -1))


def test_build_greens_function_particle_hole():
    # Block 1 is the particle-hole partner of block 0 (frequency reversed, same orbital content).
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[1], [0]],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], b1[::-1, :, :])


def test_build_greens_function_particle_hole_transposed():
    # Block 1 is frequency-reversed AND orbitally transposed relative to block 0.
    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[1], [0]],
        inequivalent_blocks=[0],
    )
    b1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])
    G = build_greens_function([b1], bs)
    assert np.allclose(G[:, 0:2, 0:2], b1)
    assert np.allclose(G[:, 2:4, 2:4], b1[::-1].swapaxes(-2, -1))
