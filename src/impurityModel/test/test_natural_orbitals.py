import numpy as np
import pytest
from impurityModel.ed import natural_orbitals
from impurityModel.ed.block_structure import BlockStructure

def test_get_huge_star_linear():
    w = np.linspace(-10, 10, 1000)
    hyb = 1j * np.ones((1000, 1, 1), dtype=complex) / (w.reshape(1000, 1, 1) + 1j * 0.1)
    
    E_huge, V_huge = natural_orbitals.get_huge_star(w, hyb, n_bins=100, grid_type="linear")
    
    assert E_huge.shape[0] <= 100
    assert E_huge.shape[0] > 0
    assert V_huge.shape[0] == E_huge.shape[0]
    assert V_huge.shape[1] == 1
    assert np.all(np.isreal(E_huge))
    assert np.all(np.isreal(V_huge))

def test_get_huge_star_logarithmic():
    w = np.linspace(-10, 10, 1000)
    hyb = 1j * np.ones((1000, 1, 1), dtype=complex) / (w.reshape(1000, 1, 1) + 1j * 0.1)
    
    E_huge, V_huge = natural_orbitals.get_huge_star(w, hyb, n_bins=100, grid_type="logarithmic")
    
    assert E_huge.shape[0] <= 100
    assert E_huge.shape[0] > 0
    assert V_huge.shape[0] == E_huge.shape[0]

def test_build_mean_field_hamiltonian():
    H_imp = np.array([[2.0]])
    E_huge = np.array([-1.0, 1.0])
    V_huge = np.array([[0.5], [0.5]])
    
    H_MF = natural_orbitals.build_mean_field_hamiltonian(H_imp, E_huge, V_huge)
    
    assert H_MF.shape == (3, 3)
    assert np.allclose(H_MF[0, 0], 2.0)
    assert np.allclose(H_MF[1, 1], -1.0)
    assert np.allclose(H_MF[2, 2], 1.0)
    assert np.allclose(H_MF[0, 1:], V_huge.T[0])
    assert np.allclose(H_MF[1:, 0], V_huge[:, 0])

def test_truncate_natural_orbitals():
    w = np.linspace(-10, 10, 1000)
    hyb = 1j * np.ones((1000, 1, 1), dtype=complex) / (w.reshape(1000, 1, 1) + 1j * 0.1)
    H_imp = np.array([[0.0]])
    
    e_opt, v_opt = natural_orbitals.truncate_natural_orbitals(H_imp, w, hyb, n_keep=4, n_bins=100, grid_type="linear")
    
    assert v_opt.shape == (4, 1, 1)
    assert e_opt.shape == (4,)
    
def test_fit_hyb_natural_orbitals():
    w = np.linspace(-5, 5, 200)
    # two identical orbitals, completely decoupled
    hyb = np.zeros((200, 2, 2), dtype=complex)
    hyb[:, 0, 0] = 1j / (w + 1j * 0.2)
    hyb[:, 1, 1] = 1j / (w + 1j * 0.2)
    
    H_imp_blocks = {0: np.array([[0.0]]), 1: np.array([[0.0]])}
    bath_states_per_orbital = 3
    
    # Simple block structure mapping (impurity indices -> imp)
    block_structure = BlockStructure(
        blocks=[[0], [1]],
        identical_blocks=[],
        transposed_blocks=[],
        particle_hole_blocks=[],
        particle_hole_transposed_blocks=[],
        inequivalent_blocks=[0, 1]
    )
    
    e_baths, h_baths = natural_orbitals.fit_hyb_natural_orbitals(
        w, hyb, H_imp_blocks, bath_states_per_orbital, block_structure, n_bins=50, grid_type="linear"
    )
    
    assert len(h_baths) == 2
    assert len(e_baths) == 2
    assert h_baths[0].shape == (3, 1, 1)
    assert h_baths[1].shape == (3, 1, 1)
    assert e_baths[0].shape == (3,)
    assert e_baths[1].shape == (3,)
    # The two identical bands should give identical bath discretizations
    assert np.allclose(h_baths[0], h_baths[1])
    assert np.allclose(e_baths[0], e_baths[1])
