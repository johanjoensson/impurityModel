import numpy as np
import scipy.optimize as opt
from typing import Optional, Dict

class BathFitter:
    """
    Fits bath parameters (energies epsilon and hoppings V) to a target hybridization function.
    Supports multi-orbital systems, real/imaginary weights, and moment constraints.
    """
    def __init__(self, w: np.ndarray, delta_target: np.ndarray, n_bath: int, n_imp: int, eta: float = 0.0, complex_v: bool = False, matsubara: bool = False):
        """
        Parameters
        ----------
        w : 1D array
            Frequencies (N_w,). Can be real frequencies or Matsubara frequencies w_n.
        delta_target : 3D array
            Target hybridization function of shape (N_w, n_imp, n_imp)
        n_bath : int
            Number of bath sites to fit
        n_imp : int
            Number of impurity orbitals
        eta : float, default 0.0
            Imaginary broadening used in the real-frequency grid. Ignored if matsubara=True.
        complex_v : bool, default False
            Whether to allow complex hopping parameters
        matsubara : bool, default False
            If True, w is interpreted as Matsubara frequencies and evaluated at z = i * w.
        """
        self.w = w
        self.delta_target = delta_target
        self.n_bath = n_bath
        self.n_imp = n_imp
        self.eta = eta
        self.complex_v = complex_v
        self.matsubara = matsubara
        
        # Optimization weights
        self.weight_real = 1.0
        self.weight_imag = 1.0
        self.w_weights = np.ones_like(w)  # frequency-dependent weights
        
        # Regularization (L2 penalty on V)
        self.weight_reg_v = 0.0
        
        self.target_moments: Dict[int, np.ndarray] = {}
        self.moment_weights: Dict[int, float] = {}

    def set_moment(self, order: int, value: np.ndarray, weight: float):
        """
        Add a moment target to the cost function.
        
        Parameters
        ----------
        order : int
            0 for M0 (total mass), 1 for M1 (center of mass), etc.
        value : np.ndarray
            Target moment matrix of shape (n_imp, n_imp)
        weight : float
            Scalar weight in the cost function
        """
        self.target_moments[order] = np.asarray(value)
        self.moment_weights[order] = weight

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Unpack 1D array into eps and V """
        eps = x[:self.n_bath]
        if self.complex_v:
            v_real = x[self.n_bath : self.n_bath + self.n_imp * self.n_bath]
            v_imag = x[self.n_bath + self.n_imp * self.n_bath :]
            v = v_real + 1j * v_imag
            v = v.reshape(self.n_imp, self.n_bath)
        else:
            v = x[self.n_bath : self.n_bath + self.n_imp * self.n_bath]
            v = v.reshape(self.n_imp, self.n_bath)
        return eps, v

    def pack(self, eps: np.ndarray, v: np.ndarray) -> np.ndarray:
        """ Pack eps and V into a 1D array """
        if self.complex_v:
            return np.concatenate([eps, v.real.flatten(), v.imag.flatten()])
        else:
            return np.concatenate([eps, v.flatten()])

    def hybridization(self, eps: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Calculate Delta_ij(w) = sum_k V_ik V_jk^* / (w + I*eta - eps_k)
        Returns shape (N_w, n_imp, n_imp)
        """
        if self.matsubara:
            z = 1j * self.w
        else:
            z = self.w + 1j * self.eta  # (N_w,)
        denom_inv = 1.0 / (z[:, None] - eps[None, :])
        v_tensor = v[:, None, :] * np.conj(v[None, :, :])  # (n_imp, n_imp, N_b)
        delta = np.einsum('ijk,wk->wij', v_tensor, denom_inv)
        return delta

    def cost_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """ Calculate the total cost function and its gradient """
        eps, v = self.unpack(x)
        if self.matsubara:
            z = 1j * self.w
        else:
            z = self.w + 1j * self.eta  # (N_w,)
        G = 1.0 / (z[:, None] - eps[None, :])
        v_tensor = v[:, None, :] * np.conj(v[None, :, :])
        delta_fit = np.einsum('ijk,wk->wij', v_tensor, G)
        
        diff = delta_fit - self.delta_target
        
        diff_real = diff.real * self.w_weights[:, None, None]
        diff_imag = diff.imag * self.w_weights[:, None, None]
        
        cost_val = self.weight_real * np.sum(diff_real**2) + self.weight_imag * np.sum(diff_imag**2)
        
        # E_star for gradient
        E_star = 2 * self.weight_real * diff_real * self.w_weights[:, None, None] - 1j * 2 * self.weight_imag * diff_imag * self.w_weights[:, None, None]
        
        # Regularization to prevent diverging hoppings
        if self.weight_reg_v > 0:
            cost_val += self.weight_reg_v * np.sum(np.abs(v)**2)
            
        # dC / deps
        G2 = G**2
        v_E_v = np.einsum('ik,wij,jk->wk', v, E_star, np.conj(v))
        grad_eps = np.real(np.sum(G2 * v_E_v, axis=0))
        
        # dC / dV
        A = np.einsum('wmj,jk->wmk', E_star, np.conj(v))
        B = np.einsum('wim,ik->wmk', E_star, v)
        
        if self.complex_v:
            grad_R = np.real(np.einsum('wk,wmk->mk', G, A + B)) + 2 * self.weight_reg_v * v.real
            grad_I = np.imag(np.einsum('wk,wmk->mk', -G, A - B)) + 2 * self.weight_reg_v * v.imag
        else:
            grad_V = np.real(np.einsum('wk,wmk->mk', G, A + B)) + 2 * self.weight_reg_v * v
        
        # Moment matching
        if len(self.target_moments) > 0:
            for order, target_M in self.target_moments.items():
                weight = self.moment_weights[order]
                if weight > 0:
                    eps_p = eps ** order
                    fit_M = np.sum(v_tensor * eps_p[None, None, :], axis=-1)
                    diff_M = fit_M - target_M
                    cost_val += weight * np.sum(np.abs(diff_M)**2)
                    
                    D_star = 2 * weight * np.conj(diff_M)
                    
                    if order > 0:
                        eps_p_minus_1 = order * (eps ** (order - 1))
                        v_D_v = np.einsum('ik,ij,jk->k', v, D_star, np.conj(v))
                        grad_eps += np.real(eps_p_minus_1 * v_D_v)
                        
                    A_M = np.einsum('mj,jk,k->mk', D_star, np.conj(v), eps_p)
                    B_M = np.einsum('im,ik,k->mk', D_star, v, eps_p)
                    
                    if self.complex_v:
                        grad_R += np.real(A_M + B_M)
                        grad_I += -np.imag(A_M - B_M)
                    else:
                        grad_V += np.real(A_M + B_M)
                        
        if self.complex_v:
            grad = self.pack(grad_eps, grad_R + 1j * grad_I)
        else:
            grad = self.pack(grad_eps, grad_V)
            
        return cost_val, grad

    def fit(self, n_starts: int = 5, method: str = 'L-BFGS-B', maxiter: int = 10000) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the numerical optimization from multiple random starts to find the global minimum.
        
        Returns
        -------
        best_eps : 1D array
            Optimized bath energies (n_bath,)
        best_v : 2D array
            Optimized hopping parameters (n_imp, n_bath)
        """
        best_cost = np.inf
        best_x = None
        
        for i in range(n_starts):
            w_max = np.max(np.abs(self.w))
            w_max = w_max if w_max > 0 else 1.0
            eps0 = np.random.uniform(-w_max, w_max, self.n_bath)
            if self.complex_v:
                v0 = np.random.randn(self.n_imp, self.n_bath) + 1j * np.random.randn(self.n_imp, self.n_bath)
            else:
                v0 = np.random.randn(self.n_imp, self.n_bath)
                
            x0 = self.pack(eps0, v0)
            res = opt.minimize(self.cost_and_grad, x0, method=method, jac=True, options={'maxiter': maxiter})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
                
        self.best_eps, self.best_v = self.unpack(best_x)
        self.best_cost = best_cost
        return self.best_eps, self.best_v
