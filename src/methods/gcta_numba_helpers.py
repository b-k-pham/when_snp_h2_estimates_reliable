"""
gcta_numba_helpers.py
Numba-accelerated helper functions for GCTA REML estimation (float64, Cholesky, module-level for JIT).
"""
import numpy as np
from numba import njit

@njit(cache=True)
def do_slogdet_cholesky(V):
    L = np.linalg.cholesky(V)
    return 2.0 * np.sum(np.log(np.diag(L)))
    
@njit(cache=True)
def compute_A_inv_w_solve(A):
    return np.linalg.solve(A,np.eye(A.shape[0]))

@njit(cache=True)
def calc_P_numba(V, X):
    n = V.shape[0]
    V_inv = compute_A_inv_w_solve(V)
    if X is None:
        X = np.ones((n, 1), dtype=np.float64)
    XtVinvX = X.T @ V_inv @ X
    XtVinvX_inv = compute_A_inv_w_solve(XtVinvX)
    rhs = (V_inv @ X) @ XtVinvX_inv @ X.T @ V_inv
    return V_inv - rhs

@njit(cache=True)
def make_AI_matrix_numba(y, P, A):
    tl = (y.T @ P @ A @ P @ A @ P @ y).item()
    tr = (y.T @ P @ A @ P @ P @ y).item()
    bl = (y.T @ P @ P @ A @ P @ y).item()
    br = (y.T @ P @ P @ P @ y).item()
    return 0.5 * np.array([[tl, tr], [bl, br]], dtype=np.float64)

@njit(cache=True)
def make_deriv_matrix_numba(y, P, A):
    top = (np.trace(P @ A) - (y.T @ P @ A @ P @ y)).item()
    bot = (np.trace(P) - (y.T @ P @ P @ y)).item()
    return -0.5 * np.array([[top], [bot]], dtype=np.float64)

@njit(cache=True)
def loglik_numba(V, X, y):
    n = V.shape[0]
    V_inv = compute_A_inv_w_solve(V)
    if X is None:
        X = np.ones((n, 1), dtype=np.float64)
    middle = do_slogdet_cholesky(X.T @ V_inv @ X)
    left = do_slogdet_cholesky(V)
    P = calc_P_numba(V, X)
    right = y.T @ P @ y
    return (-0.5 * (left + middle + right)).item()

@njit(cache=True)
def parse_my_gcta_numba(params):
    VG = params[0, 0]
    Ve = params[1, 0]
    return VG, Ve 