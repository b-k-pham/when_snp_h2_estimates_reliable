"""

These are functions used to conduct GWASH, LD Score Regression, HEELS, and GCTA
Heritability Estimation comparisons in simulation as shown in Pham et al. 2025.

This specific file contains methods to conduct data preprocessing.
"""
# Authors: Benjamin Pham

import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.io as sio
import scipy.stats as stats

from tqdm import tqdm
import sys

import gzip

if sys.platform == 'win32':
    import os
    os.environ["R_HOME"] = f"{os.environ['CONDA_PREFIX']}\\Lib\\R"


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from plotnine import *

import matplotlib.pyplot as plt 

import os
import subprocess


from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.linalg import cho_factor, cho_solve
from numba import njit

from scipy.sparse import coo_matrix,csr_matrix
#from sparse_dot_mkl import dot_product_mkl
import pickle
import warnings
warnings.filterwarnings('ignore')

import uuid
import subprocess


from src.methods.GWASH_funcs import *
from src.methods.GWASH_sim_funcs import *
from src.methods.ldsc_barebones import *

from src.methods.gcta_numba_helpers import (
    do_slogdet_cholesky,
    compute_A_inv_w_solve,
    calc_P_numba,
    make_AI_matrix_numba,
    make_deriv_matrix_numba,
    loglik_numba,
    parse_my_gcta_numba
)


from joblib import Parallel, delayed
from collections import defaultdict

from numba import njit,jit
# hacky solution to track progress from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
class CallBack(object):
    completed = defaultdict(int)

    def __init__(self, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        CallBack.completed[self.parallel] += 1
        tqdm.write("done with {}".format(CallBack.completed[self.parallel]))
        if self.parallel._original_iterable:
            self.parallel.dispatch_next()
            
#import joblib.parallel
#joblib.parallel.CallBack = CallBack

import threading

import re

import time
from pandas_plink import read_plink

import logging
# Set up logging
logging.basicConfig(level=logging.INFO)


class data_preprocessing:
    """
        Class to preprocess data (X,y,b) before any further analysis.
        
        Parameters
        ----------
        data_gen: class object
        Object that stores parameters of generated data.
        
        nPCs: int
        Number of Principal Components.
        
        regress_on_X: logical
        Should PCs be regressed out of X?
        
        regress_on_y: logical
        Should PCs be regressed out of y?
    """
    def __init__(self,data_gen,nPCs,regress_X_on_PC = True,regress_y_on_PC = True):
        self.nPCs = nPCs
        self.regress_X_on_PC = regress_X_on_PC
        self.regress_y_on_PC = regress_y_on_PC
    def extract_resids_PCs(self,X_col,U):
        """Fit regression of X_col on  U[:,0:nPCs] and get residuals.
           Parameters
           ----------
           
           X_col: ndarray of shape (n,)
           column of X matrix.
           
           
           U: array-like of shape(n,k)
           U matrix from SVD.
           
           Returns
           -------
           result.resid: ndarray of shape(m,)
           Residuals from regression of X_col on  U[:,0:nPCs]
           
           
           
        """
        model = sm.OLS(X_col, U[:,0:self.nPCs])
        result = model.fit()
        return result.resid
    def regress_y_on_PCs(self,y,U):
        """Fit regression of y_col on  U[:,0:nPCs] and get residuals.
           Parameters
           ----------
           
           y_col: ndarray of shape (n,)
           column of y.
           
           
           U: array-like of shape(n,k)
           U matrix from SVD.
           
           Returns
           -------
           result.resid: ndarray of shape(n,)
           Residuals from regression of y_col on  U[:,0:nPCs]
           
           
        """
        model = sm.OLS(y,U[:,0:self.nPCs])
        result = model.fit()
        #h2_hat_1 = result.rsquared_adj()
        return result.resid.reshape(-1,1)
    
    def regress_out_PCs(self,X,y):
        """Wrapper function of regressing PCs in X and y if the flags are set.
           Parameters
           ----------
           
           X: array-like of shape (n,m)
           Genotype Matrix.
           
           y: ndarray of shape(n,)
           Phenotype.
           
           Returns
           -------
           out: dictionary
           Contains two keys: X_res and y_res.
           X_res contains a new n x m matrix where each column of X is replaced by residuals of X_col on  U[:,0:nPCs]
           y_res contains a new ndarray of shape (n,) of just residuals of y on U[:,0:nPCs].
           
           If regress_X_on_PC is False, X_res contains X instead.
           If regress_y_on_PC is False, y_res contains y instead.
           
           
        
        """
        U, S, Vh =np.linalg.svd(X)
        X_res = np.apply_along_axis(self.extract_resids_PCs, axis=0, arr=X, U=U)
        y_res = self.regress_y_on_PCs(y,U)
        out = dict()
        if self.regress_X_on_PC:
            out['X_res'] = X_res
        else:
            out['X_res'] = X
        if self.regress_y_on_PC:
            out['y_res'] = y_res
        else:
            out['y_res'] = y
        return out
    
    def manual_standardize_and_scale_inR(self,X):
        """Standardize and scale X similarly to scale() in R.
        This means that E(X) = 0 and Var(X) = 1.
        
        Parameters:
        ----------
        X: array-like of shape (n,m)
        Genotype Matrix.
        
        Returns:
        --------
        X_centered/np.std(X_centered,ddof = 1,axis = 0): array-like of shape (n,m)
        Standardized and scaled X matrix.
        
        
        """
        X_centered = X - np.mean(X,axis = 0)
        return X_centered/np.std(X_centered,ddof = 1,axis = 0)
        
    