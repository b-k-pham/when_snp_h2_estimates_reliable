"""

These are functions used to conduct GWASH, LD Score Regression, HEELS, and GCTA
Heritability Estimation comparisons in simulation as shown in Pham et al. 2025.

This specific file contains methods to generate simulated data from AR1 and realistic LD.
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

from src.simtools.do_analysis import do_analysis
from src.simtools.data_preprocessing import data_preprocessing

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




class data_generation:
    """
        Class to generate data (X,y,b) and any transformations of these (XXT,XTX,ldscores) based on the inputted parameters.
        
        Parameters
        ----------
        n: integer
            number of individuals (individuals) to generate in X. By default, it is set to 500 as seen in 1000 Genomes.
        m: integer
            number of predictors (SNPs) to generate in X. By default, it is set to 1000 for debugging purposes.
            It should be set higher to reflect actual SNP counts in real SNP Genotype Data.
        rho1: float
            X is split into two parts. X1 consists of the first half of X from 1:n1
            rho parameter of the AR1 structure underlying X1. This value should range between 0 and 1.
        rho2: float
            X is split into two parts. X2 consists of the second half of X from n1+1: n2
            rho parameter of the AR1 structure underlying X2. This value should range between 0 and 1.
            If this value is None, then it is set to equal rho1.
        sigma_s: float
            Population stratification term from the environment. The data is generated as follows:
            y = XB + S + eps
            S consists of a vector where the first n1 elements are sigma_s and the other elements are -sigma_s
            This value also contributes to the error term:
            eps = np.random.normal(loc = 0,scale = 1-h2_pop-(sigma_s**2),size = (n,1))
        Fst: float
            Genetic drift coefficient/ Wright's Constant. In literature, the maximum value observed is 0.1.
            Standardization term for X.
            f = np.random.normal(loc = 0, scale = np.sqrt(Fst),size = m).reshape(-1,1)
            X = np.divide(X,np.sqrt(1 + f**2).reshape(1,-1))
        pm_causal: float
            Proportion of SNPs that have a non-zero effect size (defined as causal). This ranges from 0 - 1.
            If this option is not specified, then by default all SNPs (1) are causal.
        h2_pop: float
            Desired population SNP heritability. 
            This means the expression: np.var(Xb)/np.var(y) should be approximate to h2_pop.
            This ranges from 0 - 1.
        fixed_m_causal: logical
            Should pm_causal be fixed or random selected? If this option is set to True, the the first pm_causal * m SNPs have non-zero effect sizes. Otherwise, SNPs are randomly selected to be causal via binomial distribution such that the expected value is pm_causal.
        standardize: logical
            Should X be standardized? If this option is True, X is divided by np.sqrt(1 + f**2):
            X = np.divide(X,np.sqrt(1 + f**2).reshape(1,-1))
        do_ldscore_bias_correct: logical
            When constructing ld scores, should bias correction be applied?
        M_prop: integer
            Proportion of individuals with SNP. 
            In real data, individuals have a select number of SNPs (not all m SNPs)
            This only affects LD Score Regression related inputs.
        n_tilde: integer
            Sample size of reference panel. By default, n_tilde is set to n.
        realistic: logical
            Should MVN X be generated with a underlying LD structure from real data instead of AR1? 
            The m for X will be the same number of SNPs as the LD structure, not the specified m.
            rho options will have no effect since X has the same LD structure as the supplied genotype matrix.
            For this option to work, prefix must be specified.
        prefix: string
            path + bim/bed/fam file prefix of individual level genetics data that is used to get underlying LD structure.
        ref_ldscores: None or array-like
            reference LD Scores. If not none, use these instead of LD Scores from data.
        ref_mu2_hat: float
            reference mu2_hat for GWASH variance calculation. If not none, use this instead of mu2_hat from data.
        ref_mu3_hat: float
            reference mu3_hat for GWASH variance calculation. If not none, use this instead of mu3_hat from data.
        old: logical
            For certain functions, the less optimal implementation is used instead.
            For example, construct LD scores directly from estimating XTX/(n-1) and summing across rows rather than
            doing iterating over each ith LD score: (1/(n-1)^2) * X[i]XXTX[i].
        debug: logical
            Enable debugging mode. Might result in saving arrays to disk.
        use_gwash_m: logical
            Use m-1 instead of m when adjusting LD Scores. If this is set to False, can get different LD Scores that result in slightly different results when comparing to the actual LDSC implementation.
    """

    def __init__(self,n = 500,m=1000,rho1 = 0,rho2 = None,sigma_s = 0,Fst = 0,pm_causal = None,h2_pop = 0.2,fixed_m_causal = False,standardize = True,do_ldscore_bias_correct = True,M_prop = 1,n_tilde = None,realistic = False,model_Fst_in_realistic = True,prefix = None,ref_ldscores = None,ref_mu2_hat = None,ref_mu3_hat = None,ref_X = None,old = False,debug = False,use_gwash_m = False):
        
        
        # Sims related Params
        self.standardize = standardize
        self.do_ldscore_bias_correct = do_ldscore_bias_correct
        self.prefix = prefix
        self.realistic = realistic
        self.model_Fst_in_realistic = model_Fst_in_realistic
        # Data-gen related Params
        self.n = n
        m_final = 0
        if m is None: # only do it if m is none.
            if self.realistic: # if realistic, force num_snps to be the as large as the ld matrix
                #self.L = np.load(self.ld_mat_path)
                #m_final = self.L.shape[0]
                
                # Open the .npy file in binary mode
                _,_,bed = read_plink(self.prefix, verbose=False) # only load the bed file and nothing else
                bed = bed.T
                n_bed, m_final = bed.shape
                del bed
            else:
                m_final = m
            self.m = m_final
        else:
            self.m = m
        self.pm_causal = pm_causal
        if pm_causal is not None:
            self.m_causal = int(np.round(pm_causal * self.m))
        else:
            self.m_causal = m # Assume all SNPs are causal.
            self.pm_causal = 1
        self.fixed_m_causal = fixed_m_causal
        
        self.Fst = Fst
        self.rho1 = rho1
        self.rho2 = rho2
        if self.rho2 == None: # if rho2 is none, then make rho1 = rho2
            self.rho2 = self.rho1
        self.sigma_s = sigma_s
        
        self.h2_pop = h2_pop
        
        
        self.old = old # do old methods of X generation and ldscore computation (more computationally expensive)
        
        # LDSC related Params
        self.M_prop = M_prop
        self.M = np.array([M_prop* self.m])
        self.N_per_SNP = np.repeat(self.n,self.m).reshape(-1,1) # Sample Size per SNP. For now, assume each SNP has all 500 people.
        self.use_gwash_m = use_gwash_m
        #debug - turn on if need to check something w/ X
        self.debug = debug
        
        #Use ref_ldscores
        self.ref_ldscores = ref_ldscores
        self.ref_mu2_hat = ref_mu2_hat
        self.ref_mu3_hat = ref_mu3_hat
        self.ref_X = ref_X
        self.n_tilde = n_tilde
        
        if self.ref_ldscores is None:
            self.n_tilde = self.n
        else:
            self.n_tilde = n_tilde
        
    def gen_X(self, div_pop_term: bool = True, old: bool = False, realistic: bool = False, debug: bool = False):
        """Generate MVN X with AR1/realistic structure.
        X is constructed from two populations. The first half of X is of population 1 and the other individuals are of population 2.
        
        Parameters
        ----------
        div_pop_term: logical
            Should X be divided by f_standardize_term?
            f = np.random.normal(loc = 0, scale = np.sqrt(Fst),size = m).reshape(1,-1)
            f_standardize_term = np.sqrt(1 + f**2)
            X = np.divide(X,f_standardize_term)
        
        
        old: logical
        Should the older less efficient version of generating X be used?
        
        
        realistic: logical
            Should MVN X be generated with a underlying LD structure from real data instead of AR1? 
            The m for X will be the same number of SNPs as the LD structure, not the specified m.
            rho options will have no effect since X has the same LD structure as the supplied genotype matrix.
            For this option to work, prefix must be specified.
        model_Fst_in_realistic: logical
            Generate X from realistic LD Structure that has added Fst. This corresponds to the 2nd part of Figure 5 where Fst and sigma_{s} is simultaneously increased.
            For this to take effect, realistic has to be true and Fst != 0.
        debug: logical
            Should additional things (f or num_causal) be outputted to help with debugging?
            
        Returns
        -------
        
        X: array-like of shape (n,m)
           Simulated individual genotype data with either AR1(rho) structure or realistic LD structure.
           
        b: ndarray of shape (m,)
           effect sizes of SNPs.
           
        Debug Returns
        -------------
        f: ndarray of shape (m,)
           Standardization coefficient from Fst.
        num_causal: int
           Number of SNPs that are causal.
        
        
        """
        n = self.n
        Fst = self.Fst
        if n % 2 == 0:
            n1 = int(n/2)
            n2 = n1
        else:
            n1 = int(np.round(n/2))#force n1 to have the odd number of subjects
            n2 = n - n1 # and n2 to have the rest
        m = self.m
        pm_causal = self.pm_causal
        M_prop = self.M_prop
        h2_pop = self.h2_pop
        alpha = pm_causal # used to be m_causal/m. should be pm_causal
        m_causal = int(alpha * m)# should be the same as self.m_causal
        realistic = self.realistic
        model_Fst_in_realistic = self.model_Fst_in_realistic
        
        if realistic:
            # adapted from twas_sim
            #L = np.load(self.ld_mat_path)
            #m = L.shape[0]
            #X = L.dot(np.random.normal(size=(n, m)).T).T
            #del(L)
            #X -= np.mean(X, axis=0)
            #X /= np.std(X, axis=0)
            
            #load bed file
            _, _, bed = read_plink(self.prefix, verbose=False)
            bed = bed.T
            mafs = (np.mean(bed, axis=0) / 2).compute()
            bed -= mafs * 2
            bed /= np.std(bed, axis=0)
            bed = bed[:,:m] # choose a subset of m SNPs if defined.
            bed = bed.compute()
            bed_n,bed_m = bed.shape[0],m
            
            G = np.random.normal(size = (self.n,bed_n))
            X = (1/np.sqrt(bed_n))*(G @ bed) # (k x n) (n x m) = k x m matrix with LD structure of realistic bed file
            n,m = X.shape
            
            if model_Fst_in_realistic:
                V = np.eye(bed_m)
                f = np.random.normal(loc = 0, scale = np.sqrt(Fst),size = m).reshape(1,-1)
                if n % 2 == 0:
                    n1 = int(n/2)
                    n2 = n
                else:
                    n1 = int(np.round(n/2))#force n1 to have the odd number of subjects
                    n2 = n - n1 # and n2 to have the rest
                
                ## NOW SPLIT THE MATRIX INTO TWO PARTS!
                X[:n1,] = X[:n1,] + f
                X[n1:,] = X[n1:,] - f
                
                if div_pop_term:
                    if old:
                        f_standardize_term = np.sqrt(1 + f**2)
                        X = np.divide(X,f_standardize_term)
                    else:
                        X = np.divide(X,np.sqrt(1 + f**2).reshape(1,-1))

        else:
            rho1 = self.rho1
            rho2 = self.rho2
            old = self.old
            
            if old:
                Sigma1 = ar1_cor(m,rho1)
                if rho1 == rho2:
                    Sigma2 = Sigma1
                else:
                    Sigma2 = ar1_cor(m,rho2)
                #V = Sigma # V is written as a correlation matrix of some kind. For now, make it equivalent to Sigma.
                V = np.eye(m)
                f = stats.multivariate_normal.rvs(mean = np.repeat(0,m),cov = Fst * V, size = 1) #genetic drift (stratification by genetics)
                X1 = stats.multivariate_normal.rvs(mean = f,cov = Sigma1, size = n1) # this is the genotype matrix (n1 x m)
                X2 = stats.multivariate_normal.rvs(mean = -f,cov = Sigma2, size = n2) # this is the genotype matrix (n1 x m)

                X = np.concatenate([X1,X2],axis = 0)
            else:

                X = np.zeros((n,m))
                X[:, 0] = np.random.normal(loc = 0,scale = np.sqrt(1), size = n).reshape(1,-1)
                f = np.random.normal(loc = 0, scale = np.sqrt(Fst),size = m).reshape(1,-1)
                for j in range(1,m):
                    epsilon = np.random.normal(loc = 0,scale = np.sqrt(1-rho1**2),size = n)
                    X[:, j] = rho1 * X[:, j - 1] + epsilon

                ## NOW SPLIT THE MATRIX INTO TWO PARTS!
                X[:n1,] = X[:n1,] + f
                X[n1:,] = X[n1:,] - f

                if div_pop_term:
                    if old:
                        f_standardize_term = np.sqrt(1 + f**2)
                        X = np.divide(X,f_standardize_term)
                    else:
                        X = np.divide(X,np.sqrt(1 + f**2).reshape(1,-1))
        
        #b = np.random.normal(loc=0, scale=np.sqrt(h2_pop/m), size=m).reshape(-1,1)
        #np.mean(b) #-0.0009732171613908289 # should be 0
        #np.var(b,ddof = 1) #0.00020027726414197363 #should be 0.2/1000 = 0.0002
        #b = stats.multivariate_normal.rvs(mean = np.repeat(0,m),cov = np.eye(m) * (h2_pop/(m_causal)), size = 1)
        #print(m_causal)
        #print(m)
        #print(h2_pop)
        b = np.random.normal(0,np.sqrt(h2_pop/(m_causal)),m) # equivalent to above since we're assuming all b are independent anyway
        num_causal = np.nan # for diagnostic purposes
        if self.fixed_m_causal:
            b = b.reshape(-1,1)
            b[m_causal:] = 0 #make everything past the first m SNPs have 0 effect
            num_causal = b[b != 0].shape[0]
        else:
            is_causal = np.random.binomial(1,alpha,size = m) # vector of 0s or 1s indicating if a beta is causal randomly chosen.
            b = np.multiply(b,is_causal).reshape(-1,1)
            num_causal = is_causal.sum()
        if debug:
            return X,b,f,num_causal
        else:
            return X,b
        
    

    def gen_y(self,X,b):
        """Generate y from X and b in a linear model.
        
        Parameters
        ----------
        X: array-like of shape (n,m)
           Individual level genotype data.
        b: ndarray of shape (m,)
           SNP effect sizes called "betas".

        Returns
        -------
        
        y: ndarray of shape (n,)
           y = X @ b + S + error
        
            Where:
                X is the individual level genotype data
                b is the SNP effect size
                S is the environmental stratification term added to the genotype effect (X @ b)
                error is the noise term generated from distribution N(0, 1-h2_pop-(sigma_s**2))

        """
        h2_pop = self.h2_pop
        sigma_s = self.sigma_s
        
        n = X.shape[0]
        if n % 2 == 0:
            n1 = int(n/2)
            n2 = n1
        else:
            n1 = int(np.round(n/2))#force n1 to have the odd number of subjects
            n2 = n - n1 # and n2 to have the rest
        S = np.concatenate([np.ones(int(n1)) * (sigma_s),np.ones(int(n2)) * (-sigma_s)]).reshape(-1,1) #environmental (stratification by environment)
        geno_effect = np.matmul(X,b)
        #eps = stats.multivariate_normal.rvs(mean = np.repeat(0,n),cov = np.eye(n) * (1-h2_pop-(sigma_s**2)), size = 1).reshape(-1,1)
        #eps = np.random.normal(loc = 0,scale = np.sqrt(1-h2_pop-(sigma_s**2)),size = (n,1)) # way faster
        #if sigma_s is at the boundary, sigma_s**2 can be 1E-12 larger than 1-h2_pop. Round to the 8th decimal place to make sure it is 0.
        eps = np.random.normal(loc = 0,scale = np.sqrt(1-h2_pop-np.round(sigma_s**2,8)),size = (n,1)) # way faster
        y = geno_effect + S + eps
        
        return y
        
    def ar1_cor(self):
        """Generate true AR1 m x m correlation matrix. Useful for theoretical computations/debugging.
        
           Parameters
           ----------
           Uses m and rho in data_generation class
           Returns
           -------
           ndarray of shape (m,m) following rho correlation structure
        """
        m = self.m
        rho = self.rho1
        init = np.ones([m,m])
        for i in range(m):
            for j in range(m):
                init[i,j] = abs(j - i)
        return(rho**init)
    
    def make_ldscores_from_S_Tilde_orig(self,S_tilde,bias_correct = True):
        """Generate ldscores starting from correlation matrix S_tilde = X_tilde.T @ X_tilde /(n-1).
           The correlation matrix is converted to the squared correlation matrix (each element of S_tilde is squared).
           LD scores are sum across rows of the squared-correlation matrix.
           
           Parameters
           ----------
           S_tilde: sparse array-like of shape (m,m)
              Sparse representation of correlation matrix of X_tilde.
           bias_correct: logical
              Should the bias correction term for ldscores be applied?
           
           Returns
           -------
           ld_scores: ndarray of shape (m,)
           Sum across rows of the squared correlation matrix (LD scores) of each SNP.
        
        """
        M = self.M.item()
        N = self.N_per_SNP
        bias_correct = self.do_ldscore_bias_correct
        #S_tilde_rjk_squared = S_tilde**2 ### WORKS WITH NUMPY ARRAYS BUT NOT DENSE MATRICES!!!
        S_tilde_rjk_squared = S_tilde.power(2)
        ld_scores = (S_tilde_rjk_squared.sum(axis = 0).reshape(-1,1))
        if bias_correct:
            ld_scores = ld_scores - ((M-ld_scores)/(np.mean(N) - 2))
        return ld_scores
        
    def make_ldscores_from_X(self,X,bias_correct = True,use_gwash_m = False):
        """Generate ldscores starting from correlation matrix X_tilde.
           The diagonal of  XTX @ XTX is sum of squared correlation matrix across rows.
           Only the diagonal is needed to be computed.
           
           Parameters
           ----------
           
           
           X: sparse array-like of shape (n,m)
              Individual level genotype data.
           bias_correct: logical
              Should the bias correction term for ldscores be applied?
           use_gwash_m: logical
              Use M-1 as written in Schwartzman et al. 2019 instead of M as written in Bulik-Sulivan 2015 in the bias correction term for ld scores.
           
           Returns
           -------
           
           
           ld_scores: ndarray of shape (m,)
           Sum across rows of the squared correlation matrix (LD scores) of each SNP.
        
        """
        M = self.M.item()
        N = self.N_per_SNP
        use_gwash_m = self.use_gwash_m
        bias_correct = self.do_ldscore_bias_correct
        
        if use_gwash_m:
            M = M - 1
        
        XXT = (X @ X.T) # non-scaled kinship
        #slower (around 45s)
        #ldscores = np.zeros(M)
        #for j in range(M):
        #    ldscores[j] = ((1/((np.mean(N)-1)**2)) * X[:,j].reshape(-1,1).T @ XXT @ X[:,j].reshape(-1,1)).item()
        # faster (around 3s but probably uses more memory)
        ldscores = ((1/((np.mean(N)-1)**2)) * X.T @ XXT @ X).diagonal()
        if bias_correct:
            ldscores = (ldscores - (M-ldscores)/(np.mean(N) - 2))
        return ldscores.reshape(-1,1)
    
    def compute_u2_ldscores(self,X,y, old = False):
        """Wrapper to prepare u2,s2, and ldscores; inputs needed for ldsc and GWASH.
           
           Parameters
           ----------
           
           
           X: array-like of shape (n,m)
              Individual level genotype data.
           y: ndarray of shape (n,)
              Phenotype data.
           
           Returns
           -------
           
           u2: ndarray of shape (m,)
           Test-statistics of SNPs whether the effect size is non-zero.
           
           s2: float
           mean of u2.
           
           ldscores ndarray of shape (m,)
           Sum of correlations of nearby SNPs for each SNP.
        
        """
        old = self.old
        do_ldscore_bias_correct = self.do_ldscore_bias_correct
        use_gwash_m = self.use_gwash_m
        #if use_gwash_m:
        #    print('use m-1')
        #else:
        #    print('use m')
        ref_ldscores = self.ref_ldscores
        u2 = calc_u2(X,y)
        s2 = np.mean(u2)
        if self.ref_ldscores is None:
            if old:
                X_tilde_T_cov = csr_matrix(np.cov(X.T))
                ldscores = self.make_ldscores_from_S_Tilde_orig(X_tilde_T_cov,bias_correct = do_ldscore_bias_correct)
            else:
                ldscores = self.make_ldscores_from_X(X,bias_correct = do_ldscore_bias_correct,use_gwash_m = use_gwash_m)
        else:
            ldscores = self.ref_ldscores
        #ldscores_from_S_tilde_da = make_ldscores_from_S_Tilde_orig(X_tilde_T_cov,M,N_per_SNP,bias_correct = False)
        #ldsc_reg_weights = ldscores_from_S_tilde
        return u2,s2,ldscores # returns chi-squareds (y), ld scores (x)
        
    def create_GCTA_inputs(self,X,y,sim_id = None, output_files = False):
        """Create inputs to use in GCTA.
           
           Parameters
           ----------
           
           
           X: array-like of shape (n,m)
              Individual level genotype data.
           y: ndarray of shape (n,)
              Phenotype data.
           output_files: logical
              Should the outputs be written on the disk? Useful for checking against actual GCTA implementation.
              
           
           Returns
           -------
           
           df: array-like of shape ((n^2 + n)/2,4)
           Long format of Genetic-relatedness matrix of individuals (Kinship Matrix).
           Because the Kinship Matrix is symmetric, it is efficient to just take the lower triangular.
           Another included column is number of non-missing SNPs which is set to m since it's assumed
           that all n individuals have all m SNPs.
           
           tmp: array-like of shape (n,2)
           Table of identifiers of individuals to include in analysis. For now, all individuals are included.
           
           tmp1: array-like of shape (n,3)
           Table of identifiers for all individuals with phenotype.
        
        """
        n,m = X.shape
        kinship = (X @ X.T)/m

        
        idx = np.tril_indices_from(kinship)
        
        df = pd.DataFrame({
        'i': idx[0],
        'j': idx[1],
        'num_nonmiss_snps': m,
        'value': kinship[idx]})
        
        
        
        
        # GCTA indexing starts with 1 like with R
        
        df['i'] = df['i'] + 1
        df['j'] = df['j'] + 1

        fid = [z for z in range(n)]
        iid = [z for z in range(n)]
        
        tmp = pd.DataFrame([fid,iid]).T
        tmp.columns =['fid','iid']
        
        tmp1 = tmp.copy()
        tmp1['y'] = y
        
        
        
        if output_files:
            df.to_csv(sim_id+'_gcta.grm.gz',sep = '\t',header=False, index=False,compression = {'method': 'gzip', 'compresslevel': 1})
            #df.to_csv('sim.grm.gz',sep = '\t',header=False, index=False)
            #gzip.compress('sim.grm.gz')
            tmp.to_csv(sim_id + '_gcta.grm.id',sep = '\t',header=False, index=False)
            tmp1.to_csv(sim_id + '_gcta.phen',sep = '\t',header=False, index=False)

        return df,tmp,tmp1 # GRM, GRM_id, pheno
        
    def format_gcta_to_mats(self,GRM,pheno):
        """Convert GCTA inputs to full matrices.
           
           Parameters
           ----------
           
           
           GRM: array-like of shape ((n^2)/2,4)
              Long formatted Lower Triangular of Full Genetic Relatedness Matrix (Kinship Matrix).
           pheno: array-like of shape (n,3)
              Table of identifiers for all individuals with phenotype.
              
           
           Returns
           -------
           
           GRM: array-like of shape (n,n)
           
           Full Kinship Matrix.
           
           pheno: ndarray of shape (n,)
           
           Phenotype values.
        
        """
        GRM = GRM[['i','j','value']]
        GRM['i'] = GRM['i'] - 1
        GRM['j'] = GRM['j'] - 1
        GRM = coo_matrix((GRM['value'], (GRM['i'], GRM['j']))) # this is just a lower triangular
        GRM = GRM + GRM.transpose()
        GRM.setdiag(GRM.diagonal() / 2)
        pheno = np.array(pheno['y']).reshape(-1,1)
        return GRM.toarray(),pheno
        
### MAKE REF PANEL LIKE 1000 G
        # meaning I generate ldscores separately and use this to try to predict
        # using ref extremely different from X outside won't work

def gen_ref_ldscores_panel(ref_data_gen,nPCs = 5, regress_X_on_PC = False, regress_y_on_PC = False,seed = None):
    if seed is not None:
        np.random.seed(seed)
    X_ref,b_ref = ref_data_gen.gen_X()
    y_ref = ref_data_gen.gen_y(X_ref,b_ref)
    ref_data_preprocessing = data_preprocessing(ref_data_gen,nPCs = nPCs,regress_X_on_PC = regress_X_on_PC,regress_y_on_PC = regress_y_on_PC)
    X_tilde_ref = ref_data_preprocessing.manual_standardize_and_scale_inR(X_ref)
    y_tilde_ref = ref_data_preprocessing.manual_standardize_and_scale_inR(y_ref)
    # create ref_ldscores for ldsc
    ref_u2,ref_s2,ref_ldscores_panel = ref_data_gen.compute_u2_ldscores(X_tilde_ref,y_tilde_ref) # y_tilde is only needed for u2 which isn't returned. ldscores, which are computed from X only is returned.
    # create ref mu2 and ref mu3 for gwash.
    ldsc_reg_weights = ref_ldscores_panel
    sims_ref = do_analysis(ref_data_gen,X_tilde_ref,ref_u2,ref_s2,ref_ldscores_panel,ldsc_reg_weights,calc_mu_hat_2_fast = True)
    gwash_res_ref = sims_ref.do_GWASH_from_X_tilde() # compute GWASH from reference data, retrieve only mu2 and mu3.
    ref_mu2_hat = gwash_res_ref['mu2_hat']
    ref_mu3_hat = gwash_res_ref['mu3_hat']
    
    return ref_ldscores_panel,ref_mu2_hat,ref_mu3_hat,X_tilde_ref