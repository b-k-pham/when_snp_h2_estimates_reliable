"""

These are functions used to conduct GWASH, LD Score Regression, HEELS, and GCTA
Heritability Estimation comparisons in simulation as shown in Pham et al. 2025.

This specific file contains code for method implementations.
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


class do_analysis:
    def __init__(self,data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,
                 intercept = None,return_step1_only = False,calc_mu_hat_2_fast = True,calc_mu_hat_3 = True,time_it = True):
        
        self.data_gen = data_gen
        self.u2 = u2
        self.s2 = s2
        self.ldscores = ldscores
        self.ldsc_reg_weights = ldsc_reg_weights
        self.intercept = intercept
        self.X_tilde = X_tilde
        
        self.calc_mu_hat_2_fast = calc_mu_hat_2_fast # if True, take the mean of ldscores to calculate mu2hat
        self.calc_mu_hat_3 = calc_mu_hat_3
        self.time_it = time_it
        
        self.return_step1_only = return_step1_only
        
        
    def do_ldsc(self):
        u2 = self.u2
        ldscores = self.ldscores
        M = self.data_gen.M
        N_per_SNP = self.data_gen.N_per_SNP
        ldsc_reg_weights = self.ldsc_reg_weights
        intercept = self.intercept
        return_step1_only = self.return_step1_only
        ldsc_h2, ldsc_se,ldsc_intercept,weights = do_ldsc_regression(u2,ldscores,M,N_per_SNP,ldsc_reg_weights,intercept = intercept,return_step1_only = return_step1_only)
        return ldsc_h2,ldsc_se,ldsc_intercept,weights
    
    
    def ldsc_h2_crude_fixed_intercept(self):
        y = self.u2
        x = self.ldscores
        intercept = self.intercept
        M = self.data_gen.M
        N = self.data_gen.N_per_SNP
        yp = y.reshape(-1,1) - intercept
        result = sm.OLS(yp, x).fit()
        return np.multiply(result.params[0],M/np.mean(N)).item()
        #return result


    def ldsc_h2_crude_free_intercept(self):
        y = self.u2
        x = self.ldscores
        M = self.data_gen.M
        N = self.data_gen.N_per_SNP
        
        yp = y.reshape(-1,1)
        x1 = np.multiply(x,np.mean(N)/M)
        X_new = sm.add_constant(x1)
        #X_new = sm.add_constant(x)
        result = sm.OLS(yp, X_new).fit()
        #return np.multiply(result.params[0],M/np.mean(N)).item()
        fitted_intercept = result.params[0]
        #h2 = np.multiply(result.params[1],M/np.mean(N)).item()
        h2 = result.params[1]
        #tqdm.write(X_new)
        #tqdm.write(result.summary())
        return fitted_intercept,h2
    
    def do_ols(self):
        intercept = self.intercept
        if intercept is None:
            return self.ldsc_h2_crude_free_intercept()
        else:
            return self.ldsc_h2_crude_fixed_intercept()
        
    def do_GWASH_from_ldscores(self):
        u2 = self.u2
        s2 = self.s2
        n = self.data_gen.n
        ldscores = self.ldscores
        n_tilde = self.data_gen.n_tilde
        time_it = self.time_it
        GWASH = calc_gwash_from_ldscores(ldscores,s2,n,n_tilde = n_tilde, calc_mu_hat_3 = False,time_it = time_it)
        ### equivalent estimator
        return GWASH
        
    def do_weighted_GWASH_from_ldscores(self,gwash):
        u2 = self.u2
        ldscores = self.ldscores
        
        ldscores_bar = np.maximum(ldscores,1)
        n = self.data_gen.n
        m = self.data_gen.m
        
        wj = (1 + (gwash * (n/m) * ldscores_bar))**(-2)
        W_GWASH = np.mean(np.multiply(wj,u2-1))/np.mean(np.multiply(wj,ldscores_bar * (n/m)))
        
        return W_GWASH
        
    def do_GWASH_from_Stilde(self): # this is assuming you have access to the full LD matrix! With the full LD matrix, you can compute the variance directly. However, we show that it is not necessary as the empirical variance from sims are approximate!
        u2 = self.u2
        s2 = self.s2
        n = self.data_gen.n
        n_tilde = self.data_gen.n_tilde
        X_tilde_T_cov = csr_matrix(np.cov(self.X_tilde.T))
        time_it = self.time_it
        ### equivalent estimator
        GWASH = calc_gwash_from_S_tilde(X_tilde_T_cov,s2,n,n_tilde = n_tilde, calc_mu_hat_3 = True,time_it = time_it)
        return GWASH
    def do_GWASH_from_X_tilde(self):
        u2 = self.u2
        s2 = self.s2
        n = self.data_gen.n
        n_tilde = self.data_gen.n_tilde
        X_tilde = self.X_tilde
        time_it = self.time_it
        ### equivalent estimator
        #calc_gwash_from_X_tilde(X_tilde,u2,n_sumstat, n_tilde = None,time_it = True)
        GWASH = calc_gwash_from_X_tilde(X_tilde,u2,n, n_tilde = n_tilde,time_it = True)
        return GWASH
        
    # --- Optimized do_GCTA method ---
    def do_GCTA(self, y, kinship,GRM_id, tol=1e-4, iter_limit=100,use_native_gcta = False,native_gcta_path = None,cleanup = True):
        # y is input phenotype
        # kinship is input GRM.
        # GRM_id has fids and iids
        
        def format_mats_to_gcta(GRM_mat, pheno_array,GRM_id,m):
            # for now, it is hard coded that there is m nonmissing SNPs (all SNPs are present)
            i_idx, j_idx = np.tril_indices_from(GRM_mat)
            values = GRM_mat[i_idx, j_idx]
            # Indices in GCTA input are 1-based
            GRM_df = pd.DataFrame({'i': i_idx + 1, 'j': j_idx + 1,'num_nonmiss_snps':m, 'value': values})
            # Recreate phenotype DataFrame
            pheno_df = pd.DataFrame({'y': pheno_array.flatten()})
            pheno_df = pd.concat([GRM_id,pheno_df],axis = 1)
            return GRM_df, pheno_df
            
        def parse_gcta(hsq):
            tmp = pd.read_table(hsq)
            tmp1 = tmp[tmp['Source'] == 'V(G)/Vp']
            return tmp1['Variance'].item(),tmp1['SE'].item()
        def convert_time_to_seconds(time_str):
            """
            Extracts elapsed computational time from a string (in sec, min, or hour)
            and converts it to a float number of seconds.
            """
            # Look for a pattern: number + unit
            match = re.search(r'Overall computational time: ([\d\.]+) (sec|min|hour|hours|minutes)', time_str)
            if not match:
                raise ValueError("No valid time format found in the input string.")
            value, unit = match.groups()
            value = float(value)
            if unit in ['sec']:
                return value
            elif unit in ['min', 'minutes']:
                return value * 60
            elif unit in ['hour', 'hours']:
                return value * 3600

            
        def get_descs_from_gcta_log(path_to_gcta_log):
            #get num of iters and computation time from log.
            starts = []
            stops = []
            lines = []
            ct_line = ''
            def get_iteration_from_line(string):
                return int(string[:string.find('\t')])
            def get_computation_time_from_line(string):
                #Overall computational time: 52.07 sec.
                return string[string.find('Overall computational time:'):]
                
            with open(path_to_gcta_log, "r") as f:
                counter = 0
                for line in f:
                    # Process each line, which will include the newline character at the end
                    l_s = line.strip()
                    
                    if l_s == 'Running AI-REML algorithm ...':
                        starts.append(counter)
                    if l_s == 'Log-likelihood ratio converged.':
                        stops.append(counter)
                    if 'Overall computational time' in l_s:
                        ct_line = l_s
                    lines.append(l_s)
                    counter +=1

            num_iters = 0
            
            for z in range(len(stops)):
                num_iters += get_iteration_from_line(lines[stops[z]-1])
            
            return num_iters, convert_time_to_seconds(get_computation_time_from_line(ct_line))
            
        
        def do_py_GCTA(y, kinship, tol=1e-4, iter_limit=100):
            """
            Optimized GCTA REML estimation using Cholesky decomposition and Numba-accelerated helpers.
            Uses float64 throughout.
            """
            start = time.perf_counter()
            n = y.shape[0]
            y = y.astype(np.float64)
            I_n = np.eye(n, dtype=np.float64)
            kinship = kinship.astype(np.float64)
            y_temp = y - np.mean(y)
            y_temp_SSq = ((y_temp ** 2).sum()) / (n - 1)
            sigma2_g, sigma2_e = y_temp_SSq / 2, y_temp_SSq / 2
            V = sigma2_g * kinship + sigma2_e * np.eye(n, dtype=np.float64)
            P = calc_P_numba(V, None)
            LL0 = loglik_numba(V, None, y)
            rhs_g = np.trace(sigma2_g * np.eye(n) - sigma2_g ** 2 * P @ kinship)
            sigma2_g = ((sigma2_g ** 2 * y.T @ P @ kinship @ P @ y + rhs_g) / n).item()
            rhs_e = np.trace(sigma2_e * np.eye(n) - sigma2_e ** 2 * P)
            sigma2_e = ((sigma2_e ** 2 * y.T @ P @ P @ y + rhs_e) / n).item()
            stop = False
            params = np.array([sigma2_g, sigma2_e], dtype=np.float64).reshape(-1, 1)
            counter = 0
            while not stop:
                sigma2_g = params[0, 0]
                sigma2_e = params[1, 0]
                if sigma2_g < 0:
                    sigma2_g = y_temp_SSq * (10 ** -6)
                if sigma2_e < 0:
                    sigma2_e = y_temp_SSq * (10 ** -6)
                params[0, 0] = sigma2_g
                params[1, 0] = sigma2_e
                V = sigma2_g * kinship + sigma2_e * I_n
                P = calc_P_numba(V, None)
                LL = loglik_numba(V, None, y)
                if abs(LL - LL0) < tol:
                    stop = True
                else:
                    LL0 = LL
                    AI_mat = make_AI_matrix_numba(y, P, kinship)
                    deriv_mat = make_deriv_matrix_numba(y, P, kinship)
                    params = params + (compute_A_inv_w_solve(AI_mat) @ deriv_mat)
                    counter += 1
                if counter > iter_limit:
                    logging.info('GCTA exceeded iteration limit')
                    stop = True
            VG, Ve = parse_my_gcta_numba(params)
            VP = VG + Ve
            h2_est = VG / VP
            res_dict = dict()
            end = time.perf_counter()
            res_dict['VG'] = VG
            res_dict['Ve'] = Ve
            res_dict['h2_est'] = h2_est
            res_dict['num_iters'] = counter
            res_dict['time_elapsed'] = end-start
            return res_dict
            
        if use_native_gcta:
            # write input phenotype, input GRM, and GRM_id as temp files.
            native_GRM_input,native_pheno_input = format_mats_to_gcta(kinship, y,GRM_id,self.data_gen.m)
            unique_id = str(uuid.uuid4()) # generate unique id. This is a version 4 universally unique identifier which is a 128 bit value with random numbers.
            # It's globally unique which means that it's highly improbabe that two systems will generate the same id. Useful for doing these analyses in parallel as the dummy files wont get overwritten.
            
            native_GRM_input_path = unique_id+'.grm.gz'
            native_GRM_id_path = unique_id+'.grm.id'
            native_pheno_path = unique_id+'.phen'
            output_path = unique_id +'_out'
            native_GRM_input.to_csv(native_GRM_input_path,sep = '\t',header=False, index=False,compression = {'method': 'gzip', 'compresslevel': 1})
            GRM_id.to_csv(native_GRM_id_path,sep = '\t',header=False, index=False)
            native_pheno_input.to_csv(native_pheno_path,sep = '\t',header=False, index=False)
            cmd = [
            native_gcta_path, "--reml",
            "--pheno", native_pheno_path,
            "--grm-gz",unique_id,
            "--out", output_path
            ]
            q = subprocess.run(cmd)
            
            hsq_path =output_path+'.hsq'
            log_path =output_path+'.log'
            h2_est, h2_est_se = parse_gcta(output_path+'.hsq') # extract heritability est and SEs. TO DO: compute GCTA SEs in py version and compare.
            num_iters, computation_time = get_descs_from_gcta_log(output_path+'.log') # find iters that AI REMLs converge
            res = dict()
            res['h2_est'] = h2_est
            res['num_iters'] = num_iters
            res['time_elapsed'] = computation_time
            
            if cleanup:
                os.remove(native_GRM_input_path)
                os.remove(native_GRM_id_path)
                os.remove(native_pheno_path)
                
                os.remove(hsq_path)
                os.remove(log_path)
                
            
        else:
            res = do_py_GCTA(y, kinship, tol=tol, iter_limit=iter_limit)
            
        return res
    
    
    def do_HEELS(self,S,R,tol = 1e-4, maxIter = 100,LD_mat_preprocess = True,use_scaled_Z = True,use_yty = False,scale_sigma = False,scale_heels_inputs = False):
        #S = X.T @ y
        #R = X.T @ X
        #original_LD_matrix = R.copy()
        n = self.data_gen.n
        n_tilde = self.data_gen.n_tilde
        m = self.data_gen.m
        if scale_heels_inputs:
            S = S/np.sqrt(n-1)
            R = R/(n-1)
        if LD_mat_preprocess:
            start = time.perf_counter()
            # In the HEELS implementation, the LD matrix undergoes some preprocessing.
            m_ref = R.shape[0]
            n_ref = n_tilde # sim is idealized version where everyone in GWAS had SNP
            R = (R + np.transpose(R)) / 2
            R = R * n_ref / m_ref
            # drop any SNPs that are NA
            idx = np.where(~np.isnan(R).any(axis=0))[0]
            R = R[idx[:, np.newaxis], idx]
            
            n = n # use sample size per SNP for normalization step
            m = R.shape[0]  # Number of SNPs (variants)
            block_m = m # For now, m = block_m in dataset!
            nm_adj = (n/m) / (n_ref/block_m)
            R = R * nm_adj
            end = time.perf_counter()
            tqdm.write('time elapsed for LD preprocessing step: {time}'.format(time = str(end-start)))
        
        # Calculate Z_m (scaled Z-scores) using S, n, and m
        if use_scaled_Z:
            Z_m = np.multiply(S,np.sqrt(n))/np.sqrt(m)
        else:
            Z_m = S
        n = int(np.mean(n))  #redefine n as Average sample size from the dataset.
        sigma2_g = 0.1
        sigma2_e = 0.9
        
        # 1. Run Tannavee's method
        diffs = np.array([99999, 99999], dtype='d')
        counter = 0
        sigma2_g, sigma2_e = sigma2_g, sigma2_e
        start = time.perf_counter()
        while (abs(diffs[0]) > tol) and counter < maxIter:
            if counter % 10 == 0 and counter != 0:
                tqdm.write('finished {num_sims} HEELS iterations'.format(num_sims = counter))
            #tqdm.write(counter,sigma2_g,sigma2_e)
            #tqdm.write(counter,abs(diffs[0]))
            #tqdm.write(counter)
            yty = sigma2_g + sigma2_e # in run_heels(), for some reason it is set to this if you don't supply a yty.
            beta_hat, W = heels_manual.update_BLUP_joint_effect_size(
                Z_m, sigma2_e, sigma2_g, R
            )
            #tqdm.write('beta_hat:')
            #tqdm.write(beta_hat)
            #tqdm.write('old_trace:')
            #tqdm.write(np.trace(np.linalg.inv(W)))
            #tqdm.write('new_trace:')
            #W_band = band_format(R, R.shape[0]).copy()
            #W_band[0,:] = W_band[0,:] + (sigma2_e / sigma2_g) #lam in the paper code
            #chol = cholesky_banded(W_band, lower = True)
            #inv = cho_solve_banded((chol, True), np.eye(m))
            #tqdm.write(np.trace(inv))
        
            # Update sigma2_g
            sigma2_g_new = heels_manual.update_sigma2_g(
                beta_hat, m, W, sigma2_e
            )
        
            # Update sigma2_e
            sigma2_e_new = heels_manual.update_sigma2_e(
                Z_m, S, beta_hat, n,yty, use_yty=use_yty
            )
            
            if scale_sigma:
                sigma2_g_new = sigma2_g_new / (sigma2_g_new + sigma2_e_new)
                sigma2_e_new = sigma2_e_new / (sigma2_g_new + sigma2_e_new)
            
            diffs[0] = sigma2_g_new - sigma2_g
            diffs[1] = sigma2_e_new - sigma2_e
        
            sigma2_g, sigma2_e = sigma2_g_new, sigma2_e_new
            counter += 1
        end = time.perf_counter()
        tqdm.write('time elapsed for HEELS iteration: {time}'.format(time = str(end-start)))
        h2 = sigma2_g / (sigma2_g + sigma2_e)
        res_dict = dict()
        res_dict['sigma2_g'] = sigma2_g
        res_dict['sigma2_e'] = sigma2_e
        res_dict['h2_heels'] = h2
        res_dict['num_iters'] = counter
        return res_dict
        