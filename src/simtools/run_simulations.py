"""

These are functions used to conduct GWASH, LD Score Regression, and GCTA
Heritability Estimation comparisons in simulation as shown in Pham et al. 2025.

This specific file contains simulations that were run.
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

from src.simtools.data_generation import data_generation as data_gen
from src.simtools.data_preprocessing import data_preprocessing
from src.simtools.do_analysis import do_analysis


class run_simulations:
    def __init__(self,data_gen,ref_data_gen = None,scaleX = True,scaley = True,nPCs = None,num_sims = 100,regress_PC_out = False,regress_X_on_PC = True,regress_y_on_PC = True,calc_mu_hat_2_fast = True,multithreading = True,n_jobs = 20,realistic = False,run_gcta = True,reml_max_iters = 100,reml_tol = 1e-4,use_native_gcta = False,native_gcta_path = None,seed = None,debug = False,track_progress = False,batch_size = 1):
        self.data_gen = data_gen
        if ref_data_gen is None: #if there is no reference panel, assume that it is the same.
            self.ref_data_gen = self.data_gen
        else:
            self.ref_data_gen = ref_data_gen
        self.num_sims = num_sims
        self.scaleX = scaleX
        self.scaley = scaley
        self.nPCs = nPCs
        self.regress_PCs_out = regress_PC_out
        self.regress_X_on_PC = regress_X_on_PC
        self.regress_y_on_PC = regress_y_on_PC
        
        self.calc_mu_hat_2_fast = calc_mu_hat_2_fast
        
        self.multithreading = multithreading
        self.n_jobs = n_jobs
        self.realistic = realistic
        self.batch_size = batch_size
        
        # GCTA related
        self.run_gcta =  run_gcta # Run GCTA? This only matters for generating publication simulations.
        self.use_native_gcta = use_native_gcta # use the released executable gcta? this is magnitudes faster since C++ is 10x faster than python
        self.native_gcta_path = native_gcta_path # path to gcta executable. only matters if use_native_gcta == True.
        self.reml_max_iters = reml_max_iters
        self.reml_tol = reml_tol
        self.seed = seed # controls seed if specified in sims.
        
        
        #debug option
        self.debug = debug
        self.track_progress = track_progress
        
        
    def _run_simulations(self, sim_func, batch_size=1):
        """
        Helper to run simulations with or without multithreading, batching, and progress tracking.
        If batch_size > 1, each job runs a batch of simulations to reduce process overhead.
        """
        if batch_size > 1:
            # Split indices into batches
            batches = [range(i, min(i + batch_size, self.num_sims)) for i in range(0, self.num_sims, self.batch_size)]
            iterator = tqdm(batches) if self.track_progress else batches
            if self.multithreading:
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(self.run_simulation_batch)(batch) for batch in iterator
                )
            else:
                results = [self.run_simulation_batch(batch) for batch in iterator]
            # Flatten results
            results = [item for sublist in results for item in sublist]
            return pd.concat(results, axis=0).reset_index(drop=True)
        else:
            iterator = tqdm(range(self.num_sims)) if self.track_progress else range(self.num_sims)
            if self.multithreading:
                res_dfs = Parallel(n_jobs=self.n_jobs, backend='loky')(delayed(sim_func)(i) for i in iterator)
            else:
                res_dfs = [sim_func(i) for i in iterator]
            return pd.concat(res_dfs, axis=0).reset_index(drop=True)

    def run_simulation_batch(self, batch_indices):
        """
        Run a batch of simulations given a list or range of indices.
        Returns a list of results (dicts or DataFrames).
        """
        return [self.run_single_simulation(i) for i in batch_indices]

    def num_sims_simulations_sstats(self, batch_size=1):
        return self._run_simulations(self.run_single_simulation_sstats, batch_size=self.batch_size)
    def num_sims_simulations_genotype(self, batch_size=1):
        return self._run_simulations(self.run_single_simulation_genotype, batch_size=self.batch_size)
    def num_sims_simulations_publication(self, batch_size=1):
        return self._run_simulations(self.run_single_simulation_publication, batch_size=self.batch_size)


    def preprocess(self, X, y):
        """Helper for PC regression and scaling."""
        my_data_preprocessing = data_preprocessing(self.data_gen, self.nPCs, self.regress_X_on_PC, self.regress_y_on_PC)
        if self.regress_PCs_out:
            regress_res = my_data_preprocessing.regress_out_PCs(X, y)
            X, y = regress_res['X_res'], regress_res['y_res']
        X_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(X) if self.scaleX else X
        y_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(y) if self.scaley else y
        return X, y, X_tilde, y_tilde
    
    def single_simulation_sstats(self,i):
        ### RUN ONLY SUMMARY STATISTICS ANALYSES!
        logging.info('Starting simulation {i}'.format(i = i))
        if self.seed is not None:
            seed_id = self.seed + i
            np.random.seed(seed_id)
        res_dict = dict()
        start = time.perf_counter()
        #X,b,f,num_causal = self.data_gen.gen_X(realistic = self.realistic)
        X,b = self.data_gen.gen_X(realistic = self.realistic)
        y = self.data_gen.gen_y(X,b)
        
        
        my_data_preprocessing = data_preprocessing(self.data_gen,nPCs = self.nPCs,regress_X_on_PC = self.regress_X_on_PC,regress_y_on_PC = self.regress_y_on_PC)
        if self.regress_PCs_out:
            regress_res = my_data_preprocessing.regress_out_PCs(X,y)
            X = regress_res['X_res']
            y = regress_res['y_res']
            
        h2_samp_fve =  np.var(np.matmul(X,b),ddof=1)/np.var(y,ddof = 1)
        if self.scaleX:
            X_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(X)
        else:
            X_tilde = X
        if self.scaley:
            y_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(y)
        else:
            y_tilde = y
        # if self.data_gen.ref_ldscores is None, then generate ldscores. Otherwise, use self.data_gen.ref_ldscores
        u2,s2,ldscores = self.data_gen.compute_u2_ldscores(X_tilde,y_tilde)
        
        ldsc_reg_weights = ldscores
        
        
        end = time.perf_counter()
        logging.info('time to generate data: {time}'.format(time = str(end - start)))
        
        sims = do_analysis(self.data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        sims_fixed = do_analysis(self.data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,intercept = 1,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        
        start = time.perf_counter()
        
        h2_ldsc_reg,se_ldsc_reg,icpt_ldsc_reg,weights = sims.do_ldsc()
        h2_ldsc_fixed,se_ldsc_fixed,icpt_ldsc_fixed,weights_fixed = sims_fixed.do_ldsc()
        
        icpt_ols_reg,h2_ols_reg = sims.do_ols()
        h2_ols_fixed = sims_fixed.do_ols()
        
        
        ### RECOMMENDED PRACTICE WITH LDSC IF HIGH INTERCEPT IS TO DIVIDE THE CHI-SQUAREDS BY THE INTERCEPT AND RUN AGAIN TO CORRECT CONFOUNDING
        u2_corrected = u2/icpt_ldsc_reg
        s2_corrected = np.mean(u2_corrected)
        sims_corrected = do_analysis(self.data_gen,X_tilde,u2_corrected,s2_corrected,ldscores,ldsc_reg_weights,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        h2_ldsc_reg_corrected,se_ldsc_reg_corrected,icpt_ldsc_reg_corrected,weights_corrected = sims_corrected.do_ldsc()
        
        end = time.perf_counter()
        #tqdm.write('time to do ldsc and ols + correction analysis: {time}'.format(time = str(end - start)))
        
        start = time.perf_counter()
        
        gwash_res = sims.do_GWASH_from_ldscores() # in practice, compute from ldscores only.
        gwash_res_sample_theoretical = sims.do_GWASH_from_X_tilde() # Do same calculation from X_tilde() just to get theoretical variance.
        
        
        ### mu2_hat and mu3_hat used for variance calculation
        gwash_mu2_hat = gwash_res_sample_theoretical['mu2_hat']
        gwash_mu3_hat = gwash_res_sample_theoretical['mu3_hat']
        
        #print('before')
        #print('gwash_mu2_hat')
        #print(gwash_mu2_hat)
        #print('gwash_mu3_hat')
        #print(gwash_mu3_hat)
        
        # if reference mu2 hat and reference mu3 hat is provided, use that instead.
        if self.data_gen.ref_mu2_hat is not None:
            gwash_mu2_hat = self.data_gen.ref_mu2_hat
            
        if self.data_gen.ref_mu3_hat is not None:
            gwash_mu3_hat = self.data_gen.ref_mu3_hat
            
        #print('after')
        #print('gwash_mu2_hat')
        #print(gwash_mu2_hat)
        #print('gwash_mu3_hat')
        #print(gwash_mu3_hat)

        w_gwash_res = sims.do_weighted_GWASH_from_ldscores(gwash = gwash_res['gwash'])
        
        
        GWASH_se_sample_theoretical = calc_gwash_se(self.data_gen.n,self.data_gen.m,gwash_mu2_hat,gwash_mu3_hat,gwash_res['gwash'])
        
        
        end = time.perf_counter()
        
        
        if self.debug: # if debug flag is on, save objects. For example, I want to save ldscores and u2.
            with open('sim_debug_data/X_y_b_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, X)
                np.save(f, y)
                np.save(f, b)
            with open('sim_debug_data/X_tilde_y_tilde_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, X_tilde)
                np.save(f, y_tilde)
            with open('sim_debug_data/ldscores_u2_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, ldscores)
                np.save(f, u2)
            
    
        res_dict['h2_gwash'] = gwash_res['gwash']
        res_dict['h2_weighted_gwash'] = w_gwash_res
        res_dict['h2_ols_reg'] = h2_ols_reg
        res_dict['icpt_ols_reg'] = icpt_ols_reg
        res_dict['h2_ols_fixed'] = h2_ols_fixed
        res_dict['h2_ldsc_reg'] = h2_ldsc_reg
        res_dict['icpt_ldsc_reg'] = icpt_ldsc_reg
        res_dict['h2_ldsc_reg_corrected'] = h2_ldsc_reg_corrected
        res_dict['icpt_ldsc_reg_corrected'] = icpt_ldsc_reg_corrected
        res_dict['h2_ldsc_fixed'] = h2_ldsc_fixed
        res_dict['h2_samp'] = h2_samp_fve
        res_dict['h2_gwash_sample_theoretical_se'] = GWASH_se_sample_theoretical
        #res_dict['h2_gwash_true_theoretical_var'] = GWASH_var_true_theoretical
        res_dict['h2_ldsc_reg_jackknife_var'] = se_ldsc_reg
        res_dict['h2_ldsc_fixed_jackknife_var'] = se_ldsc_fixed
        logging.info('Finished simulation {i}'.format(i = i))
        #time.sleep(2) # + 2s so that the message can tqdm.write per chunk.
        return res_dict
    def single_simulation_genotype(self,i):
        ### RUN ONLY GENOTYPE (FULL DATA) ANALYSES!
        logging.info('Starting simulation {i}'.format(i = i))
        if self.seed is not None:
            seed_id = self.seed + i
            np.random.seed(seed_id)
        res_dict = dict()
        start = time.perf_counter()
        X,b = self.data_gen.gen_X(realistic = self.realistic)
        y = self.data_gen.gen_y(X,b)
        
        
        my_data_preprocessing = data_preprocessing(self.data_gen,nPCs = self.nPCs,regress_X_on_PC = self.regress_X_on_PC,regress_y_on_PC = self.regress_y_on_PC)
        if self.regress_PCs_out:
            regress_res = my_data_preprocessing.regress_out_PCs(X,y)
            X = regress_res['X_res']
            y = regress_res['y_res']
            
        h2_samp_fve =  np.var(np.matmul(X,b),ddof=1)/np.var(y,ddof = 1)
        if self.scaleX:
            X_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(X)
        else:
            X_tilde = X
        if self.scaley:
            y_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(y)
        else:
            y_tilde = y
        # if self.data_gen.ref_ldscores is None, then generate ldscores. Otherwise, use self.data_gen.ref_ldscores
        u2,s2,ldscores = self.data_gen.compute_u2_ldscores(X_tilde,y_tilde)
        
        ldsc_reg_weights = ldscores
        
        GRM,GRM_id,pheno = self.data_gen.create_GCTA_inputs(X_tilde,y_tilde) # GCTA related inputs
        input_GRM,input_pheno = self.data_gen.format_gcta_to_mats(GRM,pheno)
        
        end = time.perf_counter()
        logging.info('time to generate data: {time}'.format(time = str(end - start)))
        
        sims = do_analysis(self.data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        sims_fixed = do_analysis(self.data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,intercept = 1,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        

        start = time.perf_counter()
        
        GCTA_res = sims.do_GCTA(input_pheno,input_GRM,GRM_id,tol = self.reml_tol, iter_limit = self.reml_max_iters,use_native_gcta = self.use_native_gcta,native_gcta_path = self.native_gcta_path)

        end = time.perf_counter()
        logging.info('time to do GCTA: {time}'.format(time = str(end - start)))
        
        
        if self.debug: # if debug flag is on, save objects. For example, I want to save ldscores and u2.
            with open('sim_debug_data/X_y_b_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, X)
                np.save(f, y)
                np.save(f, b)
            with open('sim_debug_data/X_tilde_y_tilde_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, X_tilde)
                np.save(f, y_tilde)
            with open('sim_debug_data/ldscores_u2_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, ldscores)
                np.save(f, u2)
                
            # MAKE GCTA FILES HERE
            sim_id_header = 'sim_debug_data/seed{seed_num}'.format(seed_num = seed_id)
            self.data_gen.create_GCTA_inputs(X_tilde,y_tilde,sim_id_header,True)
            
        res_dict['VG'] = GCTA_res['VG']
        res_dict['Ve'] = GCTA_res['Ve']
        res_dict['h2_gcta'] = GCTA_res['h2_est']
        res_dict['h2_samp'] = h2_samp_fve
        logging.info('Finished simulation {i}'.format(i = i))
        return res_dict
        
    def run_single_simulation_sstats(self,i):
        return pd.DataFrame.from_dict(self.single_simulation_sstats(i), orient='index').T
    
    def run_single_simulation_genotype(self,i):
        return pd.DataFrame.from_dict(self.single_simulation_genotype(i), orient='index').T
    
    def single_simulation_publication(self,i):
        ### RUN ANALYSES RELATED TO FIGURES IN PAPER!
        logging.info('Starting simulation {i}'.format(i = i))
        if self.seed is not None:
            seed_id = self.seed + i
            np.random.seed(seed_id)
        res_dict = dict()
        start = time.perf_counter()
        X,b = self.data_gen.gen_X(realistic = self.realistic)
        y = self.data_gen.gen_y(X,b)
        
        
        my_data_preprocessing = data_preprocessing(self.data_gen,nPCs = self.nPCs,regress_X_on_PC = self.regress_X_on_PC,regress_y_on_PC = self.regress_y_on_PC)
        if self.regress_PCs_out:
            regress_res = my_data_preprocessing.regress_out_PCs(X,y)
            X = regress_res['X_res']
            y = regress_res['y_res']
            
        h2_samp_fve =  np.var(np.matmul(X,b),ddof=1)/np.var(y,ddof = 1)
        if self.scaleX:
            X_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(X)
        else:
            X_tilde = X
        if self.scaley:
            y_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(y)
        else:
            y_tilde = y
        # if self.data_gen.ref_ldscores is None, then generate ldscores. Otherwise, use self.data_gen.ref_ldscores
        
        u2,s2,ldscores = self.data_gen.compute_u2_ldscores(X_tilde,y_tilde)
        
        ldsc_reg_weights = ldscores
        
        if self.run_gcta:
            GRM,GRM_id,pheno = self.data_gen.create_GCTA_inputs(X_tilde,y_tilde) # GCTA related inputs
            input_GRM,input_pheno = self.data_gen.format_gcta_to_mats(GRM,pheno)
        
        end = time.perf_counter()
        
        sims = do_analysis(self.data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        sims_fixed = do_analysis(self.data_gen,X_tilde,u2,s2,ldscores,ldsc_reg_weights,intercept = 1,calc_mu_hat_2_fast = self.calc_mu_hat_2_fast)
        
        start = time.perf_counter()
        
        h2_ldsc_reg,se_ldsc_reg,icpt_ldsc_reg,weights = sims.do_ldsc()
        h2_ldsc_fixed,se_ldsc_fixed,icpt_ldsc_fixed,weights_fixed = sims_fixed.do_ldsc()
        
        
        end = time.perf_counter()
        
        start = time.perf_counter()
        
        gwash_res = sims.do_GWASH_from_ldscores() # in practice, compute from ldscores only.
        w_gwash_res = sims.do_weighted_GWASH_from_ldscores(gwash = gwash_res['gwash'])
        
        gwash_res_sample_theoretical = sims.do_GWASH_from_X_tilde() # Do same calculation from X_tilde() just to get theoretical variance.
        
        ### mu2_hat and mu3_hat used for variance calculation
        gwash_mu2_hat = gwash_res_sample_theoretical['mu2_hat']
        gwash_mu3_hat = gwash_res_sample_theoretical['mu3_hat']
        
        #print('before')
        #print('gwash_mu2_hat')
        #print(gwash_mu2_hat)
        #print('gwash_mu3_hat')
        #print(gwash_mu3_hat)
        
        # if reference mu2 hat and reference mu3 hat is provided, use that instead.
        if self.data_gen.ref_mu2_hat is not None:
            gwash_mu2_hat = self.data_gen.ref_mu2_hat
            
        if self.data_gen.ref_mu3_hat is not None:
            gwash_mu3_hat = self.data_gen.ref_mu3_hat
            
        #print('after')
        #print('gwash_mu2_hat')
        #print(gwash_mu2_hat)
        #print('gwash_mu3_hat')
        #print(gwash_mu3_hat)
        
        GWASH_se_sample_theoretical = calc_gwash_se(self.data_gen.n,self.data_gen.m,gwash_mu2_hat,gwash_mu3_hat,gwash_res['gwash'])
        
        
        end = time.perf_counter()
        #tqdm.write('time to do GWASH: {time}'.format(time = str(end - start)))
        
        
        if self.run_gcta:
            start = time.perf_counter()
            
            GCTA_res = sims.do_GCTA(input_pheno,input_GRM,GRM_id,tol = self.reml_tol, iter_limit = self.reml_max_iters,use_native_gcta = self.use_native_gcta,native_gcta_path = self.native_gcta_path)
            #S = X.T @ y
            #R = X.T @ X
            
            end = time.perf_counter()
            #tqdm.write('time to do GCTA: {time}'.format(time = str(end - start)))

        if self.debug: # if debug flag is on, save objects. For example, I want to save ldscores and u2.
            with open('sim_debug_data/X_y_b_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, X)
                np.save(f, y)
                np.save(f, b)
            with open('sim_debug_data/X_tilde_y_tilde_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, X_tilde)
                np.save(f, y_tilde)
            with open('sim_debug_data/ldscores_u2_seed{seed_num}.npy'.format(seed_num = seed_id), 'wb') as f:
                np.save(f, ldscores)
                np.save(f, u2)
            
        if self.run_gcta:
            res_dict['h2_gcta'] = GCTA_res['h2_est']
            if self.debug:
                res_dict['gcta_iters'] = GCTA_res['num_iters']
        res_dict['h2_gwash'] = gwash_res['gwash']
        res_dict['h2_ldsc_reg'] = h2_ldsc_reg
        res_dict['icpt_ldsc_reg'] = icpt_ldsc_reg
        res_dict['h2_ldsc_fixed'] = h2_ldsc_fixed
        res_dict['h2_samp'] = h2_samp_fve
        res_dict['h2_gwash_sample_theoretical_se'] = GWASH_se_sample_theoretical
        res_dict['h2_ldsc_reg_jackknife_se'] = se_ldsc_reg # since it is an estimate of the variance component, I can just square the SE to get the variance apparently
        res_dict['h2_ldsc_fixed_jackknife_se'] = se_ldsc_fixed
        logging.info('Finished simulation {i}'.format(i = i))
        return res_dict
        
    def run_single_simulation_publication(self,i):
        return pd.DataFrame.from_dict(self.single_simulation_publication(i), orient='index').T
        
        
    def debug_ref(self,i): # prove that the reference panel works.
        ### RUN ANALYSES RELATED TO FIGURES IN PAPER!
        logging.info('Starting simulation {i}'.format(i = i))
        if self.seed is not None:
            seed_id = self.seed + i
            np.random.seed(seed_id)
        res_dict = dict()
        start = time.perf_counter()
        #X,b,f,num_causal = self.data_gen.gen_X(realistic = self.realistic)
        X,b = self.data_gen.gen_X(realistic = self.realistic)
        y = self.data_gen.gen_y(X,b)
        
        
        my_data_preprocessing = data_preprocessing(self.data_gen,nPCs = self.nPCs,regress_X_on_PC = self.regress_X_on_PC,regress_y_on_PC = self.regress_y_on_PC)
        if self.regress_PCs_out:
            regress_res = my_data_preprocessing.regress_out_PCs(X,y)
            X = regress_res['X_res']
            y = regress_res['y_res']
            
        h2_samp_fve =  np.var(np.matmul(X,b),ddof=1)/np.var(y,ddof = 1)
        if self.scaleX:
            X_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(X)
        else:
            X_tilde = X
        if self.scaley:
            y_tilde = my_data_preprocessing.manual_standardize_and_scale_inR(y)
        else:
            y_tilde = y
        
        u2,s2,ldscores = self.data_gen.compute_u2_ldscores(X_tilde,y_tilde)
        
        if self.data_gen.ref_X is None:
            X_ref = X_tilde
        else:
            X_ref = self.data_gen.ref_X
        
        return {'seed_id':seed_id,'h2_samp_fve':h2_samp_fve,'ldscores':ldscores,'X_tilde':X_tilde,'X_ref':X_ref}
        
    def run_debug_ref(self,i):
        return pd.DataFrame.from_dict(self.debug_ref(i), orient='index').T
    
