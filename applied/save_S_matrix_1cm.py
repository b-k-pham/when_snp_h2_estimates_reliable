import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.io as sio
import scipy.stats as stats
import datatable as dt
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from plotnine import *

import matplotlib.pyplot as plt 

import os
import sys


from scipy.sparse import coo_matrix, csr_matrix


from pandas_plink import read_plink
import scipy.sparse as sp
import numpy as np



import pandas_plink

directory = './S_matrices/'
if not os.path.exists(directory):
    os.makedirs(directory,exist_ok=True)

for chrom_num in tqdm(range(1,23)):
    print('Processing chromosome {chrom_num}'.format(chrom_num = chrom_num))
    eur_w_bed_path = '1kg_p1_eur_ben/1kg_p1_eur_chr{chrom_num}'.format(chrom_num = chrom_num)
    bim, _, _ = pandas_plink.read_plink(eur_w_bed_path)
    bim_copy = bim.copy()
    
    
    ### adapted from how ldsc makes blocks.
    coords = bim_copy['cm'].to_numpy()
    m = coords.shape[0]
    j = 0
    block_left = np.zeros(m)
    max_dist = 1 # in LDSC, this is the default 1 cm window.
    
    
    # for a jth SNP, go through all SNPs and find the first ith SNP where distance between ith and jth SNP exceeds max_dist
    for i in range(m):
        while j < m and abs(coords[j] - coords[i]) > max_dist:
            j+= 1
        #print(j)
        block_left[i] = j
    
    bim_copy['ldsc_block_membership'] = block_left
    
    
    # TRUE LDSC IMPLEMENTATION
    import ldscore.ldscore as ld
    import ldscore.parse as ps
    
    
    def __l2_unbiased__(x, n):
        denom = n-2 if n > 2 else n  # allow n<2 for testing purposes
        sq = np.square(x)
        return sq - (1-sq) / denom
    
    
    
    
    array_file, array_obj = eur_w_bed_path+'.bed', ld.PlinkBEDFile
    snp_file, snp_obj = eur_w_bed_path+'.bim', ps.PlinkBIMFile
    ind_file, ind_obj = eur_w_bed_path+'.fam', ps.PlinkFAMFile
    
    array_snps = snp_obj(snp_file)
    m = len(array_snps.IDList)
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)
    keep_snps = None # include all SNPs
    keep_indivs = None # include all individuals
    
    
    
    
    geno_array = array_obj(array_file, n, array_snps, keep_snps=keep_snps,
            keep_indivs=keep_indivs, mafMin=0)
    
    
    coords = np.array(array_snps.df['CM'])[geno_array.kept_snps] #set(bim['cm'].to_numpy() == coords) # TRUE
    
    block_left = ld.getBlockLefts(coords, max_dist)
    
    chunk_size = 50
    block_sizes = np.array(np.arange(m) - block_left)
    block_sizes = np.ceil(block_sizes / chunk_size)*chunk_size
    
    
    
    test = pd.DataFrame(block_left)
    test.columns = ['val']
    test_counts = pd.DataFrame(test['val'].value_counts())
    
    ldsc_ldscores,idx_dict,A_mats,B_mats = geno_array.ldScoreVarBlocks(block_left, chunk_size, annot=None,mode = 'unbiased')
    
    idx_df = pd.DataFrame.from_dict(idx_dict)
    
    
    bim, _, X = read_plink(eur_w_bed_path, verbose=False)
    X = X.T.astype(float)
    mafs = (np.mean(X, axis=0)).compute()
    X -= mafs
    X /= np.std(X, axis=0)

    #for j in tqdm(range(m)):
    #    new_snp = X[:,j].compute()
    #    avg = np.mean(new_snp)
    #    denom = np.std(new_snp)
    #    X[:,j] = (new_snp - avg)/denom
    
    n,m = X.shape
    
    
    ldscores_new = np.zeros(m)
    S_new = np.zeros(shape = (m,m))
    S_review = np.zeros(shape = (m,m))
    S_indicator = np.zeros(shape = (m,m))
    annot = np.ones(m)
    m_affected = 0
    cor_sum_temp = np.zeros(m)
    #S_review = csr_matrix(np.zeros(shape = (m,m)))
    #S_new = csr_matrix(np.zeros(shape = (m,m)))
    #S_indicator = csr_matrix(np.zeros(shape = (m,m)))

    for i in range(idx_df.shape[0]):
        #A_temp = A_mats[i]
        #B_temp = B_mats[i]
        A_from_idx = idx_df['A_from'][i]
        A_to_idx = idx_df['A_to'][i]
        B_from_idx = idx_df['B_from'][i]
        B_to_idx = idx_df['B_to'][i]

        A_temp = X[:,A_from_idx:A_to_idx].compute()
        B_temp = X[:,B_from_idx:B_to_idx].compute()

        cor_sum_temp[A_from_idx:A_to_idx] += np.dot(__l2_unbiased__((A_temp.T @ B_temp)/(n),n),annot[B_from_idx:B_to_idx])
        S_review[A_from_idx:A_to_idx,B_from_idx:B_to_idx] = __l2_unbiased__((A_temp.T @ B_temp)/(n),n)
        S_new[A_from_idx:A_to_idx,B_from_idx:B_to_idx] = (A_temp.T @ B_temp)/(n)

        # Indicator matrix which shows what SNPs are involved in block computations
        S_indicator[A_from_idx:A_to_idx,B_from_idx:B_to_idx] =   1
        
        if i > 0:
            cor_sum_temp[B_from_idx:B_to_idx] += np.dot(annot[A_from_idx:A_to_idx],__l2_unbiased__((A_temp.T @ B_temp)/(n),n))
            S_review[B_from_idx:B_to_idx,A_from_idx:A_to_idx] = __l2_unbiased__(((A_temp.T @ B_temp)/(n)).T,n)
            S_new[B_from_idx:B_to_idx,A_from_idx:A_to_idx] = ((A_temp.T @ B_temp)/(n)).T

            S_indicator[B_from_idx:B_to_idx,A_from_idx:A_to_idx] =  1
            
            cor_sum_temp[B_from_idx:B_to_idx] += np.dot(__l2_unbiased__((B_temp.T @ B_temp)/(n),n),annot[B_from_idx:B_to_idx])
            S_review[B_from_idx:B_to_idx,B_from_idx:B_to_idx] = __l2_unbiased__((B_temp.T @ B_temp)/(n),n)
            S_new[B_from_idx:B_to_idx,B_from_idx:B_to_idx] = (B_temp.T @ B_temp)/(n)
            S_indicator[B_from_idx:B_to_idx,B_from_idx:B_to_idx] =  1
            
    print ('finished making S_new')
    
    chrom_file_name = 'S_chrom{chrom_num}_1cm.npz'.format(chrom_num = chrom_num)
    S_new = csr_matrix(S_new)
    mks = np.array(S_indicator.sum(axis = 0)).flatten()
    
    np.savez_compressed(directory+'{chrom_file_name}'.format(chrom_file_name = chrom_file_name), S_tilde =S_new, mks =mks)