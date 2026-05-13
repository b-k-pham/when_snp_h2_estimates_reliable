import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.io as sio
import scipy.stats as stats
import datatable as dt
from tqdm.notebook import tqdm


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from plotnine import *

import matplotlib.pyplot as plt 

import os
import sys



#import os
#os.environ["MKL_INTERFACE_LAYER"] = "ILP64"


from pandas_plink import read_plink
import scipy.sparse as sp
import numpy as np


import pandas_plink

#chrom_num = 22
dfs = []
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
    X = X.T
    mafs = (np.mean(X, axis=0)).compute()
    X -= mafs
    X /= np.std(X, axis=0)

    bim_copy = bim.copy()

    
    n,m = X.shape
    loaded = np.load('./S_matrices/S_chrom{chrom_num}_1cm.npz'.format(chrom_num = chrom_num),allow_pickle=True)
    S_new = loaded['S_tilde'].tolist()
    mks = loaded['mks']
    S_new2 = S_new @ S_new
    l2s = (S_new2.diagonal() - (mks)/(n-1)) * ((n-1)/(n-2))
    l3s = (S_new2 @ S_new).diagonal() - 3*(mks)/(n-1) - (mks**2)/(n-1)**2) * ((n-1)/(n-2))


    bim_copy['gwash_l2s'] = l2s
    bim_copy['gwash_l3s'] = l3s
    bim_copy['ldsc_l2s'] = ldsc_ldscores
    ldsc_ldscores_saved = pd.read_table('eur_w_ld_chr_ben/{chrom_num}.l2.ldscore.gz'.format(chrom_num = chrom_num))['L2']
    bim_copy['ldsc_l2s_saved'] = ldsc_ldscores_saved 
    bim_copy['abs_diff_gwash_l2s_and_ldsc_l2s'] = abs(bim_copy['gwash_l2s'] - bim_copy['ldsc_l2s'])
    # saved LD Scores from ldsc.py (what's actually read into LDSC) are rounded to the third decimal place
    bim_copy['abs_diff_gwash_l2s_and_ldsc_l2s_saved'] = abs(np.round(bim_copy['gwash_l2s'],3) - bim_copy['ldsc_l2s_saved'])
    

    bim_copy.to_csv('./S_tabs/S_chrom{chrom_num}_1cm_bim.txt'.format(chrom_num = chrom_num),index = None,sep = '\t')