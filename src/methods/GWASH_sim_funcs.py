import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
from scipy.sparse import coo_matrix,coo_array,csr_matrix
#from sparse_dot_mkl import dot_product_mkl
import time



from src.methods.GWASH_funcs import *
from src.methods.ldsc_barebones import *




#def init_process(m,n,k,rho,h2_pop):
#    Sigma_tilde = ar1_cor(m,rho)
#    covar_test = np.matmul(np.diag(np.full(m,range(1,m+1)))**(1/2),np.matmul(Sigma_tilde,np.diag(np.full(m,range(1,m+1)))**(1/2)))

#    X = stats.multivariate_normal.rvs(mean = np.repeat(0,m),cov = covar_test, size = n)

#    myX_star = X
#    b = np.random.normal(loc=0, scale=1, size=m) # 

#    denom = calc_heritability_total(b,covar_test,h2_pop)


#    arr = range(m)
#    arr_split = [z.tolist() for z in np.array_split(arr, k)]


#    SNP_groups = OrderedDict()

#    for i in range(1,k+1):
#        label = 'g'+str(i)
#        SNP_groups[label] = arr_split[i-1]
#    res = (myX_star,b,covar_test,denom,SNP_groups)
#    return(res)

def calc_u_from_sumstat(t,n):
    denom = 1 + ((t**2)/(n-2))
    right = t**2/denom
    left = (n-1)/(n-2)
    return left*right
    
def ar1_cor(m,rho):
        """Generate true AR1 m x m correlation matrix. Useful for theoretical computations/debugging.
        
           Parameters
           ----------
           Uses m and rho in data_generation class
           Returns
           -------
           ndarray of shape (m,m) following rho correlation structure
        """
        init = np.ones([m,m])
        for i in range(m):
            for j in range(m):
                init[i,j] = abs(j - i)
        return(rho**init)


def get_SNP_BP(chrom,sparse_idx):
    '''
    Input:
    chrom - chromosome to query
    sparse_idx: dataframe consisting of rsid, idx in original LD matrix, and idx in new subset matrix
    Output:
    SNP_BP: dataframe with rsids as index, BP of rsids, and idx in new subset matrix
    '''

    SNP_idx = dict()

    for z in range(len(sparse_idx['rsid'])):
        SNP_idx[sparse_idx['rsid'][z]] = sparse_idx['subset_idx'][z]

    df = dt.fread('ldsc_RA_by_chr/check_ldsc_RA_chr'+str(chrom)+'.ld').to_pandas()
    BPs = df['BP_A'].tolist() + df['BP_B'].tolist()
    SNPs = df['SNP_A'].tolist() +df['SNP_B'].tolist()
    SNP_BP = dict()
    for z in range(len(SNPs)):
        SNP_BP[SNPs[z]] = BPs[z] 
    SNP_BP = pd.DataFrame.from_dict(SNP_BP,orient = 'index')
    SNP_BP.columns = ['BP']
    SNP_BP = SNP_BP.sort_values('BP')
    SNP_BP['idx'] = [SNP_idx[z] for z in SNP_BP.index]
    
    return SNP_BP


def gwash_preprocessing(chrom,sumstat_ld_reg,n,sparse_path = 'ld_RA_by_chr_sparsemats/'):
    '''
    Input: 
    chrom: chromosome to query,
    sumstat_ld_reg: processed sumstat (same QC as ldsc),
    n: GWAS Summary Statistic Sample Size,
    sparse_path: path to ld matrix by chr sparse matrices
    
    Finds common SNPs between the sumstat and the LD matrix per chromosome.
    Subsets this LD matrix to only include the common SNPs.
    
    Output: 
    mk number of SNPs in Subsetted LD Matrix per chr
    Subsetted ld matrix of SNPs in sumstat
    s2 used in GWASH Calculation
    '''
    sumstat_ld_reg_chr = sumstat_ld_reg[sumstat_ld_reg['CHR_x'] == chrom]

    fname = sparse_path+'chr'+str(chrom)+'_idx.csv'
    sparse_idx = pd.read_csv(fname)

    sparse_idx.index = sparse_idx['rsid']
    sumstat_ld_reg_chr.index = sumstat_ld_reg_chr['SNP']

    # get SNPs that are in common with SUMSTAT and 1KG Panel
    common_snps = list(set(sumstat_ld_reg['SNP']).intersection(set(sparse_idx.index)))

    sumstat_ld_reg_chr = sumstat_ld_reg_chr.loc[common_snps]
    #del(sumstat_ld_reg_chr)
    #del(sumstat_ld_reg)
    sparse_idx = sparse_idx.loc[common_snps].sort_values(by='idx')
    A = scipy.sparse.load_npz("ld_RA_by_chr_sparsemats/chr"+str(chrom)+".npz")
    subset_idx = sparse_idx['idx'].tolist()
    A_subset = A[subset_idx,:][:,subset_idx]
    
    sparse_idx['subset_idx'] = [z for z in range(A_subset.shape[0])]
    u2j = np.array([calc_u_from_sumstat(z,n) for z in sumstat_ld_reg_chr['Z']])
    s2 = np.mean(u2j)
    
    #SNP_BP = get_SNP_BP(chrom,sparse_idx) # this takes too long.
    SNP_BP = [z for z in range(A_subset.shape[0])]
    return A_subset.shape[0],A_subset,u2j,SNP_BP


def calc_I2_mats(A_mat,calc_non_zero = True):
    I2_rows,I2_cols = get_nonzero_off_idx_from_sparse(A_mat) # only get i,js off the diagonal and non-zero
    data = np.repeat(1, len(I2_rows))
    I2_mat = coo_matrix((data, (I2_rows, I2_cols)),
                       shape=(A_mat.shape[0],
                              A_mat.shape[1]),
                       dtype=np.float64).tocsr()
    if calc_non_zero:
        A_mat_non_zero = np.squeeze(np.asarray(A_mat[I2_rows,I2_cols]))
        res = {'I2_mat':I2_mat,
               'A_mat_non_zero': A_mat_non_zero}
    else:
        res = {'I2_mat':I2_mat}
    return res


def calc_I2_I3_mats(A_mat):
    I2_rows,I2_cols = get_nonzero_off_idx_from_sparse(A_mat) # only get i,js off the diagonal and non-zero
    data = np.repeat(1, len(I2_rows))
    I2_mat = coo_matrix((data, (I2_rows, I2_cols)),
                       shape=(A_mat.shape[0],
                              A_mat.shape[1]),
                       dtype=np.float64).tocsr()
    A_mat_non_zero = np.squeeze(np.asarray(A_mat[I2_rows,I2_cols]))
    #I2I2_mat = dot_product_mkl(I2_mat, I2_mat, dense=False)
    I2I2_mat = I2_mat @ I2_mat # MEMORY ERROR OR WILL TAKE A VERY LONG TIME TO RUN. Need dot_product_mkl() from intel math library.
    I3_mat = I2_mat.multiply(I2I2_mat)
    res = {'I2_mat':I2_mat,
           'A_mat_non_zero': A_mat_non_zero,
          'I2I2_mat':I2I2_mat,
          'I3_mat':I3_mat}
    return res


def get_nonzero_off_diags_from_sparse(smat):
    row_indices, col_indices = smat.nonzero()
    off_diagonal_indices = np.where(row_indices != col_indices)[0]
    off_diagonal_values = smat[row_indices[off_diagonal_indices], col_indices[off_diagonal_indices]]
    return np.squeeze(np.asarray(off_diagonal_values))


def get_nonzero_off_idx_from_sparse(smat):
    row_indices, col_indices = smat.nonzero()
    off_diagonal_indices = np.where(row_indices != col_indices)[0]
    off_diagonal_values = smat[row_indices[off_diagonal_indices], col_indices[off_diagonal_indices]]
    return row_indices[off_diagonal_indices], col_indices[off_diagonal_indices]


def calc_gwash_est_only(S_tilde,s2,n_sumstat,n_tilde = 503):
    mk = S_tilde.shape[0] # number of SNPs in S_tilde
    mats = calc_I2_mats(S_tilde)
    n_I2 = mats['I2_mat'].sum() # number of ijs such that 1 < |i-j| <= q; get I2 indicator matrix and sum elements
    non_zero_off_diags_S_tilde = mats['A_mat_non_zero']
    mu2_hat_ld_nonzero = (np.sum(non_zero_off_diags_S_tilde**2)) - (n_I2/(n_tilde-1))
    mu2_hat = 1 + ((1/mk) * mu2_hat_ld_nonzero)
    gwash = (mk/(n_sumstat*mu2_hat)) * (s2 - 1)
    return {'gwash':gwash,'mu2_hat':mu2_hat}

#Calculate_u2 = function(X,y, scale.x = T, scale.y=T){
  
  ## Input: 
  ## X: An n*m SNP matrix
  ## y: A vector of phenotypes (size n)
  ## Output:
  ##A vector of squared u scores
  
#  if(scale.x == T){
#    X = scale(X)
#  }
  
#  if(scale.y==T){
#    y = scale(y)
#  }
  
#  return( as.vector(((t(X) %*% y)/sqrt(NROW(X)-1)))^2)
  
#}

def calc_ldscore(X):
    mat = (np.matmul(X.T,X)/(X.shape[0] -1))**2
    return mat.sum(axis = 0)

def calc_ldscore_from_Xcorr(Xcorr):
    #mat = (np.matmul(X.T,X)/(X.shape[0] -1))**2
    return (Xcorr**2).sum(axis = 0)

def calc_u2(X,y):
    u = (X.T @ y)/np.sqrt(X.shape[0]-1)
    return u**2


def calc_corrM_from_covM(covM):
    A = np.zeros((covM.shape[0],covM.shape[1]))
    for i in range(covM.shape[0]):
        bot_i = covM[i,i]
        for j in range(covM.shape[1]):
            bot_j = covM[j,j]
            A[i,j] = covM[i,j]/np.sqrt(bot_i * bot_j)
    return A

def manual_standardize(X):
    return (np.sqrt(X.shape[0] - 1)* X) / np.linalg.norm(X,axis = 0)[np.newaxis, :]

def manual_standardize_and_scale_inR(X):
    X_centered = X - np.mean(X,axis = 0).reshape(1,-1)
    return X_centered/np.std(X_centered,ddof = 1,axis = 0)
    
def calc_I2_I3_mats(A_mat):
    I2_rows,I2_cols = get_nonzero_off_idx_from_sparse(A_mat) # only get i,js off the diagonal and non-zero
    data = np.repeat(1, len(I2_rows))
    I2_mat = coo_matrix((data, (I2_rows, I2_cols)),
                       shape=(A_mat.shape[0],
                              A_mat.shape[1]),
                       dtype=np.float64).tocsr()
    A_mat_non_zero = np.squeeze(np.asarray(A_mat[I2_rows,I2_cols]))
    #I2I2_mat = dot_product_mkl(I2_mat, I2_mat, dense=False)
    I2I2_mat = I2_mat @ I2_mat # MEMORY ERROR OR WILL TAKE A VERY LONG TIME TO RUN. Need dot_product_mkl() from intel math library.
    I3_mat = I2_mat.multiply(I2I2_mat)
    res = {'I2_mat':I2_mat,
           'A_mat_non_zero': A_mat_non_zero,
          'I2I2_mat':I2I2_mat,
          'I3_mat':I3_mat}
    return res
    

def calc_gwash_from_S_tilde(S_tilde,s2,n_sumstat,n_tilde = None,calc_mu_hat_3 = True,time_it = True):
# for now, assume imputed from 1000G so n = 503. S_tilde must have the same SNPs as the summary statistics used for s2.
    #S_tilde = sparse_matrix
    #off_diags = np.extract(1 -  np.eye(S_tilde.shape[1]), S_tilde) # can't do this because size issue
    #diagonal_mask = np.eye(S_tilde.shape[1], dtype=bool)
    time_mu2_hat_start = time.time()
    # GET VALUES FROM SPARSE MATRIX
    mk = S_tilde.shape[0] # number of SNPs in S_tilde
    #n = 503 # 1kG sample size, this is the "representative auxillary dataset"
    # IF S_Tilde_ij is 0, it means that (i,j) is not part of the window and too far away so it's not in I2.
    #non_zero_off_diags_S_tilde = get_nonzero_off_diags_from_sparse(S_tilde)
    if n_tilde is None: # if n_tilde is None, by default it is the number of observations in the sumstat since it is assumed there is access to the original design matrix X.
        n_tilde = n_sumstat
    
    #mu2_hat = 1+ ((1/m)* (((non_zero_off_diags**2).sum()) - (n_diag_vals * (1/(n-1))))) # different from the expression in the paper, why not plug in directly?
    
    #mu2_hat_ld_nonzero = ((non_zero_off_diags_S_tilde**2) - (1/(n-1))).sum() # take each non-zero element off diagonal in s_tilde and - 1/(n-1) then add
    
    if calc_mu_hat_3:
        mats = calc_I2_I3_mats(S_tilde) # contains I2, I2I2, and I3
    else:
        mats = calc_I2_mats(S_tilde,calc_non_zero = True)
    
    n_I2 = mats['I2_mat'].sum() # number of ijs such that 1 < |i-j| <= q; get I2 indicator matrix and sum elements
    
    
    #non_zero_off_diags_S_tilde.shape[0] #67516598
    #np.sum(mats['I2_mat']) #67516598
    #67600959 -A_subset.shape[0] #67516598
    
    non_zero_off_diags_S_tilde = mats['A_mat_non_zero']
    
    mu2_hat_ld_nonzero = (np.sum(non_zero_off_diags_S_tilde**2)) - (n_I2/(n_tilde-1))
    
    #test = np.sum([z**2 - 1/(502) for z in non_zero_off_diags_S_tilde])
    #1+((1/A_subset.shape[0])*test) #19.892474971660246
    
    
    #mu2_hat_zero = (-(1/(n-1)))*((m**2) - m - non_zero_off_diags_S_tilde.shape[1])
    # m*m (total elements in S_tilde) - m (elements in the diagonal) - non_zero_off_diags_S_tilde.shape[1] (number of elements that are non-zero AND off diagonal)
    # all left are off diagonal elements that are zero.
    
    #mu2_hat = 1 + ((1/m) * ( mu2_hat_ld_nonzero + mu2_hat_zero))
    mu2_hat = 1 + ((1/mk) * mu2_hat_ld_nonzero)
    # equivalent to 1 + (1/m)* (sum(S_tilde^2_ij - num_of_offdiags/(n-1))
    # if there are non_ld snps, they contribute 0s. hence, the non_zero_off_diags dont need to be adjusted.
    time_mu2_hat_elapsed = time.time() - time_mu2_hat_start
    #mu2_hat #13.852828412766003
    #3.4124495843381912
    #tr_S3 = sum(np.diagonal(np.matmul(S_tilde,np.matmul(S_tilde,S_tilde)))) # not memory efficient
    #tr_S3 = (S_tilde @ (S_tilde @ S_tilde)).diagonal().reshape(-1, 1).sum()
    #S_tilde2 = S_tilde @ S_tilde
    
    if calc_mu_hat_3:
        #S_tilde2 = dot_product_mkl(S_tilde, S_tilde, dense=False)
        S_tilde2 = S_tilde @ S_tilde # WILL TAKE A VERY LONG TIME OR RUN OUT OF MEMORY WITHOUT dot_product_mkl().
        tr_S3 = S_tilde.multiply(S_tilde2).sum()
        n_I3 = np.sum(mats['I3_mat']) # number of ijks such that 1 < |i-j| <= q and 1 < |i-k| <= q and 1 < |j-k| <= q
        mu3_hat = (1/mk)*(tr_S3 - (3 * (n_I2/(n_tilde-1))*mu2_hat) - (n_I3/((n_tilde-1)**2)))
        time_mu3_hat_elapsed = time.time() - time_mu2_hat_start # now timing for mu3 hat calculation starts from the beginning
        
    #gwash = (mk/(n_sumstat*mu2_hat)) * (s2 - 1) # old expression
    gwash = (s2-1)/((n_sumstat/mk)*mu2_hat)
    
    if calc_mu_hat_3:
        res = {'gwash':gwash,'mu3_hat':mu3_hat,'mu2_hat':mu2_hat}
    else:
        res = {'gwash':gwash,'mu2_hat':mu2_hat}
    
    if time_it:
        res['time_mu2_hat'] = time_mu2_hat_elapsed
        if calc_mu_hat_3:
            res['time_mu3_hat'] = time_mu3_hat_elapsed
    res['numerator'] = mk*(s2-1)
    res['denominator'] = n_sumstat*mu2_hat
    return res
    
    
def calc_tr_Sk(m,n,K,k):
    left = ((m-1)**k)/((n-1)**k)
    right = np.linalg.matrix_power(K, k).diagonal().sum()
    return left * right
    
    
def calc_gwash_from_X_tilde(X_tilde,u2,n_sumstat, n_tilde = None,time_it = True):
    
    time_mu2_hat_start = time.time()
    mk = X_tilde.shape[1]

    if n_tilde is None: # if n_tilde is None, by default it is the number of observations in the sumstat since it is assumed there is access to the original design matrix X.
        n_tilde = n_sumstat
    
    #K = (X_tilde @ X_tilde.T)/(mk-1) # Kinship matrix
    K = (X_tilde @ X_tilde.T)/(mk-1) # Kinship matrix
    tr_S2 = calc_tr_Sk(mk,n_tilde,K,2)
    
    mu2_hat_old = (1/mk) * tr_S2 - ((mk-1)/(n_tilde-1)) # old formula, need additional bias correction term to make it eq to bias corrected ldscores
    
    mu2_hat = ((1/mk) * tr_S2 - ((mk-1)/(n_tilde-1))) * ((n_tilde-1)/(n_tilde-2))
    
    time_mu2_hat_elapsed = time.time() - time_mu2_hat_start


    tr_S3 = calc_tr_Sk(mk,n_tilde,K,3)
    #mu3_hat = ((1/mk) * tr_S3) - (3*(((mk-1))/(n_tilde-1)) * mu2_hat) - (((mk-1)*(mk-2))/((n_tilde-1)**2))
    mu3_hat_old = ((1/mk)*tr_S3) - (3*(((mk-1))/(n_tilde-1)) * mu2_hat_old) - (((mk-1)*(mk-2))/((n_tilde-1)**2))  # old formula for mu3_hat from the paper!
    mu3_hat = (((1/mk) * tr_S3) - (3*(((mk-1))/(n_tilde-1)) * mu2_hat) - (((mk-1)*(mk-2))/((n_tilde-1)**2))) * ((n_tilde-1)/(n_tilde-2))
    time_mu3_hat_elapsed = time.time() - time_mu2_hat_start
    
    numerator = mk*(np.mean(u2)-1)
    denominator = n_sumstat*mu2_hat
    
    gwash = numerator/denominator

    res = {}
    res['gwash'] = gwash
    res['mu3_hat_old'] = mu3_hat_old
    res['mu3_hat'] = mu3_hat
    res['mu2_hat_old'] = mu2_hat_old
    res['mu2_hat'] = mu2_hat
    res['time_mu2_hat'] = time_mu2_hat_elapsed
    res['time_mu3_hat'] = time_mu3_hat_elapsed
    res['numerator'] = numerator
    res['denominator'] = denominator
    return res
    
    
    
def calc_gwash_from_ldscores(ldscores,s2,n_sumstat,n_tilde = None,calc_mu_hat_3 = True,time_it = True):

    time_mu2_hat_start = time.time()
    # GET VALUES FROM SPARSE MATRIX
    mk = ldscores.shape[0] # number of SNPs in S_tilde
    #n = 503 # 1kG sample size, this is the "representative auxillary dataset"
    # IF S_Tilde_ij is 0, it means that (i,j) is not part of the window and too far away so it's not in I2.
    #non_zero_off_diags_S_tilde = get_nonzero_off_diags_from_sparse(S_tilde)
    if n_tilde is None: # if n_tilde is None, by default it is the number of observations in the sumstat since it is assumed there is access to the original design matrix X.
        n_tilde = n_sumstat
    
    #mu2_hat = 1+ ((1/m)* (((non_zero_off_diags**2).sum()) - (n_diag_vals * (1/(n-1))))) # different from the expression in the paper, why not plug in directly?
    
    #mu2_hat_ld_nonzero = ((non_zero_off_diags_S_tilde**2) - (1/(n-1))).sum() # take each non-zero element off diagonal in s_tilde and - 1/(n-1) then add
    
    #if calc_mu_hat_3:
    #    mats = calc_I2_I3_mats(S_tilde) # contains I2, I2I2, and I3
    #else:
    #    mats = calc_I2_mats(S_tilde,calc_non_zero = True)
    
    #n_I2 = mats['I2_mat'].sum() # number of ijs such that 1 < |i-j| <= q; get I2 indicator matrix and sum elements
    
    
    #non_zero_off_diags_S_tilde.shape[0] #67516598
    #np.sum(mats['I2_mat']) #67516598
    #67600959 -A_subset.shape[0] #67516598
    
    #non_zero_off_diags_S_tilde = mats['A_mat_non_zero']
    
    #mu2_hat_ld_nonzero = (np.sum(non_zero_off_diags_S_tilde**2)) - (n_I2/(n_tilde-1))
    
    #test = np.sum([z**2 - 1/(502) for z in non_zero_off_diags_S_tilde])
    #1+((1/A_subset.shape[0])*test) #19.892474971660246
    
    
    #mu2_hat_zero = (-(1/(n-1)))*((m**2) - m - non_zero_off_diags_S_tilde.shape[1])
    # m*m (total elements in S_tilde) - m (elements in the diagonal) - non_zero_off_diags_S_tilde.shape[1] (number of elements that are non-zero AND off diagonal)
    # all left are off diagonal elements that are zero.
    
    #mu2_hat = 1 + ((1/m) * ( mu2_hat_ld_nonzero + mu2_hat_zero))
    mu2_hat = np.mean(ldscores)
    # equivalent to 1 + (1/m)* (sum(S_tilde^2_ij - num_of_offdiags/(n-1))
    # if there are non_ld snps, they contribute 0s. hence, the non_zero_off_diags dont need to be adjusted.
    time_mu2_hat_elapsed = time.time() - time_mu2_hat_start
    #mu2_hat #13.852828412766003
    #3.4124495843381912
    #tr_S3 = sum(np.diagonal(np.matmul(S_tilde,np.matmul(S_tilde,S_tilde)))) # not memory efficient
    #tr_S3 = (S_tilde @ (S_tilde @ S_tilde)).diagonal().reshape(-1, 1).sum()
    #S_tilde2 = S_tilde @ S_tilde
    
    #if calc_mu_hat_3:
    #    S_tilde2 = dot_product_mkl(S_tilde, S_tilde, dense=False)
    #    tr_S3 = S_tilde.multiply(S_tilde2).sum()
    #    n_I3 = np.sum(mats['I3_mat']) # number of ijks such that 1 < |i-j| <= q and 1 < |i-k| <= q and 1 < |j-k| <= q
    #    mu3_hat = (1/mk)*(tr_S3 - (3 * (n_I2/(n_tilde-1))*mu2_hat) - (n_I3/((n_tilde-1)**2)))
    #    time_mu3_hat_elapsed = time.time() - time_mu2_hat_start # now timing for mu3 hat calculation starts from the beginning
        
    #gwash = (mk/(n_sumstat*mu2_hat)) * (s2 - 1) # old expression
    gwash = (s2-1)/((n_sumstat/mk)*mu2_hat)
    
    if calc_mu_hat_3:
        #res = {'gwash':gwash,'mu3_hat':mu3_hat,'mu2_hat':mu2_hat} 
        res = {'gwash':gwash,'mu2_hat':mu2_hat} # for now. not sure if it's possible to calculate mu3_hat from ldscores.
    else:
        res = {'gwash':gwash,'mu2_hat':mu2_hat}
    
    if time_it:
        res['time_mu2_hat'] = time_mu2_hat_elapsed
    #    if calc_mu_hat_3:
    #        res['time_mu3_hat'] = time_mu3_hat_elapsed
    res['numerator'] = mk*(s2-1)
    res['denominator'] = n_sumstat*mu2_hat
    return res
    
def calc_true_gwash_var(n,m,rho,h2_pop):
    Sigma1 = csr_matrix(ar1_cor(m,rho))
    
    tr_S2 = (Sigma1 @ Sigma1).diagonal().sum()
    tr_S3 = (Sigma1 @ Sigma1 @ Sigma1).diagonal().sum()
    mu2 = tr_S2/m
    mu3 = tr_S3/m
    GWASH_var_true_theoretical = calc_gwash_var(n,m,mu2,mu3,h2_pop)
    GWASH_se_true_theoretical = calc_gwash_se(n,m,mu2,mu3,h2_pop)

    res = dict()
    res['rho'] = rho
    res['mu2'] = mu2
    res['mu3'] = mu3
    res['n'] = n
    res['m'] = m
    res['h2_pop'] = h2_pop
    res['gwash_var'] = GWASH_var_true_theoretical
    res['gwash_se'] = GWASH_se_true_theoretical
    return res

def calc_gwash_se(n,m,mu2_hat,mu3_hat,gwash):
    left = m/(n*mu2_hat)
    middle = (2*mu3_hat*gwash)/(mu2_hat**2)
    right = gwash**2
    return np.sqrt((2/n)*(left + middle - right)) # \phi/sqrt(n) in original GWASH paper 2019. Equation 24.

def calc_gwash_var(n,m,mu2_hat,mu3_hat,gwash):
    se = calc_gwash_se(n,m,mu2_hat,mu3_hat,gwash)
    return n * ((se)**2) # equivalent to \phi**2
    
    
### BOOTSTRAP AND SUBSAMPLING
def do_bootstrap_snps(X,y,n_tilde = None,calc_mu_hat_3 = False, time_it = False,subsample = False,subsample_n = None):
    if subsample:
        new_idx = np.random.choice(np.arange(X.shape[1]),subsample_n,replace=False)
    else:
        new_idx = np.random.choice(np.arange(X.shape[1]),X.shape[1],replace=True) #sample with replacement what rows of X to take
    
    X_sim = X[:,new_idx]
    y_sim = y
    
    #STANDARDIZATION
    X_tilde = manual_standardize_and_scale_inR(X_sim)
    n = X_tilde.shape[0]
    
    X_tilde_T_cov = np.cov(X_tilde.T) # THIS IS NOT SIGMA - THIS IS S_Tilde
    #y_tilde = (np.sqrt(n-1)* y) / np.linalg.norm(y)
    #y_tilde = (y_sim - np.mean(y_sim))/np.std(y_sim,ddof = 1)
    y_tilde = manual_standardize_and_scale_inR(y_sim)
    
    u2 = calc_u2(X_tilde,y_tilde)
    s2 = np.mean(u2)
    X_tilde_T_cov_sparse = scipy.sparse.csr_matrix(X_tilde_T_cov)
    
    if n_tilde is None:
        res = calc_gwash_from_S_tilde(X_tilde_T_cov_sparse,s2,n_sumstat = n,n_tilde = n,calc_mu_hat_3 = calc_mu_hat_3,time_it = time_it)
    else:
        res = calc_gwash_from_S_tilde(X_tilde_T_cov_sparse,s2,n_sumstat = n,n_tilde = n_tilde,calc_mu_hat_3 = calc_mu_hat_3,time_it = time_it)
    
    return res['numerator'], res['denominator']


def do_bootstrap_subj(X,y,n_tilde = None, calc_mu_hat_3 = False, time_it = False,subsample = False,subsample_n = None):
    if subsample:
        new_idx = np.random.choice(np.arange(X.shape[0]),subsample_n,replace=False)
    else:
        new_idx = np.random.choice(np.arange(X.shape[0]),X.shape[0],replace=True) #sample with replacement what rows of X to take
    
    X_sim = X[new_idx,:]
    y_sim = y[new_idx]
    
    
    
    #STANDARDIZATION
    X_tilde = manual_standardize_and_scale_inR(X_sim)
    n = X_tilde.shape[0] # If subsampling, n = n1 (which is smaller). Otherwise, n = n.
    X_tilde_T_cov = np.cov(X_tilde.T) # THIS IS NOT SIGMA - THIS IS S_Tilde
    #y_tilde = (np.sqrt(n-1)* y) / np.linalg.norm(y)
    #y_tilde = (y_sim - np.mean(y_sim))/np.std(y_sim,ddof = 1)
    y_tilde = manual_standardize_and_scale_inR(y_sim)
    
    # If X_tilde and y_tilde are computed again after reshuffling, u2 has to be computed again
    u2 = calc_u2(X_tilde,y_tilde)
    s2 = np.mean(u2)
    
    X_tilde_T_cov_sparse = scipy.sparse.csr_matrix(X_tilde_T_cov)
    res = calc_gwash_from_S_tilde(X_tilde_T_cov_sparse,s2,n_sumstat = n,n_tilde = n_tilde,calc_mu_hat_3 = calc_mu_hat_3,time_it = time_it)
    return res['numerator'], res['denominator']

def bootstrap_wrapper(X,y,B,n_tilde = None,calc_mu_hat_3 = False, time_it = False,mode = 'both',subsample = False, subsample_n = None):
    gwash_num_boot_SNP = []
    gwash_denom_boot_SNP = []
    gwash_num_boot_SUBJ = []
    gwash_denom_boot_SUBJ = []
    gwash_boot_snp = []
    gwash_boot_subj = []
    
    valid_modes = ['SNP','SUBJ','both']
    if mode not in valid_modes:
        raise('invalid mode, valid modes: [SNP,SUBJ,both]')
        
        
    for b in range(B):
        if mode == 'SNP':
            num_snp,denom_snp = do_bootstrap_snps(X,y,n_tilde = n_tilde,calc_mu_hat_3 = False, time_it = False,subsample = subsample, subsample_n = subsample_n)
            
            gwash_num_boot_SNP.append(num_snp)
            gwash_denom_boot_SNP.append(denom_snp)
            gwash_snp = num_snp/denom_snp
            gwash_boot_snp.append(gwash_snp)
        elif mode == 'SUBJ':
            num_subj,denom_subj = do_bootstrap_subj(X,y,n_tilde = n_tilde,calc_mu_hat_3 = False, time_it = False,subsample = subsample, subsample_n = subsample_n)
            gwash_num_boot_SUBJ.append(num_subj)
            gwash_denom_boot_SUBJ.append(denom_subj)
            gwash_subj = num_subj/denom_subj
            gwash_boot_subj.append(gwash_subj)
        else:
            num_snp,denom_snp = do_bootstrap_snps(X,y,n_tilde = n_tilde,calc_mu_hat_3 = False, time_it = False,subsample = subsample, subsample_n = subsample_n)
            num_subj,denom_subj = do_bootstrap_subj(X,y,n_tilde = n_tilde,calc_mu_hat_3 = False, time_it = False,subsample = subsample, subsample_n = subsample_n)
            
            gwash_num_boot_SNP.append(num_snp)
            gwash_denom_boot_SNP.append(denom_snp)
            gwash_snp = num_snp/denom_snp
            gwash_boot_snp.append(gwash_snp)
            
            gwash_num_boot_SUBJ.append(num_subj)
            gwash_denom_boot_SUBJ.append(denom_subj)
            gwash_subj = num_subj/denom_subj
            gwash_boot_subj.append(gwash_subj)
            
    res = dict()
    def make_res_boot(num_boot,denom_boot,gwash_boots):
        res_boot = dict()
        res_boot = {'num_mean':np.mean(num_boot),
                     'num_std': np.std(num_boot,ddof = 1),
                    'denom_mean': np.mean(denom_boot),
                    'denom_std': np.std(denom_boot,ddof = 1),
                   'gwash_mean':np.mean(gwash_boots),
                   'gwash_std': np.std(gwash_boots,ddof = 1)}
        return res_boot
    
    if mode == 'SNP':
        res['SNP'] = make_res_boot(gwash_num_boot_SNP,gwash_denom_boot_SNP,gwash_boot_snp)
    elif mode == 'SUBJ':
        res['SUBJ'] = make_res_boot(gwash_num_boot_SUBJ,gwash_denom_boot_SUBJ,gwash_boot_subj)
    else:
        res['SNP'] = make_res_boot(gwash_num_boot_SNP,gwash_denom_boot_SNP,gwash_boot_snp)
        res['SUBJ'] = make_res_boot(gwash_num_boot_SUBJ,gwash_denom_boot_SUBJ,gwash_boot_subj)
    
    return res

    
    
### JACKKNIFE
def make_blocks(idx,num_blocks = 200):
    # input: vector of indicies
    # output: list of num_blocks vectors containing idx
    return np.array_split(idx,num_blocks)

def calc_gwash_block(block_idx,u2j,S_tilde,n_sumstat,n_tilde = None):
    #Computes GWASH with all SNPs EXCEPT those in block_idx
    S_tilde_subset = S_tilde[:,block_idx][block_idx,:]
    mk = S_tilde_subset.shape[0]
    
    s2 = np.mean(u2j[block_idx])

    res = calc_gwash_from_S_tilde(S_tilde_subset,s2,n_sumstat = n_sumstat,n_tilde = n_tilde,calc_mu_hat_3 = False,time_it = False)

    return res['gwash']

def prep_gwash_jackknife(S_tilde,n_sumstat,u2j,idx,n_tilde = None,num_blocks = 200):
    blocks = make_blocks(idx,num_blocks = num_blocks)
    gwashes_blocks = []
    for b in range(len(blocks)):
        arrays_to_combine = [arr for j, arr in enumerate(blocks) if j != b]
        idx_minus_b = np.concatenate(arrays_to_combine)
        gwashes_blocks.append(calc_gwash_block(idx_minus_b,u2j,S_tilde,n_sumstat,n_tilde = n_tilde)) #if n_tilde is None, then n_tilde = n_sumstat = nrow of design matrix
    m_blocks = np.array([z.shape[0] for z in blocks])
    #gwash_est_j = (num_blocks * gwash_est) - np.multiply((m-num_blocks),(np.array(gwashes_blocks)/m)).sum()
    #taus = (weights*gwash_est) - np.multiply((weights -1),gwashes_blocks)
    #var = np.mean((taus - gwash_est_j)**2/(weights-1))
    return num_blocks,m_blocks,np.array(gwashes_blocks)

def calc_gwash_jk_SE(gwash_est,m_blocks,n_blocks,gwashes_blocks):
    m = m_blocks.sum() #1020367
    weights = m/m_blocks
    gwash_est_j = (n_blocks * gwash_est) - ((m-m_blocks)*(np.array(gwashes_blocks)/m)).sum() #921.3705596048318
    taus = (weights*gwash_est) - np.multiply((weights -1),gwashes_blocks)
    var = np.mean((taus - gwash_est_j)**2/(weights-1))
    
    return np.sqrt(var)

    
def calc_gwash_block_subj(block_idx,X,y,n_tilde = None):
    
    X_jk = X[block_idx,:]
    y_jk = y[block_idx]
    #STANDARDIZATION
    X_tilde = manual_standardize_and_scale_inR(X_jk)
    n = X_tilde.shape[0] # number of subjects
    X_tilde_T_cov = np.cov(X_tilde.T) # THIS IS NOT SIGMA - THIS IS S_Tilde
    #y_tilde = (y_jk - np.mean(y_jk))/np.std(y_jk,ddof = 1)
    y_tilde = manual_standardize_and_scale_inR(y_jk)
    u2 = calc_u2(X_tilde,y_tilde)
    #def calc_u2(X,y):
        #u = np.matmul(X.T,y)/np.sqrt(X.shape[0]-1)
    #return u**2
    s2 = np.mean(u2)

    X_tilde_T_cov_sparse = scipy.sparse.csr_matrix(X_tilde_T_cov)
    res = calc_gwash_from_S_tilde(X_tilde_T_cov_sparse,s2,n,n_tilde = n_tilde,calc_mu_hat_3 = False,time_it = False)
    #return res['gwash']
    return res['numerator'], res['denominator']


def prep_gwash_jackknife_subj(X,y,n_sumstat,idx,n_tilde = None,num_blocks = 200):
    blocks = make_blocks(idx,num_blocks = num_blocks)
    gwashes_blocks = []
    for b in range(len(blocks)):
        arrays_to_combine = [arr for j, arr in enumerate(blocks) if j != b]
        idx_minus_b = np.concatenate(arrays_to_combine)
        num,denom = calc_gwash_block_subj(idx_minus_b,X,y,n_tilde = n_tilde)
        gwashes_blocks.append(num/denom)
    n_blocks = np.array([z.shape[0] for z in blocks])
    return num_blocks,n_blocks,np.array(gwashes_blocks)


def calc_gwash_subj_jk_SE(gwash_est,n_blocks,gwashes_blocks):
    n = n_blocks.sum() #1020367
    weights = n/n_blocks
    gwash_est_j = (n_blocks.shape[0] * gwash_est) - ((n-n_blocks)*(np.array(gwashes_blocks)/n)).sum() #921.3705596048318
    taus = (weights*gwash_est) - np.multiply((weights -1),gwashes_blocks)
    var = np.mean((taus - gwash_est_j)**2/(weights-1))
    return np.sqrt(var)
    
    
    
def make_ldscores_from_S_Tilde(S_tilde,M,N,bias_correct = True):
    S_tilde_rjk_squared = S_tilde**2
    ld_scores_no_adj = S_tilde_rjk_squared.sum(axis = 0).reshape(-1,1)
    if bias_correct:
        ld_scores = ld_scores_no_adj - ((M-ld_scores_no_adj)/np.mean(N))
    else:
        ld_scores = ld_scores_no_adj
    return ld_scores

    
    
### REGENERATED IN EVERY REPLICATE
def do_gwash_ldsc_simulations(n,m,h2_pop,num_sims,num_B,M,N_per_SNP,covar_test,b,h2_true,intercept = None):
    gwashes = []
    gwash_vars = []

    gwashes_num = []
    gwashes_denom = []

    jk_snp_SEs = []
    jk_subj_SEs = []

    mu2_hats = []
    mu3_hats = []
    mu2_hat_times = []
    mu3_hat_times = []

    ldsc_h2s = []
    ldsc_intercepts = []
    ldsc_ses = []

    ldsc_h2s_crude = []

    for i in tqdm(range(num_sims)):
        # THIS IS THE N X M POPULATION MATRIX
        X = stats.multivariate_normal.rvs(mean = np.repeat(0,m),cov = covar_test, size = n) # this is the genotype matrix (n x m)

        # RESPONSE (NOT KNOWN)
        coefficient_SNPs = b.reshape(m,1) # take the betas and make a mx1 matrix
        tau2 = np.matmul(coefficient_SNPs.T,np.matmul(covar_test,coefficient_SNPs)) # beta.T %*% Sigma %*% beta
        sigma2 = float(tau2 * ((1-h2_pop)/h2_pop))
        Geno_effect = np.matmul(X,coefficient_SNPs)
        eps = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=n).reshape(n,1)
        y = Geno_effect + eps


        ### STANDARDIZATION
        X_tilde = manual_standardize_and_scale_inR(X)
        X_tilde_T_cov = np.cov(X_tilde.T) # THIS IS NOT SIGMA - THIS IS S_Tilde

        #y_tilde = (np.sqrt(n-1)* y) / np.linalg.norm(y)
        #y_tilde = (y - np.mean(y))/np.std(y,ddof = 1)
        y_tilde = manual_standardize_and_scale_inR(y) # equivalent to above

        #ldscores = calc_ldscore_from_Xcorr(X_tilde_T_cov)
        u2 = calc_u2(X_tilde,y_tilde)
        s2 = np.mean(u2)


        ## LDSC
        # ldscores are the rowsum of S_tilde**2 (element-wise squared)
        #ldscores_from_S_tilde = make_ldscores_from_S_Tilde(X_tilde_T_cov).reshape(-1,1)
        ldscores_from_S_tilde = make_ldscores_from_S_Tilde(X_tilde_T_cov,M,N_per_SNP,bias_correct = True)

        #ldscores_from_S_tilde_bias_corrected = (ldscores_from_S_tilde - (M-ldscores_from_S_tilde)/np.mean(N_per_SNP))
        #ldscores_from_S_tilde_bias_corrected = (ldscores_from_S_tilde - (M)/np.mean(N_per_SNP))

        #l_bar = (1/m) * np.sum(((n/m)*ldscores_from_S_tilde_bias_corrected) - 1)
        l_bar = (1/m) * np.sum(((n/m)*ldscores_from_S_tilde) - 1)

        #u2 are eq to chi-squared since t statistics are equivalent to z statistics at higher degrees

        ldsc_reg_weights = ldscores_from_S_tilde # for now, x = w. This may change in different scenarios.

        ldsc_h2, ldsc_se,ldsc_intercept = do_ldsc_regression(u2,ldscores_from_S_tilde,M,N_per_SNP,ldsc_reg_weights,intercept = intercept) # returns h2_est and SE according to ldsc

        ldsc_h2s.append(ldsc_h2)
        ldsc_ses.append(ldsc_se)
        ldsc_intercepts.append(ldsc_intercept)


        ldsc_h2s_crude.append(ldsc_h2_crude(u2,ldscores_from_S_tilde,M,N_per_SNP,1))

        ## GWASH
        X_tilde_T_cov_sparse = scipy.sparse.csr_matrix(X_tilde_T_cov)
        mats = calc_I2_mats(X_tilde_T_cov_sparse)
        n_I2 = mats['I2_mat'].nnz # number of ijs such that 1 < |i-j| <= q; get I2 indicator matrix and sum elements
        non_zero_off_diags_S_tilde = mats['A_mat_non_zero']
        res = calc_gwash_from_S_tilde(X_tilde_T_cov_sparse,s2,n_sumstat = n,n_tilde = None,calc_mu_hat_3 = True,time_it = True)
        gwashes.append(res['gwash'])
        gwashes_num.append(res['numerator'])
        gwashes_denom.append(res['denominator'])

        mu2_hats.append(res['mu2_hat'])
        mu3_hats.append(res['mu3_hat'])

        mu2_hat_times.append(res['time_mu2_hat'])
        mu3_hat_times.append(res['time_mu3_hat'])
        #calculate GWASH Variance from mu3_hat
        gwash_vars.append(calc_gwash_var(n,m,res['mu2_hat'],res['mu3_hat'],res['gwash']))


        '''
        #calculate GWASH Variance from jackknife on SNPs

        jk_snp_time_start = time.time()
        num_blocks,m_blocks, gwashes_blocks = prep_gwash_jackknife(X_tilde_T_cov_sparse,n,u2,[z for z in range(m)],n_tilde = None,num_blocks = num_blocks)
        jk_snp_SEs.append(calc_gwash_jk_SE(res['gwash'],m_blocks,num_blocks,gwashes_blocks))
        jk_snp_time_elapsed = time.time() - jk_snp_time_start
        jk_snp_times.append(jk_snp_time_elapsed)

        #calculate GWASH Variance from jackknife on subjects

        jk_subj_time_start = time.time()
        num_blocks,n_blocks, gwashes_blocks = prep_gwash_jackknife_subj(X,y,n,[z for z in range(n)],n_tilde = None,num_blocks = num_blocks)
        jk_subj_SEs.append(calc_gwash_subj_jk_SE(res['gwash'],n_blocks,gwashes_blocks))
        jk_subj_time_elapsed = time.time() - jk_subj_time_start
        jk_subj_times.append(jk_subj_time_elapsed)
        '''


    # POST PROCESSING
    gwash_ses = [np.sqrt(z) for z in gwash_vars]
    sim_res = pd.DataFrame([gwashes,gwash_ses,ldsc_h2s,ldsc_ses,ldsc_intercepts,ldsc_h2s_crude]).T
    sim_res.columns = ['gwash_h2','gwash_se','ldsc_h2','ldsc_se','ldsc_intercept','ldsc_h2_crude']
    sim_res['sim_idx'] = sim_res.index

    # ADJUST LDSC h2 IF IT GETS OUT OF HAND
    ldsc_h2_adj = []
    for z in sim_res['ldsc_h2']:
        if z > 1:
            z = 1
        elif z < 0:
            z = 0
        else:
            z = z
        ldsc_h2_adj.append(z)
    sim_res['ldsc_h2_adjusted'] = ldsc_h2_adj
    return sim_res
    
    
def do_sims_alt_rho(n,m,h2_pop,num_sims,num_B,M,N_per_SNP,b,h2_true):
    h2_dict = dict()
    rhos = [0,0.2,0.4,0.6,0.8]
    for rho in rhos:
        Sigma_tilde = ar1_cor(m,rho)
        covar_test = np.matmul(np.diag(np.full(m,range(1,m+1)))**(1/2),np.matmul(Sigma_tilde,np.diag(np.full(m,range(1,m+1)))**(1/2)))
        h2_dict[str(rho)] = do_gwash_ldsc_simulations(n,m,h2_pop,num_sims,num_B,M,N_per_SNP,covar_test,b,h2_true)
    return h2_dict