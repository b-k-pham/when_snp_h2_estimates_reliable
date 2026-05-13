# DUMP THESE FUNCTIONS INTO A PYTHON FILE TO IMPORT
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
from tqdm import tqdm

import sys
if sys.platform == 'win32':
    import os
    os.environ["R_HOME"] = f"{os.environ['CONDA_PREFIX']}\\Lib\\R"

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm


def calc_mu2_hat(X,center = True,scale = True):
    # input = n x m matrix where rows are observations and columns are SNPs
    c_s_params = [center,scale]
    X_new = X
    if c_s_params[0] == True:
        X_new = (X - np.mean(X,axis = 0))
    if c_s_params[1] == True:
        X_new = X_new/np.std(X_new,axis = 0,ddof=1)
    X_new = X_new[:,~np.isnan(X_new).any(axis=0)] # grab SNPs with no NAs
    n,m = X_new.shape
    S_tilde = (1/(n-1)) * np.matmul(X_new.T,X_new)
    off_diags = np.extract(1 -  np.eye(S_tilde.shape[1]), S_tilde)
    mu2 = 1+ ((1/m)* (sum(off_diags**2) - (len(off_diags) * (1/(n-1)))))
    return (mu2)


def calc_mu3_hat(X,center = True,scale = True):
    # input = n x m matrix where rows are observations and columns are SNPs
    c_s_params = [center,scale]
    X_new = X
    if c_s_params[0] == True:
        X_new = (X - np.mean(X,axis = 0))
    if c_s_params[1] == True:
        X_new = X_new/np.std(X_new,axis = 0,ddof=1)
    X_new = X_new[:,~np.isnan(X_new).any(axis=0)] # grab SNPs with no NAs
    n,m = X_new.shape
    S_tilde = (1/(n-1)) * np.matmul(X_new.T,X_new)
    tr_S3 = sum(np.diagonal(np.matmul(S_tilde,np.matmul(S_tilde,S_tilde))))
    mu2_hat = calc_mu2_hat(X,center = center,scale = scale)
    mu3_hat = (1/m)*(tr_S3 - ((3/(n-1)) * (m*(m-1)) * mu2_hat) - ((m*(m-1)*(m-2)) * (1/(n-1)**2)))
    return(mu3_hat)
    
    
def calc_s2(X,y,center = True,scale = True):
    c_s_params = [center,scale]
    X_new = X
    y_new = y
    if c_s_params[0] == True:
        X_new = (X - np.mean(X,axis = 0))
        y_new = (y - np.mean(y,axis = 0))
    if c_s_params[1] == True:
        X_new = X_new/np.std(X_new,axis = 0,ddof=1)
        y_new = y_new/np.std(y_new,axis = 0,ddof = 1)
    X_new = X_new[:,~np.isnan(X_new).any(axis=0)] # grab SNPs with no NAs
    n,m = X_new.shape
    s2 = []
    for j in range(m):
        xj = X_new[:,j].reshape(1,n)
        s2.append((np.matmul(xj,y_new).item()* (1/np.sqrt(n-1)))**2)
    s2 = np.mean(s2)
    return(s2)
    
    
def calc_GWASH(X,y,center = True,scale = True):
    s2 = calc_s2(X,y,center=center,scale = scale)
    X_new = X[:,~np.isnan(X).any(axis=0)] # grab SNPs with no NAs
    n,m = X_new.shape
    mu2_hat = calc_mu2_hat(X,center = center,scale = scale)
    mu3_hat = calc_mu3_hat(X,center = center,scale = scale)
    gwash = (m/(n*mu2_hat))* (s2-1)
    res = {'h2_gwash': gwash,'mu2_hat': mu2_hat,'mu3_hat': mu3_hat}
    return(res)
    


def FVE_residualize_indetail_GWASH(X,y,SNP_groups,center = True,scale = True):
    group_keys = [z for z in SNP_groups.keys()] # get keys designating each group
    FVE_cumm = []
    processed = []
    for z in range(len(group_keys)):
        if len(processed) == 0:
            Xz = X[:,SNP_groups[group_keys[z]]]
            FVE_cumm.append(calc_GWASH(Xz,y,center = center,scale = scale)['h2_gwash'])
        else:
            Xz = X[:,list(set(SNP_groups[group_keys[z]] + processed))] # This is FVE(X1,X2)
            #Xz = X[:,SNP_groups[group_keys[z]] + processed]
            FVE_cumm.append(calc_GWASH(Xz,y,center = center,scale = scale)['h2_gwash'])
        processed = processed + SNP_groups[group_keys[z]]
    FVE_cond = [FVE_cumm[0]]
    for z in range(1,len(FVE_cumm)):
        FVE_cond.append(FVE_cumm[z] - FVE_cumm[z-1])
    FVE_cond = pd.DataFrame([FVE_cond])
    FVE_cumm = pd.DataFrame([FVE_cumm])
    return((FVE_cumm,FVE_cond))
    
    
def manual_standardize_and_scale_inR(X):
    X_centered = X - np.mean(X,axis = 0)
    return X_centered/np.std(X_centered,ddof = 1,axis = 0)