import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

'''
MOST OF THESE FUNCTIONS ARE COPIED AS IS FROM THE ORIGINAL LDSC PYTHON CODE (https://github.com/bulik/ldsc/tree/master). A lot of these functions are in ldsc/ldscore.

ONLY THE FUNCTIONS NECESSARY TO RUN SINGLE LDSC Regression ARE SHOWN HERE. I just took the functions out of their respective class for readability of the main function.

See ldsc_implementation_validation.ipynb for validation across 89 summary statistics that these functions work correctly (equivalent to running ldsc out of the box).
'''


'''

These functions are used to do Iterativey re-weighted least squares from ldsc/ldscore/irwls.py.

'''


def weight_by_w(x,w):
    if np.any(w <= 0):
        raise ValueError('Weights must be > 0')
    n,p = x.shape
    if w.shape != (n, 1):
        raise ValueError('w has shape {S}. w must have shape ({n},1).'.format(S=w.shape,n=n))
    w = w / float(np.sum(w))
    x_new = np.multiply(x, w)
    return x_new

def wls(x,y,w):
    x_weighted = weight_by_w(x, w)
    y_weighted = weight_by_w(y, w)
    coef = np.linalg.lstsq(x_weighted, y_weighted)
    return coef


def weights(ld, w_ld, N, M, hsq, intercept=None, ii=None):
        '''
        Regression weights.

        Parameters
        ----------
        ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned).
        w_ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included
            in the regression.
        N :  np.matrix of ints > 0 with shape (n_snp, 1)
            Number of individuals sampled for each SNP.
        M : float > 0
            Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
            the regression).
        hsq : float in [0,1]
            Heritability estimate.

        Returns
        -------
        w : np.matrix with shape (n_snp, 1)
            Regression weights. Approx equal to reciprocal of conditional variance function.

        '''
        M = float(M)
        if intercept is None:
            intercept = 1

        hsq = max(hsq, 0.0)
        hsq = min(hsq, 1.0)
        ld = np.fmax(ld, 1.0)
        w_ld = np.fmax(w_ld, 1.0)
        c = hsq * N / M 
        het_w = 1.0 / (2 * np.square(intercept + np.multiply(c.reshape(-1,1), ld)))
        # variance of chi-squared increases with ld score. SNPs with high ld should be weighed less.
        oc_w = 1.0 / w_ld #w_ld is the regression SNPs.
        w = np.multiply(het_w, oc_w)
        return w
    
    
def update(wls_out,x,w_ld,N,M,Nbar,intercept = None,ii = None):
    hsq = M * wls_out[0][0] / Nbar
    if intercept is None:
        intercept = max(wls_out[0][1])
    else:
        if w_ld.shape[1] > 1:
            raise ValueError('Design matrix has intercept column for constrained intercept regression!')
    ld = x[:, 0].reshape(w_ld.shape)  # remove intercept
    #print('ld')
    #print(ld)
    #print('w_ld')
    #print(w_ld)
    #print('N')
    #print(N)
    #print('M')
    #print(M)
    #print('hsq')
    #print(hsq)
    #print('intercept')
    #print(intercept)
    #print('ii')
    #print(ii)
    w = weights(ld, w_ld, N, M, hsq, intercept, ii)
    return w
    

#def my_irwls(x,y,w,update_func,separators = None,num_blocks = None):
#    w = np.sqrt(w)
#    for i in range(2):
#        wls_out = wls(x,y,w)
#        new_w = np.sqrt(update_func(wls_out))
#        if new_w.shape != w.shape:
#            print ('IRWLS update:', new_w.shape, w.shape)
#            raise('New weights must have same shape.')
#        else:
#            w = new_w
#    x = weight_by_w(x,w)
#    y = weight_by_w(y,w)
#    
#    jk_fast = jackknife_fast(x,y,num_blocks = num_blocks,separators = separators)
#    return jk_fast
    
def get_w_and_weighted_x_y(x,y,w,update_func,num_weight_iteration = 2,separators = None,num_blocks = None):
    w = np.sqrt(w)
    for i in range(num_weight_iteration):
        wls_out = wls(x,y,w)
        new_w = np.sqrt(update_func(wls_out))
        if new_w.shape != w.shape:
            print ('IRWLS update:', new_w.shape, w.shape)
            raise('New weights must have same shape.')
        else:
            w = new_w
    x = weight_by_w(x,w)
    y = weight_by_w(y,w)
    
    #jk_fast = jackknife_fast(x,y,num_blocks = num_blocks,separators = separators)
    return w,x,y




'''
jackknife functions from ldscore/jackknife.py
'''

def get_separators(N, n_blocks):
    '''Define evenly-spaced block boundaries.'''
    return np.floor(np.linspace(0, N, n_blocks + 1)).astype(int)

def my_block_values(X,y,num_blocks = None, separators = None):
    # First, Look at specific blocks of X and y. Then for each block, calculate XtX and Xty.
    n,p = X.shape
    if separators is None and num_blocks is not None:
        seps = get_separators(n, num_blocks)
    elif separators is not None and num_blocks is None:
        seps = separators
    else:
        raise('num_blocks must be not None or separators must be not None. Check params.')
    n_blocks = len(seps) - 1
    XtX_block_values = np.zeros((n_blocks, p, p)) # make n_blocks of p x p matrices
    Xty_block_values = np.zeros((n_blocks, p)) # make n_blocks x p
    i = 0
    #X[seps[i]:seps[i + 1], ...].T is 2 x 5595
    #y[seps[i]:seps[i + 1], ...] is 5595 x 1
    #Xty is now 1 x 2 per block
    #XtX is 2 x 2 per block
    #test = np.dot(X[seps[i]:seps[i + 1], ...].T, X[seps[i]:seps[i + 1], ...])
    #test = np.dot(X[seps[i]:seps[i + 1], ...].T, y[seps[i]:seps[i + 1], ...]).reshape((1,p))
    for i in range(n_blocks):
        XtX_block_values[i, ... ] = np.dot(X[seps[i]:seps[i + 1], ...].T, X[seps[i]:seps[i + 1], ...])
        Xty_block_values[i, ...] = np.dot(X[seps[i]:seps[i + 1], ...].T, y[seps[i]:seps[i + 1], ...]).reshape((1,p))
    return (XtX_block_values,Xty_block_values,seps)

def block_sanity_check(X,y):
    '''
    block_sanity_check(new_x,new_y)
    (array([[3.05531423e-04, 1.40862262e-05],
            [1.40862262e-05, 1.05403941e-06]]),
     array([[1.59412561e-05],
            [1.15004191e-06]]))

    xtx_b,xty_b = my_block_values(new_x,new_y,200)
    array([[3.05531423e-04, 1.40862262e-05],
           [1.40862262e-05, 1.05403941e-06]])
    array([1.59412561e-05, 1.15004191e-06])
    '''
    XtX = np.dot(X.T,X)
    Xty = np.dot(X.T,y)
    return (XtX,Xty)

def block_vals_to_est(Xty_block_values,XtX_block_values):
    
    p = XtX_block_values[0].shape[0]
    
    Xty = Xty_block_values.sum(axis = 0)
    XtX = XtX_block_values.sum(axis = 0)
    return np.linalg.solve(XtX, Xty).reshape((1,p))


def block_values_to_delete_values(Xty_block_values,XtX_block_values):
    p = XtX_block_values[0].shape[0]
    n_blocks = XtX_block_values.shape[0]
    delete_values = np.zeros((n_blocks, p))
    Xty_tot = np.sum(Xty_block_values, axis=0)
    XtX_tot = np.sum(XtX_block_values, axis=0)
    for j in range(n_blocks):
        delete_Xty = Xty_tot - Xty_block_values[j]
        delete_XtX = XtX_tot - XtX_block_values[j]
        delete_values[j,...] = np.linalg.solve(delete_XtX, delete_Xty).reshape((1, p))
    return delete_values

def delete_values_to_pseudovalues(delete_values,est):
    n_blocks,p = delete_values.shape
    pseudovalues = n_blocks*est - (n_blocks-1) * delete_values
    return pseudovalues

def jknife(pseudovalues):
    n_blocks = pseudovalues.shape[0]
    jknife_cov = np.atleast_2d(np.cov(pseudovalues.T, ddof=1) / n_blocks)
    jknife_var = np.atleast_2d(np.diag(jknife_cov))
    jknife_se = np.atleast_2d(np.sqrt(jknife_var))
    jknife_est = np.atleast_2d(np.mean(pseudovalues, axis=0))
    return (jknife_est, jknife_var, jknife_se, jknife_cov)

def jackknife_fast(x,y,num_blocks = None,separators = None):
    xtx_b,xty_b,seps = my_block_values(x,y,num_blocks = num_blocks, separators = separators)
    ests = block_vals_to_est(xty_b,xtx_b)
    delete_values = block_values_to_delete_values(xty_b,xtx_b)
    pseudovalues = delete_values_to_pseudovalues(delete_values,ests)
    #check = ests[0]-jknife(pseudovalues)[0]
    #print(check)
    #print(ests)
    #print(jknife(pseudovalues)[0])
    return (jknife(pseudovalues),seps,delete_values)


### RETRIEVE ESTIMATES FROM JK

def get_intercept(irwls_res):
    est_res = irwls_res[0][0]
    se_res = irwls_res[0][2]
    intercept = est_res[0,1]
    intercept_se = se_res[0,1]
    return intercept,intercept_se

'''
These functions are from ldscore/regressions.py and are used to do the regression after partitioning.
'''

def append_intercept(x):
    '''
    Appends an intercept term to the design matrix for a linear regression.

    Parameters
    ----------
    x : np.matrix with shape (n_row, n_col)
        Design matrix. Columns are predictors; rows are observations.

    Returns
    -------
    x_new : np.matrix with shape (n_row, n_col+1)
        Design matrix with intercept term appended.

    '''
    n_row = x.shape[0]
    intercept = np.ones((n_row, 1))
    x_new = np.concatenate((x, intercept), axis=1)
    return x_new


def remove_intercept(x):
    '''Removes the last column.'''
    n_col = x.shape[1]
    return x[:, 0:n_col - 1]


def aggregate(y,x,N,M,intercept):
    if intercept is None:
        intercept = 1
    num = M*(np.mean(y) - intercept)
    denom = np.mean(np.multiply(x.reshape(-1,1),N.reshape(-1,1)))
    return num/denom

def update_separators(s, ii):
    '''s are separators with ii masked. Returns unmasked separators.'''
    maplist = np.arange(len(ii))[np.squeeze(ii)]
    mask_to_unmask = lambda i: maplist[i]
    t = np.apply_along_axis(mask_to_unmask, 0, s[1:-1])
    t = np.hstack(((0), t, (len(ii))))
    return t


def combine_twostep_jknife(irwls_res1,irwls_res2,c,Nbar=1):
    step1_jknife_delete_values = irwls_res1[2]
    step2_jknife_delete_values = irwls_res2[2]
    n_blocks, n_annot = step1_jknife_delete_values.shape
    n_annot -= 1
    if n_annot > 2:
        raise ValueError('twostep not yet implemented for partitioned LD Score.')
    step1_int,step1_se = get_intercept(irwls_res1)
    step2_jknife_est = irwls_res2[0][0]
    est = np.hstack(
            (step2_jknife_est.reshape((1, 1)), np.array(step1_int).reshape((1, 1))))
    
    
    delete_values = np.zeros((n_blocks, n_annot + 1))
    
    delete_values[:, n_annot] = step1_jknife_delete_values[:, n_annot]
    delete_values[:, 0:n_annot] = step2_jknife_delete_values -\
            c * (step1_jknife_delete_values[:, n_annot] -
                 step1_int).reshape((n_blocks, n_annot))  # check this
    pseudovalues = delete_values_to_pseudovalues(delete_values, est)
    jknife_est, jknife_var, jknife_se, jknife_cov = jknife(
            pseudovalues)
    
    
    return (est, jknife_se, jknife_est, jknife_var, jknife_cov, delete_values)
    
    

    
# ldsc report result functions


def format_single_jackknife(jknife,intercept,step1_only = False):
    '''formatter so the single jackknife result can be used in ldsc report result functions'''
    jknife_est, jknife_var, jknife_se, jknife_cov = jknife[0]
    delete_values = jknife[2]
    if step1_only:
        est = jknife_est
    else:
        est = np.hstack([jknife_est.reshape(-1,1),np.array(intercept).reshape(-1,1)])
    return (est, jknife_se, jknife_est, jknife_var, jknife_cov, delete_values)


'''
These functions come from regressions.py
'''

def ldsc_coef(jknife, Nbar, n_annot = 1):
    '''Get coefficient estimates + cov from the jackknife.'''
    n_annot = n_annot
    coef = jknife[0][0, 0:n_annot] / Nbar
    coef_cov = jknife[4][0:n_annot, 0:n_annot] / Nbar ** 2
    coef_se = np.sqrt(np.diag(coef_cov))
    return coef, coef_cov, coef_se

def ldsc_cat(jknife, M, Nbar, coef, coef_cov):
    '''Convert coefficients to per-category h2 or gencov.'''
    cat = np.multiply(M, coef)
    cat_cov = np.multiply(np.dot(M.T, M), coef_cov)
    cat_se = np.sqrt(np.diag(cat_cov))
    return cat, cat_cov, cat_se

def ldsc_tot(cat, cat_cov):
    '''Convert per-category h2 to total h2 or gencov.'''
    tot = np.sum(cat)
    tot_cov = np.sum(cat_cov)
    tot_se = np.sqrt(tot_cov)
    return tot, tot_cov, tot_se

    
'''
Main wrapper function of LDSC Regression.
'''

def do_ldsc_regression(y,x,M,N,w,intercept = None,num_weight_iteration = 2,twostep = None,return_step1_only = False):
    
    x_orig = x
    M_tot = np.sum(M)
    Nbar = np.mean(N) # mean sample size per SNP
    n_snp, n_annot = x.shape
    x_tot = np.sum(x, axis=1).reshape((n_snp, 1))
    # 0th step, crude estimate of heritability.
    tot_agg = aggregate(y,x_tot,N,M_tot,intercept)
    Nbar = np.mean(N)
    x = np.multiply(N.reshape(-1,1), x) / Nbar
    
    constrain_intercept = intercept is not None
    ### if not self.constrain_intercept: # constrain_intercept is FALSE by default
    if not constrain_intercept:
        x = append_intercept(x)
        x_tot = append_intercept(x_tot)
        yp = y
    else:
        yp = y - intercept
        intercept_se = 'NA'
    ###

    '''
    By default:
        if n_annot == 1:
            if args.two_step is None and args.intercept_h2 is None:
                args.two_step = 30

    step1_ii = None
            if twostep is not None:
                step1_ii = y < twostep
    '''
    # TRUE IF n_annot == 1 which for now is always True
    initial_w = weights(x_orig, w, N.reshape(-1,1), M_tot, tot_agg, intercept=intercept, ii=None)
    if twostep is None and intercept is None:
        twostep = 30
    step1_ii = None
    if twostep is not None:
        step1_ii = y < twostep # check if there's any SNPs that have chi2 < twostep.
    
    
    if step1_ii is not None and constrain_intercept:
        raise ValueError('twostep is not compatible with constrain_intercept.')
    elif step1_ii is not None:
        n1 = np.sum(step1_ii) # 1119173
        twostep_filtered = n_snp - n1
        x1 = x[np.squeeze(step1_ii), :]

        yp1, w1, N1, initial_w1 = map(lambda a: a[step1_ii].reshape((n1, 1)), (yp, w, N, initial_w))


        ### WHY TWO-STEP JACKKNIFE:
        # EXPLAINED HERE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4797329/
        '''
        SNPs with very large effect sizes can cause large LD score regression standard errors
        with unconstrained intercept.
        Working solution is to remove SNPs with chi-squared > 80.

        If you want to estimate the LD Score regression intercept, this is fine BUT
        this biases h2 and rho_g towards 0.

        The second step is needed which is estimating h2 using all SNPs and
        constraining the intercept with the estimated intercept found in the first step.


        For the two-step estimator, if we were to estimate the intercept in the first step, 
        then obtain a jackknife standard error for the second step treating the intercept as fixed,
        the standard error would be biased downwards, because it would not take into account
        the uncertainty in the intercept. Instead, we jackknife both steps of the procedure,
        which appropriately accounts for uncertainty in the intercept and yields a valid standard error.

        '''
        ### DO STEP 1
        #irwls_res = my_irwls(x1,yp1,initial_w1,w1,N1,M_tot,Nbar,ii = step1_ii,num_blocks = 200)

        update_func1 = lambda a: update(a, x1, w1, N1, M_tot, Nbar, ii=step1_ii)

        #irwls_res = my_irwls(x1,yp1,initial_w1,update_func1,num_blocks = 200) # split this into two parts now, the weights and the irwls result
        num_blocks = 200
        separators = None
        
        w1,wx1,wy1 = get_w_and_weighted_x_y(x1,yp1,initial_w1,update_func1,num_blocks = num_blocks,num_weight_iteration = num_weight_iteration)
        irwls_res = jackknife_fast(wx1,wy1,num_blocks = num_blocks,separators = separators)
        step1_int,step1_se = get_intercept(irwls_res)
        
        if return_step1_only:
            jk_comb_res = format_single_jackknife(irwls_res,step1_int,True)
            final_weights = w1
            intercept_out = step1_int
        else:
            yp = yp - step1_int
            yp = yp.reshape(-1,1)
            x = remove_intercept(x)
            x_tot = remove_intercept(x_tot)


            s = update_separators(irwls_res[1], step1_ii)

            ### DO STEP 2


            update_func2 = lambda a: update(a, x_tot, w, N, M_tot, Nbar, step1_int)

            #irwls_res2 = my_irwls(x,yp,initial_w,update_func2,separators = s) # split this into two parts now, the weights and the irwls result
            num_blocks = None
            w2,wx2,wy2 = get_w_and_weighted_x_y(x,yp,initial_w,update_func2,separators = s,num_weight_iteration = num_weight_iteration)
            irwls_res2 = jackknife_fast(wx2,wy2,num_blocks = num_blocks,separators = s)
            
            c = np.sum(np.multiply(initial_w, x))/np.sum(np.multiply(initial_w, np.square(x)))
            jk_comb_res = combine_twostep_jknife(irwls_res,irwls_res2,c,Nbar)
            
            final_weights = w2
    else:
        update_func = lambda a: update(a, x_tot, w, N, M_tot, Nbar, intercept)
        #jk_comb_res = format_single_jackknife(my_irwls(x, yp,initial_w, update_func, num_blocks= 200),intercept)
        num_blocks = 200
        separators = None
        
        w1,wx1,wy1 = get_w_and_weighted_x_y(x, yp,initial_w, update_func, num_blocks= 200,num_weight_iteration = num_weight_iteration)
        irwls_res = jackknife_fast(wx1,wy1,num_blocks = num_blocks,separators = separators)
        jk_comb_res = format_single_jackknife(irwls_res,intercept)
        
        final_weights = w1
        
    if intercept is not None:
        intercept_out = intercept
    else:
        intercept_out = step1_int
    coef, coef_cov, coef_se = ldsc_coef(jk_comb_res, Nbar)
    cat, cat_cov, cat_se = ldsc_cat(jk_comb_res, M, Nbar, coef, coef_cov)
    tot, tot_cov, tot_se = ldsc_tot(cat, cat_cov)
    
    return (tot,tot_se,intercept_out,final_weights)
    

    
    
    
### CRUDE ESTIMATIONS (UNWEIGHTED, FIX INTERCEPT TO 1)

def ldsc_h2_crude(y,x,M,N,intercept):
    yp = y.reshape(-1,1) - intercept
    result = sm.OLS(yp, x).fit()
    return np.multiply(result.params[0],M/np.mean(N)).item()
    
### CRUDE ESTIMATIONS (UNWEIGHTED, CALCULATE A FREE INTERCEPT)

    
    
def do_sumstat_ldsc_crude(sumstat_df,intercept = 1):
    ldsc_path = base_link+'GWASH_vs_ldsc/ldsc/'
    ldscore_path = ldsc_path+'eur_w_ld_chr/'
    regression_path = ldsc_path+'eur_w_ld_chr/' # this is the same, but can change with other datasets later.
    #ldscore_path = ldsc_path +'myeur_phase3_hapmap3_ldscores/'
    #regression_path = ldsc_path +'myeur_phase3_hapmap3_ldscores/'
    

    ld_scores = [pd.read_table(ldscore_path + str(z)+'.l2.ldscore.gz') for z in range(1,23)]
    ld_scores = pd.concat(ld_scores,axis = 0) # 1290028; matches ldsc code

    #sumstat_df = dt.fread(path_to_sumstats+sumstat_file).to_pandas()
    #sumstat_df
    sumstat_ld = pd.merge(ld_scores,sumstat_df,how = 'inner',on = 'SNP') # 1119409 SNPS remain; matches ldsc code

    # the same as ld_scores
    regression_snps = [pd.read_table(ldscore_path + str(z)+'.l2.ldscore.gz') for z in range(1,23)]
    regression_snps = pd.concat(regression_snps,axis = 0)


    common = True # get # SNPs with MAF > 0.05 in annotation only.
    # SNPs with MAF > 0.05 are considered to be "common"
    #Otherwise, get # of all SNPs
    #https://github.com/bulik/ldsc/wiki/LD-File-Formats#l2m_5_50
    '''
    One line, # of columns = number of annotations in the accompanying .l2.ldscore file in the same order.
    Each column contains the number of SNPs in the corresponding annotation category with MAF > 5%.
    The .l2.M file format is the same, except without the restriction on MAF.
    There are no .l2.M files released.
    '''

    if common:
        M_files = [pd.read_table(ldscore_path +str(z)+'.l2.M_5_50',header = None) for z in range(1,23)]
    else:
        M_files = [pd.read_table(ldscore_path +str(z) + '.l2.M',header = None) for z in range(1,23)]

    sumstat_ld_reg = pd.merge(sumstat_ld,regression_snps, how = 'inner', on = 'SNP') # 1119409 SNPs in common

    l2s = [[z for z in ld_scores.columns][-1]] # in this dataset, there's only one annotation

    if len(l2s) > 1:
        sumstat_ld_reg = sumstat_ld_reg[sumstat_ld_reg['CHISQ'] <= max(0.001*sumstat_ld_reg.N.max(), 80)]


    # M files show how many SNPs that have MAF > 0.05.
    M_annot = pd.concat(M_files,axis = 0).sum(axis = 0).to_numpy()
    M = M_annot


    N = sumstat_ld_reg['N'].to_numpy() # Sample Size PER SNP # THIS IS N in the ldsc code
    w = sumstat_ld_reg['L2_y'].to_numpy().reshape(-1,1) # regression weights 
    
    if 'CHISQ' not in sumstat_ld_reg.columns:
        sumstat_ld_reg['CHISQ'] = [z**2 for z in sumstat_ld_reg['Z']]
        
    y = sumstat_ld_reg['CHISQ'].to_numpy()
    
    x = sumstat_ld_reg['L2_x'].to_numpy().reshape(-1,1)

    ldsc_h2_est_crude = ldsc_h2_crude(y,x,M,N,intercept = intercept)
    return ldsc_h2_est_crude