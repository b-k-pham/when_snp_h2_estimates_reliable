"""

These are functions used to conduct GWASH, LD Score Regression, HEELS, and GCTA
Heritability Estimation comparisons in simulation as shown in Pham et al. 2025.

This specific file contains code used for generating plots.
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
from mizani.formatters import label_number

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


class visualize:
    def __init__(self,res_dict_raw,res_dict,xparam,h2_pop,types_to_plot = None, h2_reasonable_ylim = True,CI_level = 0.95,CI_mode = 'sd',plot_h2_samp = True,draw_whisker_mode = 'default',draw_outliers = False, publication = False,return_studies_plot_df = False,sd_constant = 1,plot_se_mode = 'all',legend_position = 'none',legend_title = element_blank()):
        self.res_dict_raw = res_dict_raw
        self.res_dict = res_dict
        self.xparam = xparam
        self.h2_pop = h2_pop
        self.types_to_plot = types_to_plot
        self.h2_reasonable_ylim = h2_reasonable_ylim
        self.CI_level = CI_level
        self.CI_mode = CI_mode
        self.plot_h2_samp = plot_h2_samp
        self.draw_whisker_mode = draw_whisker_mode
        self.draw_outliers = draw_outliers
        self.publication = publication # Should the figures be publication ready?
        # This means only h2_gwash, h2_ldsc_free, h2_ldsc_fixed, 
        self.sd_constant = sd_constant
        self.plot_se_mode = plot_se_mode #change se plot mode.
        self.legend_position = legend_position #legend position
        self.legend_title = legend_title
        self.return_studies_plot_df = return_studies_plot_df
        
        if self.publication:
            res_dict_raw_new = dict()
            res_dict_raw_new_w_se = dict()
            q = self.res_dict_raw
            for key in q.keys():
                res_dict_raw_new[key] = q[key][[z for z in q[key].columns if z in ['h2_gcta','h2_gwash','h2_ldsc_reg','h2_ldsc_fixed']]]
                res_dict_raw_new_w_se[key] = q[key][[z for z in q[key].columns if z in ['h2_gcta','h2_gwash','h2_ldsc_reg','h2_ldsc_fixed','h2_gwash_sample_theoretical_se','h2_ldsc_reg_jackknife_se','h2_ldsc_fixed_jackknife_se']]]
            self.res_dict_raw = res_dict_raw_new
            self.res_dict_raw_w_se  = res_dict_raw_new_w_se
        
    def convert_to_long(self,h2_df,key):
        return pd.DataFrame({'h2':h2_df[key],'type':key,'sim_n':h2_df.index})

    def prep_a_h2_long_df(self,key):
        df = self.res_dict_raw[key]
        h2_df = df[[z for z in df.columns if 'h2' in z]]
        h2_long = []
        for z in h2_df.columns:
            h2_long.append(self.convert_to_long(h2_df,z))
        h2_long = pd.concat(h2_long,axis = 0).reset_index(drop = True)
        h2_long[self.xparam] = key
        return h2_long

    def prep_all_h2_long_df(self):
        keys = self.res_dict_raw.keys()
        h2_long_dfs = []
        for key in keys:
            h2_long_dfs.append(self.prep_a_h2_long_df(key))
        h2_long_dfs = pd.concat(h2_long_dfs,axis = 0).reset_index(drop = True)
        return h2_long_dfs
    def create_est_sd_df(self):
        keys =  self.res_dict_raw.keys()
        est_sd_dfs = []
        for key in keys:
            temp = self.res_dict_raw[key]
            
            h2_cols = self.types_to_plot
            if h2_cols is None: # if None, then just make it all h2 columns excluding h2_samp
                h2_cols = [z for z in temp.columns if 'h2' in z and 'h2_samp' not in z]
            
            if self.plot_h2_samp: # add it back in if needed.
                h2_cols = [z for z in h2_cols] + ['h2_samp']
            
            
                
            temp = temp[h2_cols]
            est_df = pd.DataFrame(temp.mean(axis = 0))
            num_sims = len(temp) # for CI calculation
            est_df['type'] = est_df.index
            est_df = est_df.reset_index(drop = True)
            est_df.columns = ['est','type']
            

            sd_df = pd.DataFrame(temp.std(axis = 0,ddof = 1))
            sd_df['type'] = sd_df.index
            sd_df = sd_df.reset_index(drop = True)
            sd_df.columns = ['sd','type']
            
            quantile_df = temp.quantile([0.05,0.95]).T
            quantile_df['type'] = quantile_df.index
            quantile_df = quantile_df.reset_index(drop = True)
            quantile_df.columns = ['quantile_lower','quantile_upper','type']

            est_sd_df = pd.merge(est_df,sd_df)
            est_sd_df = pd.merge(est_sd_df,quantile_df)[['est','sd','quantile_lower','quantile_upper','type']]
            # ensure that x entries have same number of sig figs
            key = np.format_float_positional(np.float64(key),min_digits = 2)
            est_sd_df[self.xparam] = key
            
            est_sd_df['sd_plus'] = est_sd_df['est'] + est_sd_df['sd']
            est_sd_df['sd_minus'] = est_sd_df['est'] - est_sd_df['sd']
            
            est_sd_df['sd_plus_user'] = est_sd_df['est'] + self.sd_constant*est_sd_df['sd']
            est_sd_df['sd_minus_user'] = est_sd_df['est'] - self.sd_constant*est_sd_df['sd']
            
            zstat = scipy.stats.norm.ppf(1 - ((1-self.CI_level)/2))
            CI_pm = est_sd_df['sd'] * zstat/np.sqrt(num_sims)
            CI_plus = est_sd_df['est'] + CI_pm
            CI_minus = est_sd_df['est'] - CI_pm
            

            est_sd_df['CI_plus'] = CI_plus
            est_sd_df['CI_minus'] = CI_minus
            
            
            #est_sd_df = pd.concat([est_sd_df,quantile_df],axis = 1)
            

            est_sd_df = est_sd_df[['est','sd_plus','sd_minus','sd_plus_user','sd_minus_user','CI_plus','CI_minus','sd','quantile_lower','quantile_upper','type',self.xparam]]
            est_sd_dfs.append(est_sd_df)
        est_sd_dfs = pd.concat(est_sd_dfs,axis = 0).reset_index(drop = True)
        return est_sd_dfs
    
    def plot_boxplots(self):
        h2_long_df = self.prep_all_h2_long_df()
        h2_long_df['type'] = pd.Categorical(h2_long_df['type'],categories = ['h2_gcta','h2_gwash','h2_weighted_gwash','h2_ols_reg','h2_ldsc_reg','h2_ols_fixed','h2_ldsc_fixed','h2_samp'])
        num_sims = h2_long_df['sim_n'].max() + 1
        breaks_legend = ['h2_gcta','h2_gwash','h2_weighted_gwash','h2_ols_reg','h2_ldsc_reg','h2_ols_fixed','h2_ldsc_fixed'] #ordered h2s
        color_legend = ["purple","green","lightgreen","blue","red",'#6C9CFF','#FB93C4'] #color corresponding to order
        if self.plot_h2_samp == False:
            h2_long_df = h2_long_df[h2_long_df['type'] != 'h2_samp']
        else:
            breaks_legend = breaks_legend + ['h2_samp']
            color_legend = color_legend + ['grey']
        color_obj = scale_fill_manual(color_legend,breaks = breaks_legend)
        guide_obj = guides(fill = guide_legend())
        
        
        def mean_func(y):
            return np.mean(y)

        def ymin_func(y,num_sims = 0,mode = 'sd'):
            if mode == 'sd':
                return np.mean(y) - (self.sd_constant*np.std(y,ddof = 1))
            else:
                return np.mean(y) - (1.96 * np.std(y,ddof = 1))/(np.sqrt(num_sims))

        def ymax_func(y,num_sims = 0,mode = 'sd'):
            if mode == 'sd':
                return np.mean(y) + (self.sd_constant*np.std(y,ddof = 1))
            else:
                return np.mean(y) + (1.96 * np.std(y,ddof = 1))/(np.sqrt(num_sims))
            
        def ymin_fun_sd(y):
            return ymin_func(y,num_sims = 1,mode = 'sd')
        def ymax_fun_sd(y):
            return ymax_func(y,num_sims = 1,mode = 'sd')
            
        def ymin_fun_CI(y):
            return ymin_func(y,num_sims = num_sims,mode = 'CI')
        def ymax_fun_CI(y):
            return ymax_func(y,num_sims = num_sims,mode = 'CI')
        
        if self.draw_outliers:
            base_boxplot_no_whiskers = ggplot(h2_long_df, aes(self.xparam, "h2", fill="type")) +geom_boxplot(coef = 0) + color_obj + guide_obj
            base_boxplot = ggplot(h2_long_df, aes(self.xparam, "h2", fill="type")) +geom_boxplot() + color_obj + guide_obj
        else:
            base_boxplot_no_whiskers = ggplot(h2_long_df, aes(self.xparam, "h2", fill="type")) +geom_boxplot(coef = 0,outlier_shape = '') + color_obj + guide_obj
            base_boxplot = ggplot(h2_long_df, aes(self.xparam, "h2", fill="type")) +geom_boxplot(outlier_shape = '') + color_obj + guide_obj
        
        if self.draw_whisker_mode == 'CI':
            boxplot_obj =  base_boxplot_no_whiskers + stat_summary(aes(fill = 'type'),fun_y=mean_func,geom='point',position = position_dodge(width = 0.8),size = 2) + stat_summary(fun_y=mean_func,fun_ymin=ymin_fun_CI,fun_ymax=ymax_fun_CI,geom='errorbar',color = 'black',position = position_dodge(width = 0.8))
        elif self.draw_whisker_mode == 'sd':
            boxplot_obj = base_boxplot_no_whiskers + stat_summary(aes(fill = 'type'),fun_y=mean_func,geom='point',position = position_dodge(width = 0.8),size = 2) + stat_summary(fun_y=mean_func,fun_ymin=ymin_fun_sd,fun_ymax=ymax_fun_sd,geom='errorbar',color = 'black',position = position_dodge(width = 0.8))
        else:
            boxplot_obj = base_boxplot
        
        if self.h2_reasonable_ylim:
            return  boxplot_obj+ coord_cartesian(ylim=(0, 1)) + geom_hline(aes(yintercept = self.h2_pop),color = 'red',linetype='dotted')
        else:
            return boxplot_obj + geom_hline(aes(yintercept = self.h2_pop),color = 'red',linetype='dotted')
        
    def plot_lineplot(self):
        h2_est_sd_df = self.create_est_sd_df().fillna(0)
        h2_est_sd_df = h2_est_sd_df[~h2_est_sd_df['type'].str.contains("var")] # only get estimates, not the theoretical variances.
        cats = h2_est_sd_df[self.xparam].unique()
        h2_est_sd_df[self.xparam] = pd.Categorical(h2_est_sd_df[self.xparam],categories = cats,ordered = True)
        n_xparam = len(h2_est_sd_df[self.xparam].unique())
        
        gray = '#666666'
        purple = '#9933FF'
        mytheme = theme(panel_grid=element_line(color=purple),
                        panel_grid_major=element_line(size=1.4, alpha=0),
                        panel_grid_minor=element_line(size=1.4, alpha=0),
                        plot_title_position ='none',
                        axis_title=element_text(size=27),
                        axis_text=element_text(size=25),
                        axis_ticks_major=element_line(size=1),
                        axis_ticks_minor=element_line(size=1),
                        legend_title = self.legend_title,
                        legend_text=element_text(size=20),
                        figure_size=(8, 7),
                        legend_position=self.legend_position) #plot_title=element_text(size=20),
                        
                
                
        
        # later, add h2_gwash_weighted
        #h2_est_sd_df['type'] = pd.Categorical(h2_est_sd_df['type'],categories = ['h2_gcta','h2_gwash','h2_ols_reg','h2_ols_fixed','h2_ldsc_reg','h2_ldsc_fixed','h2_samp'])
        h2_est_sd_df['type'] = pd.Categorical(h2_est_sd_df['type'],categories = ['h2_gcta','h2_gwash','h2_weighted_gwash','h2_ols_reg','h2_ldsc_fixed','h2_ols_fixed','h2_ldsc_reg','h2_ldsc_reg_corrected','h2_heels','h2_samp'])
        
        
        breaks_legend = ['h2_gcta','h2_gwash','h2_weighted_gwash','h2_ols_reg','h2_ldsc_reg','h2_ols_fixed','h2_ldsc_fixed','h2_ldsc_reg_corrected','h2_heels'] #ordered h2s
        color_legend = ["purple","green","lightgreen","blue","red",'#6C9CFF','#FB93C4','orange','violet'] #color corresponding to order
        shape_legend = ['^','.','o','d','d','D','D','d','x']
        
        if self.plot_h2_samp:
            breaks_legend = breaks_legend + ['h2_samp']
            color_legend = color_legend + ['grey']
            shape_legend = shape_legend + ['p']
            
        
            
            
        shape_dict = pd.DataFrame(shape_legend,breaks_legend).T.to_dict()
        if self.publication:
            shapes = [shape_dict[z][0] for z in h2_est_sd_df['type']]
        else:
            shapes = [shape_dict[z][0] for z in h2_est_sd_df['type']]
        
        h2_est_sd_df['shape'] = shapes
       
        #visualize(res_dict_raw,res_dict,'sigma_s',h2_pop = 0.2,plot_CIs = True,CI_level = 0.95, h2_reasonable_ylim = False,plot_h2_samp = False).plot_lineplot()+ scale_color_manual(["purple","green","blue","red",'#6C9CFF','#FB93C4'],breaks = ['h2_gcta','h2_gwash','h2_ols_reg','h2_ols_fixed','h2_ldsc_reg','h2_ldsc_fixed']) + scale_shape_identity() + guides(color = guide_legend(override_aes = {'shape': ['^','o',6,6,7,7]}) )

        points = geom_point(position = position_dodge(width = 1),size = 6)
        
        h2_pop_line =  geom_hline(yintercept = self.h2_pop, linetype = 'dashed',color = 'red')
        borders = geom_vline(xintercept = [z +0.5 for z in range(n_xparam + 1)],color = 'white',size = 0.5)
        err_bar = geom_errorbar(h2_est_sd_df,aes(ymin = 'CI_minus', ymax = 'CI_plus'),size = 1.25,position = position_dodge(width = 1))# lower bar is mean - 1.96 sd/np.sqrt(num_sims), upper bar is mean + 1.96 sd/np.sqrt(num_sims)
        sd_bar = geom_errorbar(h2_est_sd_df,aes(ymin = 'sd_minus_user', ymax = 'sd_plus_user'),size = 1.25,position = position_dodge(width = 1)) # lower bar is mean - sd_constant*sd, upper bar is mean + sd_constant*sd
        quant_bar = geom_errorbar(h2_est_sd_df,aes(ymin = 'quantile_lower', ymax = 'quantile_upper'),size = 1.25,position = position_dodge(width = 1)) # lower bar is 0.05 quantile, upper bar is 0.95 quantile
        sd_bar_default = geom_errorbar(h2_est_sd_df,aes(ymin = 'sd_minus', ymax = 'sd_plus'),size = 1.25,position = position_dodge(width = 1))# lower bar is mean - 1*sd, upper bar is mean + 1*sd
        #vis objs
        if self.publication:
            #breaks_legend = ['h2_gcta','h2_gwash','h2_ldsc_reg','h2_ldsc_fixed'] #ordered h2s
            #color_legend = ["purple","green","red",'#FB93C4'] #color corresponding to order
            #shape_legend = ['^','.','d','D']
            breaks_legend = ['h2_gcta','h2_gwash','h2_ldsc_fixed','h2_ldsc_reg'] #ordered h2s
            color_legend = ["purple","green",'#FB93C4',"red"] #color corresponding to order
            shape_legend = ['^','.','D','d']
        
        if self.publication:
            color_obj = scale_color_manual(color_legend,breaks = breaks_legend,labels = ['GCTA','GWASH','LDSC Fixed Intercept','LDSC Free Intercept'])
        else:
            color_obj = scale_color_manual(color_legend,breaks = breaks_legend)
        guide_obj = guides(color = guide_legend(override_aes = {'shape': shape_legend}))
        plot_obj = ggplot(h2_est_sd_df,aes(self.xparam,'est',color = 'type',shape = 'shape')) + points + h2_pop_line + borders + mytheme
        
        if self.h2_reasonable_ylim:
            plot_obj = plot_obj + coord_cartesian(ylim=(0, 1))
        if self.CI_mode == 'CI':
            plot_obj = plot_obj + err_bar
        elif self.CI_mode == 'sd':
            plot_obj = plot_obj + sd_bar
        elif self.CI_mode == 'quantile':
            plot_obj = plot_obj + quant_bar
        else:
            plot_obj = plot_obj + sd_bar_default
        fmt2 = label_number(accuracy=0.01)  # 2 decimal places
        return plot_obj + color_obj + scale_shape_identity() + guide_obj #+ scale_y_continuous(labels=fmt2)
        
    # Make se plot df ready to be used  in plotnine.
    # res_dict_raw should have ses instead of var.

    
    def plot_seplot(self):
        def make_plot_ready_se_long_df(res_dict_raw,mode = None):
            sim_std_dict = dict() # empirical std dataframe (this is the true value that the standard error estimates should go to)
            se_mean_dict = dict() # average SE dataframe
            se_quantile_dict = dict()
            for k in res_dict_raw.keys():
                # ensure that x entries have same number of sig figs
                k2 = np.format_float_positional(np.float64(k),min_digits = 2)
                sim_std_dict[k2] = res_dict_raw[k][['h2_gwash','h2_ldsc_reg','h2_ldsc_fixed']].std(ddof = 1,axis = 0)
                se_res = res_dict_raw[k][['h2_gwash_sample_theoretical_se','h2_ldsc_reg_jackknife_se','h2_ldsc_fixed_jackknife_se']]
                se_mean_dict[k2] = se_res.mean(axis = 0)
                se_quantile_dict[k2] = se_res.quantile([0.05,0.95])
            sim_std_df = pd.DataFrame.from_dict(sim_std_dict).T
            param_index = sim_std_df.index
            sim_std_df = sim_std_df.reset_index(drop = True)
            
            
            ### MC STD OF h2 ESTIMATE. THIS IS THE TRUE VALUE THE AVG SE SHOULD BE
            sim_std_long = []
            for col in sim_std_df.columns:
                temp = sim_std_df[[col]]
                temp.columns = ['sim_std']
                temp['param'] = param_index
                temp['type'] = col
                sim_std_long.append(temp)
            
            sim_std_long = pd.concat(sim_std_long,axis = 0).reset_index(drop = True)
            
            se_mean_df = pd.DataFrame.from_dict(se_mean_dict).T
            param_index = se_mean_df.index
            se_mean_df = se_mean_df.reset_index(drop = True)
            
            ### AVERAGE of SE ESTIMATE. THIS SHOULD BE NEAR THE MC STD OF h2 ESTIMATE
            se_mean_long = []
            for col in se_mean_df.columns:
                temp = se_mean_df[[col]]
                temp.columns = ['se_mean']
                temp['param'] = param_index
                temp['type'] = col
                se_mean_long.append(temp)
            
            se_mean_long = pd.concat(se_mean_long,axis = 0).reset_index(drop = True)
            se_mean_long['type'] = se_mean_long['type'].replace('_sample_theoretical_se','',regex=True).replace('_jackknife_se','',regex=True)
            
            ### QUANTILE df
            #se_quantile_df = pd.DataFrame.from_dict(se_quantile_dict).T
            #param_index = se_quantile_df.index
            #se_quantile_df = se_quantile_df.reset_index(drop = True)
            
            se_quantile_long = []
            #for col in se_mean_df.columns:
            
            #se_quantile_df
            se_quantile_long = []
            for k in se_quantile_dict.keys():
                se_quantile_df = se_quantile_dict[k]
                for col in se_quantile_df.columns:
                    label = col.replace('_sample_theoretical_se','').replace('_jackknife_se','')
                    temp = se_quantile_df[[col]]
                    temp.columns = ['se_quantile']
                    temp = temp.T
                    temp['type'] = label
                    temp.columns = ['quantile_lower','quantile_upper','type']
                    temp = temp.reset_index(drop = True)
                    temp['param'] = k
                    se_quantile_long.append(temp)
            se_quantile_long = pd.concat(se_quantile_long,axis = 0).reset_index(drop = True)
            #left_join_long_df = pd.concat([se_mean_long,sim_std_long,se_quantile_long],axis = 1)
            #left_join_long_df = left_join_long_df.loc[:,~left_join_long_df.columns.duplicated()]
            
            left_join_long_df = pd.concat([se_mean_long,sim_std_long],axis = 1)
            left_join_long_df = left_join_long_df.loc[:,~left_join_long_df.columns.duplicated()]
            
            se_long_df_final = pd.merge(left_join_long_df,se_quantile_long)

            if mode == 'all':
                se_long_df_final_constr = se_long_df_final
            elif mode == 'no_ldsc_reg':
                se_long_df_final_constr = se_long_df_final[se_long_df_final['type']!='h2_ldsc_reg']
            else:
                raise('mode needs to be defined')

            return se_long_df_final_constr
        mytheme = theme(plot_title=element_text(size=20),
                        axis_title=element_text(size=27),
                        axis_text=element_text(size=25),
                        legend_text=element_text(size=20),
                        figure_size=(8, 7),
                        legend_position=self.legend_position)
        def pretty_breaks_from_data(x, n=5):
            x = np.asarray(x)
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            if xmin == xmax:
                return [xmin]
            # expand a bit
            span = xmax - xmin
            xmin -= 0.05 * span
            xmax += 0.05 * span
            return np.linspace(xmin, xmax, n)
        se_long_df_final_constr = make_plot_ready_se_long_df(self.res_dict_raw_w_se,mode = self.plot_se_mode)
        se_long_df_final_constr['type'] = se_long_df_final_constr['type'].replace('h2_gwash','GWASH').replace('h2_ldsc_fixed','LDSC Fixed Intercept').replace('h2_ldsc_reg','LDSC Free Intercept').replace('h2_ldsc_reg','LDSC Free Intercept')

        quant_bar = geom_errorbar(se_long_df_final_constr,aes(ymin = 'quantile_lower', ymax = 'quantile_upper'),size = 1.25,position = position_dodge(width = 1))
        points = geom_point(position = position_dodge(width = 1),size = 6)
        mc_std = geom_point(se_long_df_final_constr, aes('param', y = 'sim_std', color = 'type'),alpha = 0.7,shape = '*',size = 6,position = position_dodge(width = 0.35))
        
        param = self.xparam
        axis_labels = ylab(r'$\hat{\mathrm{se}}$') + xlab(param)
        
        breaks_legend = ['GWASH','LDSC Fixed Intercept','LDSC Free Intercept'] #ordered h2s
        color_legend = ["green",'#FB93C4','red'] #color corresponding to order
        color_obj = scale_color_manual(color_legend,breaks = breaks_legend)
        fmt2 = label_number(accuracy=0.01)  # 2 decimal places
        breaks_y = pretty_breaks_from_data(se_long_df_final_constr["se_mean"], n=5)
        p = ggplot(se_long_df_final_constr,aes('param','se_mean',color = 'type'))+points + quant_bar + mc_std + axis_labels + color_obj #+ scale_y_continuous(breaks = breaks_y,labels=fmt2)
        
        return p + mytheme
        
    def make_plot_seplot_prop(self):
        def make_plot_ready_se_prop_long_df(res_dict_raw,log = True,mode = None):
            sim_std_dict = dict() # empirical std dataframe (this is the true value that the standard error estimates should go to)
            se_mean_dict = dict() # average SE dataframe
            se_quantile_dict = dict()
            for k in res_dict_raw.keys():
                # ensure that x entries have same number of sig figs
                k2 = np.format_float_positional(np.float64(k),min_digits = 2)
                sim_std_dict[k2] = res_dict_raw[k][['h2_gwash','h2_ldsc_reg','h2_ldsc_fixed']].std(ddof = 1,axis = 0)
                se_res = res_dict_raw[k][['h2_gwash_sample_theoretical_se','h2_ldsc_reg_jackknife_se','h2_ldsc_fixed_jackknife_se']]
                se_prop_df = pd.concat([pd.DataFrame(res_dict_raw[k][['h2_gwash_sample_theoretical_se','h2_ldsc_reg_jackknife_se','h2_ldsc_fixed_jackknife_se']].iloc[:,q]/res_dict_raw[k][['h2_gwash','h2_ldsc_reg','h2_ldsc_fixed']].std(ddof = 1,axis = 0).iloc[q]) for q in range(3)],axis = 1)
                if log:
                    se_prop_df = np.log10(se_prop_df) #base10 is interpretable, exponential not so much
                se_mean_dict[k2] = se_prop_df.mean(axis = 0)
                se_quantile_dict[k2] = se_prop_df.quantile([0.05,0.95])
            se_quantile_long = []
            for k in se_quantile_dict.keys():
                se_quantile_df = se_quantile_dict[k]
                for col in se_quantile_df.columns:
                    label = col.replace('_sample_theoretical_se','').replace('_jackknife_se','')
                    temp = se_quantile_df[[col]]
                    temp.columns = ['se_quantile']
                    temp = temp.T
                    temp['type'] = label
                    temp.columns = ['quantile_lower','quantile_upper','type']
                    temp = temp.reset_index(drop = True)
                    temp['param'] = k
                    se_quantile_long.append(temp)
            se_quantile_long = pd.concat(se_quantile_long,axis = 0).reset_index(drop = True)
            se_quantile_long['type'] = se_quantile_long['type'].replace('h2_gwash','GWASH').replace('h2_ldsc_fixed','LDSC Fixed Intercept').replace('h2_ldsc_reg','LDSC Free Intercept').replace('h2_ldsc_reg','LDSC Free Intercept')
            
            return se_quantile_long
            
        mytheme = theme(plot_title=element_text(size=20),
                axis_title=element_text(size=27),
                axis_text=element_text(size=25),
                legend_text=element_text(size=20),
                figure_size=(8, 7),
                legend_position=self.legend_position)
        se_quantile_long_df = make_plot_ready_se_prop_long_df(self.res_dict_raw_w_se,mode = 'all')
        buffer = 0.5
        lim_num = np.max([abs(z) for z in se_quantile_long_df['quantile_lower'].tolist()] + [z for z in se_quantile_long_df['quantile_upper'].tolist()]).item() + buffer
        param = self.xparam
        breaks_legend = ['GWASH','LDSC Fixed Intercept','LDSC Free Intercept'] #ordered h2s
        color_legend = ["green",'#FB93C4','red'] #color corresponding to order
        color_obj = scale_color_manual(color_legend,breaks = breaks_legend)
        p = ggplot(se_quantile_long_df,aes(x  = 'param',color = 'type')) + geom_errorbar(aes(ymin = 'quantile_lower', ymax = 'quantile_upper'),size = 2.5,position = position_dodge(width = 1)) + geom_hline(yintercept = 0,linetype = 'dashed',size = 1,color = 'blue') + xlab(param) +  ylab(r'$\log_{10}\left(\frac{\hat{\mathrm{se}}}{\mathrm{se}_{\mathrm{MC}}}\right)$') + color_obj #+ coord_flip(ylim=[-lim_num,lim_num])
        return p + mytheme
        
    def plot_studies_passed(self):
        ### Get Studies that pass Z-score threshold (h2_est/se_est)
        # if use_est_se = True, use estimated standard error otherwise use True Standard Errors (Simulated Standard Errors).
        def get_studies_accepted(res_dict_raw,thresh = 4,use_est_se = True,relax_upper_bound = False):
            dfs = []
            if relax_upper_bound:
                upper_bound = np.inf
            else:
                upper_bound = 1
            for key in res_dict_raw.keys():
                test = res_dict_raw[key]
                # compute h2 z-scores
                gwash_h2_is_valid = ((test['h2_gwash'] <= upper_bound) & (test['h2_gwash'] >= 0)).astype(np.float32)
                ldsc_reg_h2_is_valid = ((test['h2_ldsc_reg'] <= upper_bound) & (test['h2_ldsc_reg'] >= 0)).astype(np.float32)
                ldsc_fixed_h2_is_valid = ((test['h2_ldsc_fixed'] <= upper_bound) & (test['h2_ldsc_fixed'] >= 0)).astype(np.float32)

                if use_est_se:
                    gwash_h2_z = test['h2_gwash']/test['h2_gwash_sample_theoretical_se']
                    ldsc_reg_h2_z = test['h2_ldsc_reg']/test['h2_ldsc_reg_jackknife_se']
                    ldsc_fixed_h2_z = test['h2_ldsc_fixed']/test['h2_ldsc_fixed_jackknife_se']
                else:
                    gwash_h2_z = test['h2_gwash']/test['h2_gwash'].std(ddof = 1)
                    ldsc_reg_h2_z = test['h2_ldsc_reg']/test['h2_ldsc_reg'].std(ddof = 1)
                    ldsc_fixed_h2_z = test['h2_ldsc_fixed']/test['h2_ldsc_fixed'].std(ddof = 1)

                gwash_h2_z_is_valid = np.float32(gwash_h2_z > thresh)
                ldsc_reg_h2_z_is_valid = np.float32(ldsc_reg_h2_z > thresh)
                ldsc_fixed_h2_z_is_valid = np.float32(ldsc_fixed_h2_z > thresh)

                gwash_h2_check = gwash_h2_is_valid + gwash_h2_z_is_valid
                ldsc_reg_h2_check = ldsc_reg_h2_is_valid + ldsc_reg_h2_z_is_valid
                ldsc_fixed_h2_check = ldsc_fixed_h2_is_valid + ldsc_fixed_h2_z_is_valid

                

                # check if z scores pass lb. if it does, then good!
                studies_passed_gwash = (gwash_h2_check == 2).mean() * 100
                studies_passed_ldsc_reg = (ldsc_reg_h2_check == 2).mean() * 100
                studies_passed_ldsc_fixed = (ldsc_fixed_h2_check == 2).mean() * 100
                
                studies_passed_df = pd.concat([pd.DataFrame([studies_passed_ldsc_reg,'ldsc_reg']).T,pd.DataFrame([studies_passed_ldsc_fixed,'ldsc_fixed']).T,pd.DataFrame([studies_passed_gwash,'gwash']).T],axis = 0).reset_index(drop = True)
                studies_passed_df.columns = ['perc_passed','method']
                # ensure that x entries have same number of sig figs
                k2 = np.format_float_positional(np.float64(key),min_digits = 2)
                studies_passed_df['val'] = k2
                dfs.append(studies_passed_df)
            dfs = pd.concat(dfs,axis = 0).reset_index(drop = True)
            dfs['perc_passed'] = [float(z) for z in dfs['perc_passed']]
            dfs['method'] = dfs['method'].str.replace('gwash','GWASH').str.replace('ldsc_reg','LDSC Free Intercept').str.replace('ldsc_fixed','LDSC Fixed Intercept')
            dfs['method'] = pd.Categorical(dfs['method'],categories=['GWASH','LDSC Fixed Intercept','LDSC Free Intercept'])
            #dfs['method']= pd.Categorical(dfs['method'], categories=["ldsc_reg", "ldsc_fixed", "gwash"], ordered=True)
            dfs['threshold'] = thresh
            return dfs
        studies_plot_df = get_studies_accepted(self.res_dict_raw_w_se,6,use_est_se = True,relax_upper_bound = False)
        studies_plot_df_use_emp_se = get_studies_accepted(self.res_dict_raw_w_se,6,use_est_se = False,relax_upper_bound = False)
        studies_plot_df['perc_passed_w_emp_se'] = studies_plot_df_use_emp_se['perc_passed']
        thresh = studies_plot_df['threshold'].mean()
        
        mytheme = theme(plot_title=element_text(size=20),
                        axis_title=element_text(size=27),
                        axis_text=element_text(size=25),
                        legend_title=element_text(size=14),
                        legend_text=element_text(size=10),
                        figure_size=(8, 7),
                        legend_position=self.legend_position)

        y_label = 'Percent of Studies Passed'
        param = self.xparam
        x_label = param
        plot_title = r'Percent of Studies Passed | ' + param + ', threshold = ' + str(thresh)

        studies_plot_df['method'] = studies_plot_df['method'].str.replace('gwash','GWASH').str.replace('ldsc_reg','LDSC Free Intercept').str.replace('ldsc_fixed','LDSC Fixed Intercept')
        studies_plot_df['method'] = pd.Categorical(studies_plot_df['method'],categories=['GWASH','LDSC Fixed Intercept','LDSC Free Intercept'])

        p = ggplot(studies_plot_df,aes(x = 'val',y = 'perc_passed',fill = 'method')) + geom_bar(stat = "identity", position="dodge",alpha = 0.3) + geom_bar(aes(x = 'val',y = 'perc_passed_w_emp_se'),stat = 'identity', position = 'dodge') + ylab(y_label) + xlab(x_label) + scale_fill_manual(values=["green", '#FB93C4', "red"],labels = ['GWASH','LDSC Fixed Intercept','LDSC Free Intercept']) + mytheme + ylim(0,100)
        if self.return_studies_plot_df:
            return studies_plot_df
        else:
            return p
    def make_vis_matrix(self,h2_col):
        
        keys1 = self.res_dict.keys()
        temp = dict()
        for key1 in keys1:
            temp[key1] = dict()
            keys2 = self.res_dict[key1].keys()
            for key2 in keys2:
                temp[key1][key2] = self.res_dict[key1][key2][h2_col].item()
        return pd.DataFrame.from_dict(temp)