import os
from functools import reduce
import pandas as pd
import numpy as np
import pickle
def combine_all_dicts(list_of_dicts):
    return reduce(lambda a, b: {**a, **b}, list_of_dicts)
    
def make_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
        
 
### DATA WRANGLING


def res_dict_to_dfs(res_dict):
    keys = res_dict.keys()
    est_dfs = []
    std_dfs = []
    for key in keys:
        df = pd.DataFrame.from_dict(res_dict[key])
        est_df = pd.DataFrame(df.mean(axis = 0)).T
        est_df.index = [key]
        std_df = pd.DataFrame(df.std(axis = 0, ddof = 1)).T
        std_df.index = [key]
        est_dfs.append(est_df)
        std_dfs.append(std_df)
    est_dfs = pd.concat(est_dfs,axis = 0)
    std_dfs = pd.concat(std_dfs,axis = 0)
    return est_dfs,std_dfs

def make_combined_df(df_estimates,df_stddev):
    df_estimates = df_estimates.round(3)
    df_stddev = df_stddev.round(3)
    df_combined = df_estimates.astype(str) + ' (' + df_stddev.astype(str) + ')'
    return df_combined

def res_dict_to_latex(res_dict):
    est_df, std_df = res_dict_to_dfs(res_dict)
    df_comb = make_combined_df(est_df,std_df)
    logging.info(df_comb.to_latex().replace("\\\n", "\\ \hline\n"))
    return df_comb

def est_std_df(res_dfs):
    est_df = pd.DataFrame(res_dfs.mean(axis = 0)).T
    std_df = pd.DataFrame(res_dfs.std(axis = 0,ddof = 1)).T
    return make_combined_df(est_df,std_df)
    

def get_savedata(base_path,label):
    raw_path = base_path + 'raw_data/{label}/'.format(label = label)
    tab_path = base_path + 'overleaf_tables/{label}/'.format(label = label)
    raw_files = [raw_path + z for z in os.listdir(raw_path)]
    overleaf_files = [tab_path + z for z in os.listdir(tab_path)]
    return raw_files,overleaf_files
def load_savedata(raw_files,overleaf_files,idx):
    file = open(raw_files[idx], 'rb')
    res_dict_raw = pickle.load(file)

    file = open(overleaf_files[idx], 'rb')
    res_dict = pickle.load(file)
    return res_dict_raw,res_dict
    
def pkl_file_loader(link):
    file = open(link,'rb')
    #print(link)
    pkl_file = pickle.load(file)
    return pkl_file
    
    
