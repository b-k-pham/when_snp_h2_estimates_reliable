import pandas as pd
import numpy as np
orig_ldscores = pd.read_table('eur_w_ld_chr/22.l2.ldscore.gz')
#orig_ldscores['L2']
MSEs = []
for indiv in range(378+1):
    exclusion_ldscores = pd.read_table('ldscores_22_exclude_id/22_without_id{id}.l2.ldscore.gz'.format(id = indiv))['L2']
    MSEs.append(((exclusion_ldscores - np.round(orig_ldscores['L2'],3))**2).mean())

df = pd.DataFrame([z for z in range(379)])
df.columns = ['include_all_except_id']
df['MSE'] = MSEs

df = df.sort_values('MSE')

df.to_csv('MSE_ldscores_22.txt',sep = '\t')