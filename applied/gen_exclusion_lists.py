import pandas as pd
import os

sample = pd.read_table('ALL_1000G_phase1integrated_v3_impute/ALL_1000G_phase1integrated_v3.sample',sep = ' ')
fam = pd.read_table('1kg_phase1_all/1kg_phase1_all.fam',header = None,sep = ' ')
fam.columns = ['zero','id','rel1','rel2','gender','phenotype']
#set(sample['sample'] == fam['id']) #{True}
fam['group'] = sample['group']

temp = fam[fam['group'] == 'EUR']


eur_fam = temp[['zero','id']].reset_index(drop = True)

os.makedirs('exclusion_lists/',exist_ok=True)


for z in range(len(eur_fam['id'])):
    eur_fam[eur_fam['id'] != eur_fam['id'][z]].to_csv('exclusion_lists/eur_people_to_keep_except_id{z}.txt'.format(z = str(z)),sep = '\t', header = None,index = None)