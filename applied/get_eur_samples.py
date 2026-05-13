import pandas as pd
sample = pd.read_table('ALL_1000G_phase1integrated_v3_impute/ALL_1000G_phase1integrated_v3.sample',sep = ' ')
fam = pd.read_table('1kg_phase1_all/1kg_phase1_all.fam',header = None,sep = ' ')
fam.columns = ['zero','id','rel1','rel2','gender','phenotype']
#set(sample['sample'] == fam['id']) #{True}
fam['group'] = sample['group']

temp = fam[fam['group'] == 'EUR']
eur_fam = temp[['zero','id']]
eur_fam.to_csv('all_eur_samples.txt',sep = '\t', header = None,index = None)
#eur_fam[eur_fam['id'] != 'HG00119'].to_csv('eur_people_to_keep.txt',sep = '\t', header = None,index = None)