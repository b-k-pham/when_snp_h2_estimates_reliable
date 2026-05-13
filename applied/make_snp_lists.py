import pandas as pd
import os


os.makedirs('eur_w_ld_chr_snps/',exist_ok=True)

for z in range(1,23):
    temp = pd.read_table('eur_w_ld_chr/{z}.l2.ldscore.gz'.format(z = str(z)))
    temp1 = temp[['SNP']]
    temp1.to_csv('eur_w_ld_chr_snps/snps_to_keep_chr{z}'.format(z = str(z)),header = None,index = None)