# Applied Data


## Description

These scripts are used to create quantities that can be used to compute $`\tilde{\mu}_{2}`$ and $`\tilde{\mu}_{3}`$ in GWASH with the 1 CM window LDSC uses to compute LD Scores. This allows for the direct comparison between GWASH and LDSC. The ``eur_w_ld_chr_ben.tar.gz`` archive contains the LD Scores for each chromosome constructed from the 1000 Genomes Phase 1 Reference panel using the standard LDSC package and the ``S_tabs.tar.gz`` archive contains these quantities per chromosome with a comparison to the LD Scores produced by LDSC.

The ``ldscore`` directory contains scripts found from the original LDSC repository (https://github.com/bulik/ldsc). The ``ldscore/ldscore.py`` script is modifed to output the window indicies so that one could get the SNPs in the 1 CM window to compute the GWASH quantities.

## Prerequisites
1. plink [https://www.cog-genomics.org/plink/1.9/]
2. LDSC [https://github.com/bulik/ldsc] (needed for comparisons)
3. Released 1000G P1 LD Scores [https://zenodo.org/records/8182036] (needed for comparisons)

## Reproducibility

The prepared panel can be downloaded here: [zenodo link]. For reproducibility, we outline the steps for preprocessing here:

1. Download the 1000 Genomes Phase 1 Panel here [https://www.cog-genomics.org/plink/1.9/resources]
2. Download the cm map as written in the original Bulik-Sulivan et al. 2015 LDSC paper [https://mathgen.stats.ox.ac.uk/impute/data_download_1000G_phase1_integrated.html]. Extract into ALL_1000G_phase1integrated_v3_impute/.
3. Filter SNPs in released ``eur_w_ld_chr/`` with the following code snippet:
```
import pandas as pd

for z in range(1,23):
    temp = pd.read_table('eur_w_ld_chr/{z}.l2.ldscore.gz'.format(z = str(z)))
    temp1 = temp[['SNP']]
    temp1.to_csv('eur_w_ld_chr_snps/snps_to_keep_chr{z}'.format(z = str(z)),header = None,index = None)
```
4. Keep EUR subjects. There should be 379.
```
import pandas as pd
sample = pd.read_table('ALL_1000G_phase1integrated_v3_impute/ALL_1000G_phase1integrated_v3.sample',sep = ' ')
fam = pd.read_table('1kg_phase1_all/1kg_phase1_all.fam',header = None,sep = ' ')
fam.columns = ['zero','id','rel1','rel2','gender','phenotype']
#set(sample['sample'] == fam['id']) #{True}
fam['group'] = sample['group']

temp = fam[fam['group'] == 'EUR']
eur_fam = temp[['zero','id']]
eur_fam.to_csv('all_eur_samples.txt',sep = '\t', header = None,index = None)

```
5. Run the following to split into chromosomes.
```
mkdir -p 1kg_p1_eur/
for i in {1..22}
do
  plink --allow-extra-chr --bfile 1kg_phase1_all/1kg_phase1_all --chr ${i} --keep all_eur_samples.txt --extract eur_w_ld_chr_snps/snps_to_keep_chr${i} --make-bed --out 1kg_p1_eur/1kg_p1_eur_chr${i}
done
```
6. To find the individual excluded, since it was not explicitly written in the main text, each individual is excluded and the LD Scores for Chromosome 22 are computed with the remaining individuals. These LD Scores are compared against Chromosome 22 of the released LD Scores. The version of the computed LD Scores with the lowest MSE is chosen. Clone LDSC from the main repo (https://github.com/bulik/ldsc) and run the following to construct LD Scores with an individual excluded:
```
mkdir -p temp/
mkdir -p ldscores_22_exclude_id/

chr=22
for i in {0..378};
do
list_path=exclusion_lists/eur_people_to_keep_except_id${i}.txt
plink --bfile 1kg_phase1_all/1kg_phase1_all --cm-map ALL_1000G_phase1integrated_v3_impute/genetic_map_chr${chr}_combined_b37.txt $chr --extract eur_w_ld_chr_snps/snps_to_keep_chr$chr --chr $chr --keep $list_path --make-bed --out temp/temp${chr} --allow-extra-chr
python ldsc/ldsc.py --bfile temp/temp22 --ld-wind-cm 1 --out ldscores_22_exclude_id/22_without_id${i} --yes-really
done

rm -r temp/
```
Then compute the MSE between these LD Scores and the released LD Scores.

```
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
```

The LD Scores for Chromosome 22 with the smallest MSE is that without id 90 so this subject is removed.

7. Make plink files for the 378 individuals chosen from step 6. Construct LD Scores from the genotype file.
   
```
list_path=exclusion_lists/eur_people_to_keep_except_id90.txt

mkdir -p 1kg_p1_eur_ben/
mkdir -p eur_w_ld_chr_ben/

for chr in {1..22};
do
plink --bfile 1kg_phase1_all/1kg_phase1_all --cm-map ALL_1000G_phase1integrated_v3_impute/genetic_map_chr${chr}_combined_b37.txt $chr --extract eur_w_ld_chr_snps/snps_to_keep_chr$chr --chr $chr --keep $list_path --make-bed --out 1kg_p1_eur_ben/1kg_p1_eur_chr${chr} --allow-extra-chr
python ldsc/ldsc.py --bfile 1kg_p1_eur_ben/1kg_p1_eur_chr${chr} --ld-wind-cm 1 --out eur_w_ld_chr_ben/${chr}  --yes-really
done
```

8. Now, $`\tilde{\mu}_{2}`$ and $`\tilde{\mu}_{3}`$ need to be constructed per $`k`$th chromosome. This is denoted as $`(\tilde{\mu}_{2})_{k}`$ and $`(\tilde{\mu}_{3})_{k}`$. We first start by creating $\hat{R}$ as done in step 7 and saving it as a sparse matrix for later. This is done with the python script ``save_S_matrix_1cm.py``.

9. Then, an aggregate table saving each $`j`$th-SNP contribution to a $`(\tilde{\mu}_{2})_{k}`$ and $`(\tilde{\mu}_{3})_{k}`$ is created with the python script``make_S_tabs.py``. The output is in a directory called ``S_tabs/``. For instance, the chromosome 1 file would be: ``S_tabs/S_chrom1_1cm_bim.txt``. The expected output containing all 22 chromosomes is in the archive ``S_tabs.tar.gz``. For each file, we also compare each $`j`$th $`(\tilde{\mu}_{2})_{k}`$ SNP contribution to the $`\tilde{\ell}_{j}`$ and see that it is equivalent. Below is the first 5 rows for Chromosome 1:

|    |   chrom | snp        |       cm |    pos | a0   | a1   |   i |   gwash_l2s |   gwash_l3s |   ldsc_l2s |   ldsc_l2s_saved |   abs_diff_gwash_l2s_and_ldsc_l2s |   abs_diff_gwash_l2s_and_ldsc_l2s_saved |
|---:|--------:|:-----------|---------:|-------:|:-----|:-----|----:|------------:|------------:|-----------:|-----------------:|----------------------------------:|----------------------------------------:|
|  0 |       1 | rs12565286 | 0.410292 | 721290 | C    | G    |   0 |     1.24317 |     11.2383 |    1.24317 |            1.243 |                       1.11022e-15 |                                       0 |
|  1 |       1 | rs3094315  | 0.488776 | 752566 | G    | A    |   1 |     6.39312 |     63.7336 |    6.39312 |            6.393 |                       2.13163e-14 |                                       0 |
|  2 |       1 | rs3131972  | 0.488868 | 752721 | A    | G    |   2 |     6.26837 |     61.4436 |    6.26837 |            6.268 |                       5.32907e-15 |                                       0 |
|  3 |       1 | rs3131969  | 0.489734 | 754182 | A    | G    |   3 |     6.24003 |     60.5763 |    6.24003 |            6.24  |                       5.32907e-15 |                                       0 |
|  4 |       1 | rs1048488  | 0.492507 | 760912 | C    | T    |   4 |     6.46808 |     70.3367 |    6.46808 |            6.468 |                       2.39808e-14 |                                       0 |


From this table, one can see that LDSC by default (ldsc_l2s_saved) rounds $`\tilde{\ell}_{j}`$ to the third decimal pt.  Also, each $`j`$th SNP $`(\tilde{\mu}_{2})_{k}`$ (gwash_l2s) is approximate to the unrounded $`\tilde{\ell}_{j}`$ (ldsc_l2s).







