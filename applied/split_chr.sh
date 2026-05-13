mkdir -p 1kg_p1_eur/
for i in {1..22}
do
  plink --allow-extra-chr --bfile 1kg_phase1_all/1kg_phase1_all --chr ${i} --keep all_eur_samples.txt --extract eur_w_ld_chr_snps/snps_to_keep_chr${i} --make-bed --out 1kg_p1_eur/1kg_p1_eur_chr${i}
done
