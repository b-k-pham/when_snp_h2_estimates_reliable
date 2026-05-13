list_path=exclusion_lists/eur_people_to_keep_except_id90.txt

mkdir -p 1kg_p1_eur_ben/
mkdir -p eur_w_ld_chr_ben/
for chr in {1..22};
do
plink --bfile 1kg_phase1_all/1kg_phase1_all --cm-map ALL_1000G_phase1integrated_v3_impute/genetic_map_chr${chr}_combined_b37.txt $chr --extract eur_w_ld_chr_snps/snps_to_keep_chr$chr --chr $chr --keep $list_path --make-bed --out 1kg_p1_eur_ben/1kg_p1_eur_chr${chr} --allow-extra-chr
python ldsc/ldsc.py --bfile 1kg_p1_eur_ben/1kg_p1_eur_chr${chr} --ld-wind-cm 1 --out eur_w_ld_chr_ben/${chr}  --yes-really
done