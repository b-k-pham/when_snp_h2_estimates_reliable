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