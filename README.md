# When can Whole-Genome SNP-heritability be reliably estimated from summary statistics?


## Description

This repository contains simulation results and code from "When can Whole-Genome SNP-heritability be reliably estimated from summary statistics?". The preprint can be read here [https://www.biorxiv.org/content/10.64898/2026.05.13.724972v1].


## Getting Started

The dependencies for the code are as follows:
```
  - python=3.11
  - jupyterlab
  - pandas
  - scipy
  - tqdm
  - scikit-learn
  - plotnine
  - numba
  - pandas-plink
  - natsort
```

These packages can be installed manually with pip. For convenience, we provide a .yml recipe file for use in conda. You can setup the environment with the command:

```
conda env create -f gwash_env.yml
```

Then, the environment should be activated:

```
conda activate gwash
```

For a quick demo, please go to ``notebooks/Reproducibility.ipynb``.

## Simulation Results

Below is the tree of the main simulation results. In brief, the main simulation results utilize a reference panel that matches the distribution of the sample.
```
├── save_data
│   ├── main_text
│   │   ├── AR1
│   │   │   ├── double_param
│   │   │   │   ├── Fst005pm_causal.pkl
│   │   │   │   ├── Fst005rho.pkl
│   │   │   │   ├── Fst005sigma_s.pkl
│   │   │   │   ├── Fst01pm_causal.pkl
│   │   │   │   ├── Fst01rho.pkl
│   │   │   │   └── Fst01sigma_s.pkl
│   │   │   └── single_param
│   │   │       ├── Fst.pkl
│   │   │       ├── pm_causal.pkl
│   │   │       ├── rho.pkl
│   │   │       └── sigma_s.pkl
│   │   └── realistic
│   │       ├── double_param
│   │       │   ├── Fst005pm_causal.pkl
│   │       │   ├── Fst005sigma_s.pkl
│   │       │   ├── Fst01pm_causal.pkl
│   │       │   └── Fst01sigma_s.pkl
│   │       └── single_param
│   │           ├── Fst.pkl
│   │           ├── pm_causal.pkl
│   │           └── sigma_s.pkl
```
The supplementary results under ``save_data/supplementary`` is structured similarly to the main simulation results and uses individual level data directly.

The simulated data has $n = 5000$ and $m = 10000$. ``AR1`` consists of results for $\mathbf{X}$ with an AR1 correlation structrure. ``realistic`` consists of results with an LD correlation structure from Chromosome 22 in 1000 Genomes Phase 1 where the first $m$ SNPs are used.

The results are saved in a pickle format and can be loaded in python:
```
import pickle
file = open('save_data/main_text/AR1/single_param/rho.pkl')
saved_data = pickle.load(file)
```
``saved_data`` is a dictionary of pandas DataFrames with each key as the parameter value (e.g.: heritability estimates with $\rho = 0.2$ in AR1 can be accessed with ``saved_data['0.2']``). There are 1000 rows corresponding to each simulation replicate under $seed = 123$.

There are 9 columns:

```
h2_gcta	h2_gwash	h2_ldsc_reg	icpt_ldsc_reg	h2_ldsc_fixed	h2_samp	h2_gwash_sample_theoretical_se	h2_ldsc_reg_jackknife_se	h2_ldsc_fixed_jackknife_se
```


- ``h2_gcta`` is the heritability estimate from GCTA.
- ``h2_gwash`` is the heritability estimate from GWASH.
- ``h2_ldsc_reg`` is the heritability estimate from LDSC with estimated free intercept.
- ``icpt_ldsc_reg`` is the corresponding estimated intercept.
- ``h2_ldsc_fixed`` is the heritability estimate from LDSC with intercept constrained to 1.
- ``h2_samp`` is the true FVE computed directly from individual-level data and SNP effect-sizes. This quantity is unknown in practice.
- ``h2_gwash_sample_theoretical_se`` is the standard error for the GWASH heritability estimator.
- ``h2_ldsc_reg_jackknife_se`` is the standard error from block-jackknife for the LDSC with estimated free intercept estimator.
- ``h2_ldsc_fixed_jackknife_se`` is the standard error from block-jackknife for the LDSC with intercept constrained to 1.




