"""
Author: Xiuyuan Wang
Date: 2025-06
"""
import glob
import re

import numpy as np
import pandas as pd
import scanpy as sc
import os
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
# ==============================
# Constants
# ==============================
GENE_COUNT = 2048
MATRIX_SIZE = 128
TARGET_CLUSTERS = 7

def convert_raw_to_h5ad(raw_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(raw_dir.glob("*.h5"))
    label_files = sorted(raw_dir.glob("*.csv"))
    pos_files = sorted(raw_dir.glob("*.txt"))

    for h5_file, label_file, pos_file in zip(h5_files, label_files, pos_files):
        sample_id = h5_file.name.split('_')[0]

        adata = sc.read_10x_h5(h5_file)
        adata.var_names_make_unique()

        labels = pd.read_csv(label_file)
        labels["barcode"] = labels["key"].str.split("_").str[1]
        labels.set_index("barcode", inplace=True)
        adata.obs["ground_truth"] = adata.obs_names.map(labels["ground_truth"])

        position_df = pd.read_csv(
            pos_file,
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"]
        )
        position_df.set_index("barcode", inplace=True)
        for col in position_df.columns:
            adata.obs[col] = adata.obs_names.map(position_df[col])

        out_path = output_dir / f"{sample_id}_with_label_and_position.h5ad"
        adata.write(out_path)
        print(f"[Step 1] Saved: {out_path}")

# ==============================
# Step 2: Convert h5ad to .npy matrices
# ==============================
def downsample_matrix_odd_1(matrix):
    downsampled = matrix[::2, ::2, :].copy()
    downsampled[::2, ::2, :] = 0
    downsampled[1::2, 1::2, :] = 0
    return downsampled

def downsample_matrix_odd_2(matrix):
    downsampled = matrix[::2, ::2, :].copy()
    downsampled[1::2, ::2, :] = 0
    downsampled[::2, 1::2, :] = 0
    return downsampled

def downsample_matrix_even_1(matrix):
    downsampled = matrix[1::2, 1::2, :].copy()
    downsampled[::2, ::2, :] = 0
    downsampled[1::2, 1::2, :] = 0
    return downsampled

def downsample_matrix_even_2(matrix):
    downsampled = matrix[1::2, 1::2, :].copy()
    downsampled[1::2, ::2, :] = 0
    downsampled[::2, 1::2, :] = 0
    return downsampled

def convert_h5ad_to_npy(sample_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    files=['HR','LR_even_1','LR_even_2','LR_odd_1','LR_odd_2']
    for f in files:
        output_path=output_dir / f"{f}"
        output_path.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(sample_file)
    sample_id = sample_file.name.split('_')[0]
    adata = adata[:, ~adata.var_names.duplicated()].copy()

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)

    hvg_genes = adata.var[adata.var['highly_variable']].sort_values(
        by='dispersions_norm', ascending=False
    ).index[:GENE_COUNT]
    adata = adata[:, hvg_genes]

    X = adata.X.toarray()
    HR = np.zeros((MATRIX_SIZE, MATRIX_SIZE, GENE_COUNT))
    rows = adata.obs['array_row'].astype(int).values
    cols = adata.obs['array_col'].astype(int).values
    for i in range(len(rows)):
        HR[rows[i], cols[i], :] = X[i, :]

    np.save(output_dir / 'HR' / f"{sample_id}_HR.npy", HR)
    np.save(output_dir / 'LR_even_1' / f"{sample_id}_LR_even_1.npy", downsample_matrix_even_1(HR))
    np.save(output_dir / 'LR_even_2' / f"{sample_id}_LR_even_2.npy", downsample_matrix_even_2(HR))
    np.save(output_dir / 'LR_odd_1' / f"{sample_id}_LR_odd_1.npy", downsample_matrix_odd_1(HR))
    np.save(output_dir / 'LR_odd_2' / f"{sample_id}_LR_odd_2.npy", downsample_matrix_odd_2(HR))

    print(f"[Step 2] Saved HR/LR for: {sample_id}")

def create_file_list(args,input_dir):
    file_list = sorted(os.listdir(input_dir / 'HR'))
    samples = []
    for f in file_list:
        samples.append("_".join(re.split('_|\.', f)[:-2]))
    samples = pd.DataFrame(samples, columns=['Sample'])
    samples.to_csv(str(input_dir)+f'/{args.tissue}_samplelist.csv', index=False)
    print(f"[Step 3] Saved: {input_dir}")

def Prepocessing(args):
    base = Path(os.path.join(args.data_root,f"{args.tissue}"))
    raw = base / "RAW/"
    matched = base / "RAW/Matched/"
    npy_out = base / "Input/"

    print(">> Step 1: Convert raw files to h5ad")
    convert_raw_to_h5ad(raw, matched)

    print("\n>> Step 2: Convert h5ad to npy matrices")
    h5ad_files = sorted(matched.glob("*.h5ad"))
    for f in h5ad_files:
        convert_h5ad_to_npy(f, npy_out)

    print("\n>> Step 3: Creating a list of files")
    create_file_list(args,npy_out)



