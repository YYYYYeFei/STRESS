import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import scipy.sparse as sp
from anndata import AnnData
import h5py  # Needed for Slide-seq data loading
import warnings
from mode.PreProcessing import create_file_list
warnings.filterwarnings("ignore")


# --- Utility Functions (unchanged) ---

def extract_hvg_from_reference(outs_path, n_top_genes=2048, sample_type="visiumhd"):
    """
    从参考样本中提取 top N HVG gene 名单.
    根据 `sample_type` 参数，支持 VisiumHD 或 Slide-seq 数据。
    """
    print(f"🔍 提取 HVG 基因列表 from: {outs_path} (Sample Type: {sample_type})")

    if sample_type == "visiumhd":
        matrix_file = os.path.join(outs_path, "filtered_feature_bc_matrix.h5")
        adata = sc.read_10x_h5(matrix_file)
    elif sample_type == "slideseq":
        adata = load_slide_seq_h5(outs_path)  # For Slide-seq, outs_path is the .h5 file directly
    else:
        raise ValueError("Unsupported sample type. Choose 'visiumhd' or 'slideseq'.")

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat", subset=False)

    hvg_genes = adata.var[adata.var["highly_variable"]].index.tolist()
    print(f"✅ 提取到 HVG 基因数: {len(hvg_genes)}")
    return hvg_genes


def build_grid_h5ad(grid_expression, grid_size, gene_names, save_path):
    """构建网格化表达的 h5ad 文件"""
    H, W, G = grid_expression.shape
    grid_flat = grid_expression.reshape(-1, G)

    # Filter out grid cells with no expression
    non_zero_idx = np.where(np.any(grid_flat > 0, axis=1))[0]
    grid_flat = grid_flat[non_zero_idx]

    grid_y = np.repeat(np.arange(H), W)
    grid_x = np.tile(np.arange(W), H)
    obs_names = [f"{y}_{x}" for y, x in zip(grid_y, grid_x)]
    obs = pd.DataFrame(index=np.array(obs_names)[non_zero_idx])
    obs["grid_y"] = grid_y[non_zero_idx]
    obs["grid_x"] = grid_x[non_zero_idx]

    var = pd.DataFrame(index=gene_names)

    adata_grid = AnnData(X=sp.csr_matrix(grid_flat), obs=obs, var=var)
    adata_grid.obsm["spatial"] = np.vstack([obs["grid_x"].values, obs["grid_y"].values]).T
    adata_grid.write_h5ad(save_path)
    print(f"✅ Grid h5ad saved to {save_path}")


def load_slide_seq_h5(file_path):
    """Load Slide-seq data from a custom .h5 file into an AnnData object."""
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'])  # spot by gene count matrix
        Y = np.array(f['Y']) if 'Y' in f else None  # true labels (optional)
        cell = np.array(f['cell']).astype(str)  # spot barcode
        gene = np.array(f['gene']).astype(str)  # gene names
        pos = np.array(f['pos'])  # spatial location (x, y)

    # Construct DataFrame for AnnData
    df_counts = pd.DataFrame(X, columns=gene, index=cell)

    obs_data = pd.DataFrame(index=cell)
    obs_data['x'] = pos[:, 0]
    obs_data['y'] = pos[:, 1]
    if Y is not None:
        obs_data['label'] = Y

    adata = sc.AnnData(df_counts)
    adata.obs = obs_data
    adata.obsm["spatial"] = pos  # Store spatial coordinates in .obsm['spatial'] for consistency
    return adata


# --- Main Processing Function (modified) ---
def process_spatial_sample(outs_path, grid_size, output_npy, hvg_genes, save_h5ad_path, sample_type="visiumhd"):
    """
    处理单个空间转录组样本，映射到统一 HVG 基因空间，并网格化保存。
    根据 `sample_type` 参数，支持 VisiumHD 或 Slide-seq 数据。
    """
    print(f"🔄 处理样本: {outs_path} (类型: {sample_type})")

    if sample_type == "visiumhd":
        matrix_file = os.path.join(outs_path, "filtered_feature_bc_matrix.h5")
        adata = sc.read_10x_h5(matrix_file)
        pos_file = os.path.join(outs_path, "spatial/tissue_positions.parquet")
        pos_df = pd.read_parquet(pos_file)
        pos_df.set_index("barcode", inplace=True)
        adata.obs = adata.obs.join(pos_df, how="left")
        adata = adata[~adata.obs["pxl_row_in_fullres"].isna()].copy()
        adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    elif sample_type == "slideseq":
        adata = load_slide_seq_h5(outs_path)  # For Slide-seq, outs_path is the .h5 file directly
    else:
        raise ValueError("Unsupported sample type. Choose 'visiumhd' or 'slideseq'.")

    # Normalize + log1p
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Subset to HVG and ensure order consistency
    adata.var_names_make_unique()
    # Ensure all HVG genes are present in the adata object before subsetting
    missing_hvg = [gene for gene in hvg_genes if gene not in adata.var_names]
    if missing_hvg:
        print(f"⚠️ Warning: Missing HVG genes in current sample: {missing_hvg}. These will be skipped.")
        hvg_genes = [gene for gene in hvg_genes if gene in adata.var_names]

    adata = adata[:, hvg_genes].copy()

    # Map to grid coordinates
    coords = adata.obsm["spatial"]
    # Handle potential 0-division if all coords are the same (e.g., single spot data)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    # Avoid division by zero if max_x == min_x or max_y == min_y
    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x == 0:
        grid_x = np.zeros_like(coords[:, 1], dtype=int)
    else:
        grid_x = ((coords[:, 1] - min_x) / range_x * (grid_size - 1)).astype(int)

    if range_y == 0:
        grid_y = np.zeros_like(coords[:, 0], dtype=int)
    else:
        grid_y = ((coords[:, 0] - min_y) / range_y * (grid_size - 1)).astype(int)

    # Initialize grid expression tensor
    gene_dim = adata.shape[1]
    grid_expression = np.zeros((grid_size, grid_size, gene_dim), dtype=np.float32)
    grid_counts = np.zeros((grid_size, grid_size), dtype=np.int32)

    # --- FIX START ---
    # Check if adata.X is a sparse matrix, if so, convert it to a dense array
    if sp.issparse(adata.X):
        dense_X = adata.X.toarray()
    else:
        dense_X = adata.X  # It's already a numpy array
    # --- FIX END ---

    for i in tqdm(range(adata.shape[0]), desc=f"Mapping to {grid_size}×{grid_size} grid"):
        gx, gy = grid_x[i], grid_y[i]
        # Ensure grid indices are within bounds
        if 0 <= gy < grid_size and 0 <= gx < grid_size:
            grid_expression[gy, gx, :] += dense_X[i]
            grid_counts[gy, gx] += 1

    nonzero_mask = grid_counts > 0
    for g in range(gene_dim):
        grid_expression[:, :, g][nonzero_mask] /= grid_counts[nonzero_mask]

    np.save(output_npy, grid_expression)
    print(f"✅ Saved: {output_npy}")

    build_grid_h5ad(grid_expression, grid_size, hvg_genes, save_h5ad_path)

def simulated_preprocess(args):
    # =========================== VisiumHD 处理流程 ===========================
    base = os.path.join(args.data_root, f"Simulated/{args.tissue}")
    root_dir = base+"/RAW/"
    output_directory = base+"/Input"
    if args.tissue=='VisiumHD':
        sample_name = "FF_Human_Tonsil"
    elif args.tissue == 'Slide-seq':
        sample_name = "Mouse_hippocampus_Slide_seq_v2"
    else:
        sample_name=None

    # 确保输出目录存在
    os.makedirs(os.path.join(output_directory, "HR"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "GT"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "Reference"), exist_ok=True)

    print("\n--- Processing VisiumHD Data ---")

    # Step 1: 从高分辨率 VisiumHD 样本提取统一 HVG
    if args.tissue=='VisiumHD':
        ref_outs = os.path.join(root_dir, f"{sample_name}/square_008um")
        sample_type="visiumhd"
    else:
        ref_outs = os.path.join(root_dir, f"{sample_name}.h5")
        sample_type = "slideseq"
    hvg_genes = extract_hvg_from_reference(ref_outs, n_top_genes=2048, sample_type=sample_type)

    # Step 2: 处理多个 VisiumHD 分辨率
    process_spatial_sample(
        outs_path=ref_outs,
        grid_size=128,
        output_npy=os.path.join(output_directory, f"HR/{sample_name}_HR.npy"),
        hvg_genes=hvg_genes,
        save_h5ad_path=os.path.join(output_directory, f"Reference/{sample_name}_128.h5ad"),
        sample_type=sample_type
    )

    process_spatial_sample(
        outs_path=ref_outs,
        grid_size=256,
        output_npy=os.path.join(output_directory, f"GT/{sample_name}_GT.npy"),
        hvg_genes=hvg_genes,
        save_h5ad_path=os.path.join(output_directory, f"Reference/{sample_name}_256.h5ad"),
        sample_type=sample_type
    )

    print("\n>>Creating a list of files")
    create_file_list(args,Path(output_directory))