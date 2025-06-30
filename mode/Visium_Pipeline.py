import os
import re
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
from mode.test import test_hyper
warnings.filterwarnings("ignore")


class VisiumPipeline:
    def __init__(self, args, input_dir, output_dir, analysis_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.analysis_dir = Path(analysis_dir)
        self.args=args

        # Define and create output subdirectories
        self.hr_dir = self.output_dir / 'HR'
        self.or_dir = self.output_dir / 'OR'
        self.enhanced_dir = self.output_dir / 'Enhanced'
        self.plot_dir = self.analysis_dir / 'plots'

        # Ensure all output directories exist
        for dirname in [self.hr_dir, self.or_dir, self.enhanced_dir, self.plot_dir]:
            dirname.mkdir(parents=True, exist_ok=True)

        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("All subdirectories have been created successfully.")

    def raw_to_input(self, sample_name):
        print(f"\n===== Start processing sample: {sample_name} =====")
        sample_path = self.input_dir / sample_name

        try:
            if sample_path.is_dir():
                print(f"Reading 10X Visium sample from directory: {sample_path}")
                adata = sc.read_visium(str(sample_path))
            elif sample_path.with_suffix('.h5ad').exists():
                sample_path = sample_path.with_suffix('.h5ad')
                print(f"Reading AnnData sample from file: {sample_path}")
                adata = sc.read_h5ad(sample_path)
            else:
                raise FileNotFoundError(
                    f"Could not find directory or .h5ad file for sample '{sample_name}' in {self.input_dir}")

            print(f"Processing {sample_path} with original shape: {adata.shape}")

            # --- Data Preprocessing ---
            adata.var_names_make_unique()  # Ensure gene names are unique
            sc.pp.filter_genes(adata, min_counts=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2048, flavor='seurat_v3')

            hvg_adata = adata[:, adata.var['highly_variable']].copy()
            print(f"Selected {hvg_adata.shape[1]} highly variable genes.")

            # --- Create High-Resolution Matrix ---
            rows = hvg_adata.obs['array_row'].astype(int).values
            cols = hvg_adata.obs['array_col'].astype(int).values

            # For consistency, we can fix the size to 128x128, assuming the data fits.
            HR_matrix = np.zeros((128, 128, hvg_adata.shape[1]))
            X_dense = hvg_adata.X.toarray() if not isinstance(hvg_adata.X, np.ndarray) else hvg_adata.X

            for idx in range(len(rows)):
                r, c = rows[idx], cols[idx]
                if r < 128 and c < 128:
                    HR_matrix[r, c, :] = X_dense[idx, :]

            # --- Save Outputs ---
            hr_path = self.hr_dir / f'{sample_name}_HR.npy'
            or_path = self.or_dir / f'{sample_name}_OR.h5ad'

            np.save(hr_path, HR_matrix)
            hvg_adata.write_h5ad(or_path)

            print("\n>>Creating a list of files")
            self.create_file_list(self.args.test_tissue,self.output_dir)

            print(f"High-resolution matrix saved to: {hr_path}")
            print(f"Original-resolution AnnData saved to: {or_path}")
            print(f"Sample {sample_name} processing complete.")

        except Exception as e:
            print(f"An error occurred while processing sample {sample_name}: {e}")

    def create_file_list(self,tissue, input_dir):
        file_list = sorted(os.listdir(input_dir / 'HR'))
        samples = []
        for f in file_list:
            samples.append("_".join(re.split('_|\.', f)[:-2]))
        samples = pd.DataFrame(samples, columns=['Sample'])
        samples.to_csv(str(input_dir) + f'/{tissue}_samplelist.csv', index=False)
        print(f"sample lists file saved: {input_dir}")

    def _refine_points(self, adata):
        x, y = adata.obs['array_col'], adata.obs['array_row']
        new_x, new_y = np.copy(x), np.copy(y)
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            if (x_i % 4 == 1) and (y_i % 4 == 0):
                new_x[i] += 1
            elif (x_i % 4 == 0) and (y_i % 4 == 1):
                new_x[i] -= 1
            elif (x_i % 4 == 3) and (y_i % 4 == 2):
                new_x[i] += 1
            elif (x_i % 4 == 2) and (y_i % 4 == 3):
                new_x[i] -= 1

        adata.obsm["spatial"] = np.column_stack([new_x, new_y])
        adata.obs['array_col'] = new_x
        adata.obs['array_row'] = new_y

    def output_to_h5ad(self, sample_name, model_output_dir, cutoff=1.3):
        print(f"\n===== Start converting model output to H5AD for: {sample_name} =====")

        try:
            pred_path = Path(model_output_dir) / f"{sample_name}_res_fold10.npy"
            or_path = self.or_dir / f'{sample_name}_OR.h5ad'

            if not pred_path.exists():
                raise FileNotFoundError(f"Model output file not found: {pred_path}")
            if not or_path.exists():
                raise FileNotFoundError(f"Original-resolution file not found: {or_path}")

            pred_data = np.load(pred_path)
            adata_or = sc.read_h5ad(or_path)

            print(f"Loaded prediction data with shape: {pred_data.shape}")

            C, H, W = pred_data.shape
            row_idx = np.arange(H).reshape(1, H, 1)
            col_idx = np.arange(W).reshape(1, 1, W)
            block_mask = ((row_idx // 2) + (col_idx // 2)) % 2 == 0
            pred_data = np.where(block_mask, pred_data, 0)

            mask = np.all(pred_data < cutoff, axis=0)
            pred_data[:, mask] = 0

            predicted_data_reshaped = pred_data.reshape(C, -1).T
            adata_pred = sc.AnnData(predicted_data_reshaped)
            adata_pred = adata_pred[adata_pred.X.sum(axis=1) > 0, :].copy()

            non_empty_indices = np.where(predicted_data_reshaped.sum(axis=1) > 0)[0]

            x_coords, y_coords = np.unravel_index(non_empty_indices, (H, W))

            adata_pred.obsm["spatial"] = np.column_stack([x_coords, y_coords])
            adata_pred.obs["array_row"] = y_coords
            adata_pred.obs["array_col"] = x_coords
            adata_pred.var.index = adata_or.var.index
            self._refine_points(adata_pred)

            enhanced_path = self.enhanced_dir / f"{sample_name}_enhanced.h5ad"
            adata_pred.write(enhanced_path)

            print(f"Enhanced-resolution AnnData saved to: {enhanced_path}")
            print(f"Model output processing for sample {sample_name} complete.")

        except Exception as e:
            print(f"An error occurred during model output conversion: {e}")

    def _tune_resolution(self, adata, target_clusters, method="leiden", initial_resolution=1.0, tol=0, max_iter=20):
        """
        A helper function to automatically tune the clustering resolution to achieve a target number of clusters.
        """
        assert method in ["leiden", "louvain"], "Clustering method must be 'leiden' or 'louvain'."

        resolution = initial_resolution
        lower_bound, upper_bound = 0.01, 5.0  # Expanded search range

        for i in range(max_iter):
            if method == "leiden":
                sc.tl.leiden(adata, resolution=resolution, key_added=method)
            else:
                sc.tl.louvain(adata, resolution=resolution, key_added=method)

            num_clusters = adata.obs[method].nunique()
            print(f"Iteration {i + 1}: Resolution={resolution:.4f}, Clusters={num_clusters}")

            if abs(num_clusters - target_clusters) <= tol:
                print(f"Target number of clusters ({target_clusters}) reached.")
                break

            if num_clusters < target_clusters:
                lower_bound = resolution
                resolution = (resolution + upper_bound) / 2
            else:
                upper_bound = resolution
                resolution = (lower_bound + resolution) / 2
        else:
            print(
                f"Max iterations reached. Could not exactly match target clusters. Final cluster count: {num_clusters}")

        return adata, resolution

    def cluster_and_visualize(self, sample_name, target_clusters, labels=['leiden', 'louvain']):
        """
        PART 3: CLUSTER_AND_VISUALIZATION
        Processes, clusters, and visualizes the original and enhanced resolution .h5ad files.

        Workflow:
        1. Load the original and enhanced resolution .h5ad files.
        2. Perform PCA and compute neighbor graphs for both AnnData objects.
        3. Use `_tune_resolution` to find the best resolution for both clustering methods (Leiden, Louvain).
        4. Generate a comparison plot showing clustering results for original and enhanced data.
        5. Save the visualization plot.

        Args:
            sample_name (str): The name of the sample.
            target_clusters (int): The desired number of clusters.
            labels (list): A list of clustering algorithms to use (e.g., ['leiden', 'louvain']).
        """
        print(f"\n===== Start clustering and visualization for: {sample_name} =====")
        try:
            or_path = self.or_dir / f'{sample_name}_OR.h5ad'
            enhanced_path = self.enhanced_dir / f"{sample_name}_enhanced.h5ad"

            if not or_path.exists() or not enhanced_path.exists():
                raise FileNotFoundError("Original and enhanced .h5ad files must be generated first.")

            adata_gt = sc.read_h5ad(or_path)
            adata_pred = sc.read_h5ad(enhanced_path)

            print("Original data (Ground Truth) AnnData loaded successfully.")
            print("Enhanced data (Prediction) AnnData loaded successfully.")

            # --- Process and cluster both anndata objects ---
            for ad, name in zip([adata_gt, adata_pred], ["Original", "Enhanced"]):
                print(f"\n--- Processing {name} data ---")
                sc.pp.pca(ad, n_comps=50)
                sc.pp.neighbors(ad, use_rep='X_pca')
                for method in labels:
                    print(f"Tuning {method} clustering for {name} data...")
                    self._tune_resolution(ad, target_clusters, method=method)

            # --- Save the refined AnnData objects ---
            print("\n--- Saving refined AnnData objects ---")
            self.output_to_h5ad(adata_gt, f"{sample_name}_{target_clusters}_OR_refined.h5ad", self.enhanced_dir)
            self.output_to_h5ad(adata_pred, f"{sample_name}_{target_clusters}_enhanced_refined.h5ad", self.enhanced_dir)

            # --- Visualization ---
            n_labels = len(labels)
            fig, axes = plt.subplots(2, n_labels, figsize=(12 * n_labels, 20), constrained_layout=True)
            fig.suptitle(f'Sample {sample_name} - Target Clusters: {target_clusters}', fontsize=20, weight='bold')

            for i, label in enumerate(labels):
                ax_gt = axes[0, i] if n_labels > 1 else axes[0]
                ax_pred = axes[1, i] if n_labels > 1 else axes[1]

                # Extract coordinates
                gt_x, gt_y = adata_gt.obs['array_col'], adata_gt.obs['array_row']
                pred_x, pred_y = adata_pred.obs['array_col'], adata_pred.obs['array_row']

                # Get unique labels and create a unified color map
                # Ensure the label column exists before trying to get categories
                if label not in adata_gt.obs or label not in adata_pred.obs:
                    print(
                        f"Error: Clustering label '{label}' not found in AnnData objects. Skipping plotting for this label.")
                    continue

                unique_labels = adata_gt.obs[label].astype('category').cat.categories.tolist()
                # Sort unique_labels for consistent color mapping if they are numeric strings
                try:
                    unique_labels.sort(key=int)
                except ValueError:
                    unique_labels.sort()  # Fallback for non-numeric labels

                colors = plt.get_cmap('tab20')(range(len(unique_labels)))
                label_color_map = {cat: colors[j] for j, cat in enumerate(unique_labels)}

                # Map colors to each spot based on its cluster label
                gt_colors = adata_gt.obs[label].astype(str).apply(lambda x: label_color_map.get(x, 'gray')).tolist()
                pred_colors = adata_pred.obs[label].astype(str).apply(lambda x: label_color_map.get(x, 'gray')).tolist()

                # Plot original resolution (Ground Truth)
                ax_gt.scatter(gt_x, gt_y, c=gt_colors, s=50, edgecolors='none')  # spot_size=50
                ax_gt.set_title(f"Original - {label.capitalize()}", fontsize=16)
                ax_gt.set_xlabel("")
                ax_gt.set_ylabel("")
                ax_gt.invert_yaxis()  # Important for spatial plots
                ax_gt.set_xticks([])
                ax_gt.set_yticks([])
                ax_gt.spines['top'].set_visible(False)
                ax_gt.spines['bottom'].set_visible(False)
                ax_gt.spines['left'].set_visible(False)
                ax_gt.spines['right'].set_visible(False)

                # Plot enhanced resolution (Prediction)
                ax_pred.scatter(pred_x, pred_y, c=pred_colors, s=15,
                                edgecolors='none')  # spot_size=1, adjusted to 15 for better visibility as per example
                ax_pred.set_title(f"Enhanced - {label.capitalize()}", fontsize=16)
                ax_pred.set_xlabel("")
                ax_pred.set_ylabel("")
                ax_pred.invert_yaxis()  # Important for spatial plots
                ax_pred.set_xticks([])
                ax_pred.set_yticks([])
                ax_pred.spines['top'].set_visible(False)
                ax_pred.spines['bottom'].set_visible(False)
                ax_pred.spines['left'].set_visible(False)
                ax_pred.spines['right'].set_visible(False)

                # Create legend patches
                legend_patches = [mpatches.Patch(color=label_color_map[cat], label=f"Cluster {cat}") for cat in
                                  unique_labels]

                # Place legends on the right side of each subplot
                ax_gt.legend(handles=legend_patches, title='Clusters', loc='upper left', bbox_to_anchor=(1.05, 1),
                             fontsize=10)
                ax_pred.legend(handles=legend_patches, title='Clusters', loc='upper left', bbox_to_anchor=(1.05, 1),
                               fontsize=10)

            # Save the plot
            plot_path = self.plot_dir / f"{sample_name}_cluster_{target_clusters}.pdf"
            plt.savefig(plot_path, format="pdf", bbox_inches="tight")
            print(f"Visualization plot saved to: {plot_path}")
            plt.show()
            plt.close(fig)  # Close the figure to free up memory

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred during clustering and visualization: {e}")


def test_visium_pipline(args):
    # --- 1. Set up paths and parameters ---
    # TODO: Please provide the correct paths for your system.

    # Path to the directory containing your raw Visium data
    # (e.g., folders from Space Ranger or your own .h5ad files)
    DATA_BASE=args.data_root + f"/Independent_test/{args.test_tissue}"
    INPUT_DATA_DIR = DATA_BASE + '/RAW'

    # Path to the directory where all processed files will be saved.
    OUTPUT_DIR = DATA_BASE + '/Input'

    # Path to the directory containing the .npy output files from your model.
    RESULT_BASE = f'./result/{args.tissue}/{args.model_name}/Hyper_independent_test_{args.test_tissue}_{args.gene_num}'

    ANALYSIS_DIR=RESULT_BASE + '/Analysis'
    MODEL_OUTPUT_DIR = RESULT_BASE + '/Output'

    # TODO: Define the list of samples to process.
    # The names should match the directory names or .h5ad file names (without extension)
    # in your INPUT_DATA_DIR.
    sample_list = ['Section_1_P']

    # TODO: Define the target number of clusters for visualization.
    target_cluster_count = 8

    # --- 2. Initialize and run the pipeline ---
    # Make sure the specified directories exist before running the pipeline.
    Path(INPUT_DATA_DIR).mkdir(parents=True, exist_ok=True)
    # Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    pipeline = VisiumPipeline(args, input_dir=INPUT_DATA_DIR, output_dir=OUTPUT_DIR, analysis_dir=ANALYSIS_DIR)

    # --- 3. Execute each step of the pipeline for each sample ---
    for sample in sample_list:
        try:
            # Step 1: From raw data to model input
            pipeline.raw_to_input(sample_name=sample)

            # Step 2: Testing model.
            test_hyper(args)

            # Step 2: From model output to enhanced h5ad
            pipeline.output_to_h5ad(
                sample_name=sample,
                model_output_dir=MODEL_OUTPUT_DIR,
                cutoff=1.3  # This threshold can be adjusted
            )

            # Step 4: Clustering and visualization
            pipeline.cluster_and_visualize(
                sample_name=sample,
                target_clusters=target_cluster_count,
                labels=['leiden', 'louvain']  # You can choose just one, e.g., ['leiden']
            )
        except FileNotFoundError as fnf_error:
            print(f"\nSkipping sample '{sample}' due to a missing file: {fnf_error}")
        except Exception as e:
            print(f"\nAn unexpected error occurred while processing sample '{sample}': {e}")