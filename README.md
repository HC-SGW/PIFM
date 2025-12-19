# Prior Informed Flow Matching (PIFM) for graph reconstruction

This repository contains the implementation of PIFM on tasks of link prediction, expansion, and denoising. It contains the flow models, minimal GDSS dependencies, and Node2Vec utilities in a form that is easy to move between projects.

PIFM was built under Python 3.7.16.

## Contents

| Path | Purpose |
| ---- | ------- |
| `main.py` | Unified CLI that dispatches to the expansion (inpainting) and denoise (fake-edge removal) workflows. |
| `baselines/train_node2vec_baseline.py` | Baseline Node2Vec + link predictor to pre-compute priors for the expansion workflow and as a baseline. |
| `utils/denoiser.py` | Flow architecture used by both workflows. |
| `GDSS/` | Minimal GDSS modules required by the denoiser. |
| `requirements.txt` | Python 3.7.16 dependencies. |

By default the Python modules create `utils/outputs/` (and the Node2Vec baseline writes to `baselines/outputs/`) to store checkpoints, reconstructions, and loss curves. Export `PIFM_OUTPUT_ROOT` to redirect all outputs elsewhere. Both default folders are ignored by git already.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # use .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
# install a compatible PyTorch wheel if needed
```

## Data Expectations

All trainers expect pickled lists of graphs and masks that follows the following structure:

```
data/
  train_graphs.pkl         # list[nx.Graph] or adjacency tensors
  val_graphs.pkl
  test_graphs.pkl
  masks_drop0.2/           # drop-specific masks for expansion runs (name matches --drop_prob)
    train_masks.pkl        # mask==1 means observed/kept edge
    val_masks.pkl
    test_masks.pkl
  train_fake_edge_masks.pkl     # for denoise workflow (1 marks injected fake edge)
  ...
```

You can adapt the loaders in `main.py` if your storage format differs, but the default expansion workflow expects the `masks_drop{drop_prob}` convention shown above.

## Preparing Expansion Priors

Generate Node2Vec-based priors before running the expansion workflow:

```bash
python baselines/train_node2vec_baseline.py \
  --name demo \
  --drop_rate 0.2 \
  --train_graphs data/train_graphs.pkl \
  --train_masks data/train_masks.pkl \
  --val_graphs data/val_graphs.pkl \
  --val_masks data/val_masks.pkl \
  --output_dir outputs/n2v_demo
```

The script writes reconstructions to `outputs/n2v_demo/…/graphs_npy/`. Point the expansion commands at those folders via `--prior_*_dir`.

## Running with `main.py`

`main.py` provides a CLI with multiple subcommands. Pick the subcommand name and pass the matching arguments.

### Expansion workflow 

```bash
python main.py train_expansion \
  --name demo_expansion \
  --train_pkl data/train_graphs.pkl \
  --val_pkl data/val_graphs.pkl \
  --prior_train_dir outputs/n2v_demo/train/graphs_npy/raw \
  --prior_val_dir outputs/n2v_demo/val/graphs_npy/raw \
  --drop_prob 0.2 \
  --epochs 100 \
  --batch_size 2
```

```bash
python main.py sample_expansion \
  --name demo_expansion \
  --sample_pkl data/test_graphs.pkl \
  --mask_pkl data/test_masks.pkl \
  --prior_test_dir outputs/n2v_demo/test/graphs_npy/raw \
  --ckpt outputs/models/latest_time_n2v \
  --n_steps 1000 \
  --save_plots
```

### Subgraph diffusion workflow (recommended for LP)

Use the subgraph diffusion pipeline when you have a single large graph and want link prediction with edge-centered subgraphs. Training uses `train_expansion` and inference uses `sample_expansion`, both with `--subgraph_lp`.

Recommended scripts (Cora examples, uniform subsampling + edge-centered) live under `scripts/`:

- `scripts/run_cora_LP_n2v_edgeCentered_full.sh` (Node2Vec prior, k=3)
- `scripts/run_cora_LP_sageheart_edgeCenter_k1.sh` (GraphSAGE-HEART prior, k=1)
- `scripts/run_cora_LP_sageheart_edgeCenter_k4_large.sh` (GraphSAGE-HEART prior, k=4)

These scripts all use `cfg/dataset_unif.yaml` with `sampling_method: uniform`. In practice, the best results come from:

- `--subgraph_prior node2vec` or `--subgraph_prior graphsage_heart`
- `sampling_method: uniform` in the dataset config

Inference method choice:

- `sample_expansion --subgraph_lp --test_edge_centered_subgraphs` uses edge-centered k-hop test subgraphs.
- If you omit `--test_edge_centered_subgraphs`, test sampling follows the dataset config (SaGress-style sampling).

Key tunable parameters for subgraph diffusion:

- Subgraph sampling: `--k_hop`, `--max_nodes`, `--target_coverage`, `--train_edge_drop_p`, `--lap_pe_dim`, `--split_seed`, `--val_ratio`, `--test_ratio`, `--node_select_graph`
- Dataset config (uniform sampling): `sampling_method`, `subgraph_size`, `per_node_samples_unif`
- Model/training: `--epochs`, `--batch_size`, `--hidden_dim`, `--num_layers`, `--num_linears`, `--c_init`, `--c_hid`, `--c_final`, `--lr`, `--seed`
- Inference: `--n_steps`

Prior-specific knobs (recommended):

- Node2Vec: `--subgraph_n2v_dim`, `--subgraph_n2v_walk_length`, `--subgraph_n2v_walks_per_node`, `--subgraph_n2v_context_size`, `--subgraph_n2v_epochs`, `--subgraph_clf_epochs`, `--subgraph_n2v_batch_size`, `--subgraph_neg_ratio`, `--subgraph_clf_lr`, `--subgraph_n2v_device`
- GraphSAGE-HEART: `--graphsage_heart_dim`, `--graphsage_heart_hidden_dim`, `--graphsage_heart_layers`, `--graphsage_heart_epochs`, `--graphsage_heart_lr`, `--graphsage_heart_neg_ratio`, `--graphsage_heart_dropout`, `--subgraph_sage_heart_weight_decay`, `--graphsage_heart_device`

### Denoise workflow

By default the denoise workflow now uses a Gaussian prior (no external files required).  
```bash
python main.py train_denoise \
  --name demo_denoise \
  --train_pkl data/train_graphs.pkl \
  --val_pkl data/val_graphs.pkl \
  --train_fake_edge_mask_pkl data/train_fake_edge_masks.pkl \
  --val_fake_edge_mask_pkl data/val_fake_edge_masks.pkl \
  --epochs 100 \
  --batch_size 2
```

```bash
python main.py sample_denoise \
  --name demo_denoise \
  --sample_pkl data/test_graphs.pkl \
  --fake_edge_mask_pkl data/test_fake_edge_masks.pkl \
  --ckpt outputs/models/latest_time_n2v_fake \
  --n_steps 500
```

To reuse Node2Vec priors, set `--fake_prior_init prior` and provide the corresponding directories.

```bash
python main.py train_denoise \
  --name demo_denoise \
  --train_pkl data/train_graphs.pkl \
  --val_pkl data/val_graphs.pkl \
  --train_fake_edge_mask_pkl data/train_fake_edge_masks.pkl \
  --val_fake_edge_mask_pkl data/val_fake_edge_masks.pkl \
  --fake_prior_init prior \
  --fake_prior_train_dir priors/fake_train \
  --fake_prior_val_dir priors/fake_val \
  --epochs 100 \
  --batch_size 2
```

```bash
python main.py sample_denoise \
  --name demo_denoise \
  --sample_pkl data/test_graphs.pkl \
  --fake_edge_mask_pkl data/test_fake_edge_masks.pkl \
  --fake_prior_init prior \
  --fake_prior_test_dir priors/fake_test \
  --ckpt outputs/models/latest_time_n2v_fake \
  --n_steps 500
```

### Additional CLI flags

| Flag | Applies to | Description |
| ---- | ---------- | ----------- |
| `--prior_init {prior,gaussian,baseline}` | expansion | Choose external priors, symmetric Gaussian fill, or baseline (copy `A_obs`). |
| `--fake_prior_init {prior,gaussian}` | denoise | Same idea but for fake-edge training/sampling. |
| `--save_plots / --no_save_plots` | sampling modes | Toggle per-graph PNG artefacts. Defaults to saving plots. |
| `--traj_plot` | expansion sampling | Emit trajectory panels for a subset of graphs (`--traj_max_samples`). |
| `PIFM_OUTPUT_ROOT` | env var | Redirects all outputs (models, loss curves, reconstructions) to a different folder. |


## Outputs

By default the diffusion CLI writes under `utils/outputs/`:

```
utils/outputs/
  models/                # checkpoints (+ latest symlinks)
  loss_curve/            # PNGs per training run
  output_inter/
    MMSE_raw/            # expansion sampling runs
    MMSE_fake/           # denoise sampling runs
    difference_inter/    # CSV summaries and last-row aggregates
```

The Node2Vec baseline stores its cached priors under `baselines/outputs/output_inter/n2vdiff_priors/…` by default, mirroring the same `PIFM_OUTPUT_ROOT` override.

If you prefer a different location, export `PIFM_OUTPUT_ROOT=/path/to/storage` before running the commands.

## Development Tips

- Install optional dependencies (`pot`, `torch-scatter`, etc.) before enabling MMD metrics.
- Use `--n_steps` judiciously: larger values can impact the sampling results in many different ways.
- When running in Gaussian mode, priors do not need to be loaded from disk: the samplers will synthesise the A₀ from noise.

## Attribution

Portions of this repository are adapted from [Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations (GDSS)](https://github.com/harryjo97/GDSS), which accompanies the ICML 2022 paper by Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. In particular:

- `GDSS/` contains modules copied or lightly modified from the original GDSS implementation.
- `utils/denoiser.py` derives from the GDSS denoiser and has been adapted to integrate with PIFM workflows.

If you build on these components, please cite the GDSS work:

```bibtex
@article{jo2022GDSS,
  author    = {Jaehyeong Jo and
               Seul Lee and
               Sung Ju Hwang},
  title     = {Score-based Generative Modeling of Graphs via the System of Stochastic
               Differential Equations},
  journal   = {arXiv:2202.02514},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.02514}
}
```

Please refer to the upstream GDSS repository for license terms governing those files.
