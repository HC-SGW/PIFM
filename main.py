# Flexible prior (external or Gaussian) + time-embedded flow

import os
import time
import argparse
import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

from utils.paths import (
    DIFFERENCE_DIR,
    KGRID_RAW_DIR,
    LOSS_CURVE_DIR,
    MMSE_FAKE_DIR,
    MMSE_RAW_DIR,
    MODELS_DIR,
)
from utils.tensor_utils import (
    add_masked_symmetric_noise,
    collate_graphs,
    invert_perm,
    linear_coeffs,
    permute_square,
    set_seed,
    sym_zero_diag_valid,
    zero_diag_,
)
from utils.initialization import (
    build_initial_A0,
    build_initial_A0_lp,
    init_A0_gaussian_masked,
    init_gaussian_on_omega,
)
from utils.priors import load_priors_from_npy_dir
from utils.evaluation import (
    aggregate_last_rows_for_run,
    compute_consensus,
    compute_metric_row_single,
    evaluate_and_save_real,
    make_pdf_from_dir_with_metric_rows,
    masked_upper_mse,
    posterior_eval_on_val_samples,
    save_five_panel,
)
from utils.denoiser import *
from main_sub import train_subgraph_lp, infer_subgraph_lp


# -------------------------
# Train Expansion
# -------------------------
def train_expansion(args, masked_only: bool = False):
    """Training entrypoint supporting external priors or Gaussian initialization.
    When `masked_only` is True, updates, noise, and losses are restricted to masked edges.
    """
    mode = getattr(args, 'prior_init', 'prior')
    scope = "masked-only" if masked_only else "union"
    print(f"======= Starting training (prior_init={mode}, region={scope}) =======")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    print("📥 Loading pre-split graph datasets and masks...")
    train_graphs = pickle.load(open(args.train_pkl, 'rb'))
    val_graphs = pickle.load(open(args.val_pkl, 'rb'))

    mask_dir = os.path.join(os.path.dirname(args.train_pkl), f"masks_drop{args.drop_prob}")
    train_masks_np = pickle.load(open(os.path.join(mask_dir, 'train_masks.pkl'), 'rb'))
    val_masks_np = pickle.load(open(os.path.join(mask_dir, 'val_masks.pkl'), 'rb'))

    train_masks = [torch.from_numpy(m).float() for m in train_masks_np]
    val_masks = [torch.from_numpy(m).float() for m in val_masks_np]

    use_external_prior = mode == 'prior'
    train_priors = train_z1d = val_priors = val_z1d = None
    if use_external_prior:
        train_dir = getattr(args, 'prior_train_dir', None) or getattr(args, 'n2v_prior_train_dir', None)
        val_dir = getattr(args, 'prior_val_dir', None) or getattr(args, 'n2v_prior_val_dir', None)
        if not train_dir or not val_dir:
            raise ValueError('Training with external priors requires --prior_train_dir and --prior_val_dir.')
        print(f"⌛ Loading priors from:\n  train → {train_dir}\n  val   → {val_dir}")
        train_priors, train_z1d = load_priors_from_npy_dir(train_dir, train_masks, args)
        val_priors, val_z1d = load_priors_from_npy_dir(val_dir, val_masks, args)

    class GraphMaskDataset(Dataset):
        def __init__(self, graphs, masks, priors=None, z1d=None):
            self.graphs = graphs
            self.masks = masks
            self.priors = priors
            self.z1d = z1d
        def __len__(self):
            return len(self.graphs)
        def __getitem__(self, i):
            G = self.graphs[i]
            A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32) if isinstance(G, nx.Graph) else G.float()

            prior = None
            if self.priors is not None:
                val = self.priors[i]
                if val is not None:
                    prior = val.float() if torch.is_tensor(val) else torch.tensor(val, dtype=torch.float32)

            z_feat = None
            if self.z1d is not None:
                z_val = self.z1d[i]
                if z_val is not None:
                    z_feat = z_val.float() if torch.is_tensor(z_val) else torch.tensor(z_val, dtype=torch.float32)

            return A, self.masks[i].float(), prior, z_feat

    def collate_fn(batch):
        As, Ms, Ys, Zs = zip(*batch)
        A_batch, node_mask = collate_graphs(As)
        max_n = A_batch.size(1)
        B = len(Ms)
        M_padded = torch.zeros(B, max_n, max_n, dtype=torch.float32)
        for i, M in enumerate(Ms):
            n = M.size(0)
            M_padded[i, :n, :n] = M
        if any(Y is not None for Y in Ys):
            Y_padded = torch.zeros(B, max_n, max_n, dtype=torch.float32)
            for i, Y in enumerate(Ys):
                if Y is None:
                    continue
                n = Y.size(0)
                Y_padded[i, :n, :n] = Y
        else:
            Y_padded = None
        if any(Z is not None for Z in Zs):
            Z_padded = torch.zeros(B, max_n, dtype=torch.float32)
            for i, Z in enumerate(Zs):
                if Z is None:
                    continue
                n = Z.size(0)
                Z_padded[i, :n] = Z
        else:
            Z_padded = None
        return A_batch, node_mask, M_padded, Y_padded, Z_padded

    train_loader = DataLoader(
        GraphMaskDataset(train_graphs, train_masks, train_priors, train_z1d),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        GraphMaskDataset(val_graphs, val_masks, val_priors, val_z1d),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    def _size_of(g):
        if isinstance(g, torch.Tensor):
            return g.size(0)
        if isinstance(g, nx.Graph):
            return g.number_of_nodes()
        raise TypeError('Unsupported graph type')

    Nmax = max(_size_of(g) for g in train_graphs)
    denoiser = DenoiseNetworkA(
        max_feat_num=1,
        max_node_num=Nmax,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,
    ).to(device)
    optimizer = optim.Adam(denoiser.parameters(), lr=args.lr)

    os.makedirs(MODELS_DIR, exist_ok=True)
    run_ts = time.strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join(MODELS_DIR, f"time_n2v_{args.name}_{args.drop_prob}drop_{run_ts}")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(MODELS_DIR, 'latest_time_n2v')
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(ckpt_dir, latest_path)
    except OSError:
        pass

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        train_sum_per_graphs = 0.0
        train_graphs_seen = 0

        for A_batch, node_mask, edge_mask, Y_prior, z1d in train_loader:
            A_batch = A_batch.to(device)
            node_mask = node_mask.to(device)
            edge_mask = edge_mask.to(device)
            Y_prior = Y_prior.to(device) if Y_prior is not None else None
            z1d = z1d.to(device) if z1d is not None else None
            B, N, _ = A_batch.size()

            if use_external_prior:
                if Y_prior is None or z1d is None:
                    raise RuntimeError('Missing priors/z-coordinates in prior mode.')
                for i in range(B):
                    p = torch.argsort(z1d[i], dim=0)
                    z1d[i] = z1d[i].index_select(0, p)
                    node_mask[i] = node_mask[i].index_select(0, p)
                    A_batch[i] = permute_square(A_batch[i], p)
                    edge_mask[i] = permute_square(edge_mask[i], p)
                    Y_prior[i] = permute_square(Y_prior[i], p)

            edge_mask = sym_zero_diag_valid(edge_mask, node_mask)
            A_obs = sym_zero_diag_valid(A_batch * edge_mask, node_mask)
            if masked_only:
                update_mask = sym_zero_diag_valid(1.0 - edge_mask, node_mask)
                loss_mask = update_mask
                noise_mask = edge_mask
                A0_clean = build_initial_A0_lp(
                    args,
                    A_true=A_batch,
                    edge_mask=edge_mask,
                    node_mask=node_mask,
                    prior=Y_prior,
                )
            else:
                update_mask = sym_zero_diag_valid(1.0 - A_obs, node_mask)
                loss_mask = A_obs
                noise_mask = A_obs
                A0_clean = build_initial_A0(
                    args,
                    A_obs=A_obs,
                    node_mask=node_mask,
                    prior=Y_prior,
                    noise_std=0.0,
                )
            A0 = A0_clean.clone()
            if args.train_noise_std > 0:
                A0 = add_masked_symmetric_noise(
                    M=A0,
                    node_mask=node_mask,
                    edge_mask=noise_mask,
                    sigma=args.train_noise_std,
                    clip01=True,
                )

            t = torch.rand(B, device=device)
            alpha, beta, alpha_dot, beta_dot = linear_coeffs(t)
            av, bv = alpha.view(B, 1, 1), beta.view(B, 1, 1)
            I_t = sym_zero_diag_valid(av * A0 + bv * A_batch, node_mask)

            inp = I_t.unsqueeze(1)
            x_feat = torch.zeros(B, N, 1, device=device)
            b_pred = denoiser(x_feat, inp, node_mask, t)
            b_pred = sym_zero_diag_valid(b_pred, node_mask)
            b_pred = b_pred * update_mask

            target = sym_zero_diag_valid(A_batch - A0, node_mask)
            loss = masked_upper_mse(b_pred, target, node_mask, loss_mask, reduction="per_graph")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_sum_per_graphs += float(loss.item()) * B
            train_graphs_seen += B

        train_loss = train_sum_per_graphs / max(1, train_graphs_seen)
        train_losses.append(train_loss)

        denoiser.eval()
        val_sum_per_graphs = 0.0
        val_graphs_seen = 0
        with torch.no_grad():
            for A_batch, node_mask, edge_mask, Y_prior, z1d in val_loader:
                A_batch = A_batch.to(device)
                node_mask = node_mask.to(device)
                edge_mask = edge_mask.to(device)
                Y_prior = Y_prior.to(device) if Y_prior is not None else None
                z1d = z1d.to(device) if z1d is not None else None
                B, N, _ = A_batch.size()

                if use_external_prior:
                    if Y_prior is None or z1d is None:
                        raise RuntimeError('Missing priors/z-coordinates in prior mode.')
                    for i in range(B):
                        p = torch.argsort(z1d[i], dim=0)
                        z1d[i] = z1d[i].index_select(0, p)
                        node_mask[i] = node_mask[i].index_select(0, p)
                        A_batch[i] = permute_square(A_batch[i], p)
                        edge_mask[i] = permute_square(edge_mask[i], p)
                        Y_prior[i] = permute_square(Y_prior[i], p)

                edge_mask = sym_zero_diag_valid(edge_mask, node_mask)
                A_obs = sym_zero_diag_valid(A_batch * edge_mask, node_mask)
                if masked_only:
                    update_mask = sym_zero_diag_valid(1.0 - edge_mask, node_mask)
                    loss_mask = update_mask
                    noise_mask = edge_mask
                    A0_clean = build_initial_A0_lp(
                        args,
                        A_true=A_batch,
                        edge_mask=edge_mask,
                        node_mask=node_mask,
                        prior=Y_prior,
                    )
                else:
                    update_mask = sym_zero_diag_valid(1.0 - A_obs, node_mask)
                    loss_mask = A_obs
                    noise_mask = A_obs
                    A0_clean = build_initial_A0(
                        args,
                        A_obs=A_obs,
                        node_mask=node_mask,
                        prior=Y_prior,
                        noise_std=0.0,
                    )
                A0 = A0_clean.clone()
                if args.val_noise_std > 0:
                    A0 = add_masked_symmetric_noise(
                        M=A0,
                        node_mask=node_mask,
                        edge_mask=noise_mask,
                        sigma=args.val_noise_std,
                        clip01=True,
                    )

                t = torch.rand(B, device=device)
                alpha, beta, alpha_dot, beta_dot = linear_coeffs(t)
                av, bv = alpha.view(B, 1, 1), beta.view(B, 1, 1)
                I_t = sym_zero_diag_valid(av * A0 + bv * A_batch, node_mask)

                inp = I_t.unsqueeze(1)
                x_feat = torch.zeros(B, N, 1, device=device)
                b_pred = denoiser(x_feat, inp, node_mask, t)
                b_pred = sym_zero_diag_valid(b_pred, node_mask)
                b_pred = b_pred * update_mask

                target = sym_zero_diag_valid(A_batch - A0, node_mask)
                l = masked_upper_mse(b_pred, target, node_mask, loss_mask, reduction="per_graph")
                val_sum_per_graphs += float(l.item()) * B
                val_graphs_seen += B

        val_loss = val_sum_per_graphs / max(1, val_graphs_seen)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        if args.val_posterior_every > 0 and (epoch % args.val_posterior_every == 0):
            posterior_eval_on_val_samples(epoch, val_loader, denoiser, device, args, masked_only=masked_only)

        if (epoch % args.ckpt_every == 0) or (epoch == args.epochs):
            ckpt_name = f"ep{epoch:04d}.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(denoiser.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved @ epoch {epoch}: {ckpt_path}")

    os.makedirs(LOSS_CURVE_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title(f"Training and Validation Loss over Epochs (init={mode})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ts = time.strftime('%Y%m%d_%H%M%S')
    loss_plot_path = os.path.join(LOSS_CURVE_DIR, f"loss_curve_{mode}_{ts}.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"📉 Loss curve saved to: {loss_plot_path}")

# -------------------------
# Train Link Prediction
# -------------------------
def train_LP(args):
    """Wrapper for link-prediction training on masked edges only."""
    return train_expansion(args, masked_only=True)

# -------------------------
# Sample Expansion
# -------------------------
def sample_expansion(args, masked_only: bool = False):
    """
    When `masked_only` is True, restrict updates/noise to masked edges only.
    """
    mode = getattr(args, "prior_init", "prior")
    base_dir = MMSE_RAW_DIR
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{args.name}_{args.drop_prob}drop_{ts}")
    rounded_dir     = os.path.join(run_dir, "A0_rounded")
    rounded_raw_dir = os.path.join(run_dir, "A0_raw")
    recon_raw_dir   = os.path.join(run_dir, "recon_raw")

    os.makedirs(recon_raw_dir, exist_ok=True)
    os.makedirs(rounded_dir, exist_ok=True)
    os.makedirs(rounded_raw_dir, exist_ok=True)

    plot_dir = os.path.join(run_dir, "plots") if getattr(args, "save_plots", True) else None
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    all_metric_rows = []
    
    prefix = os.path.basename(run_dir)
    flow_label = "Sample LP" if masked_only else "Sample Expansion"
    region_scope = "masked-only" if masked_only else "union"
    run_tag = prefix

    print(f"[Run] Starting {flow_label} (prior_init={mode}, scope={region_scope}) → {run_tag}: {run_dir}", flush=True)
    print("[Device] Initializing device...", flush=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"[Device] CUDA available: {cuda_available} | Using device: {device}", flush=True)
    if cuda_available:
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Build A1_list and masks from either pkl lists or single npy
    if args.sample_pkl and args.mask_pkl:
        with open(args.sample_pkl, 'rb') as f:
            sample_graphs = pickle.load(f)
        A1_list = [
            torch.tensor(nx.to_numpy_array(g), dtype=torch.float32, device=device)
            if isinstance(g, nx.Graph) else g.to(dtype=torch.float32, device=device)
            for g in sample_graphs
        ]
        with open(args.mask_pkl, 'rb') as f:
            sample_masks = pickle.load(f)
        M_list = [torch.from_numpy(m).to(device).float() for m in sample_masks]
        print(f"→ Loaded {len(A1_list)} graphs and masks for external sampling.")
    elif args.input_graph and args.mask_npy:
        A = np.load(args.input_graph).astype(np.float32)
        M = np.load(args.mask_npy).astype(np.float32)
        A1_list = [torch.from_numpy(A).to(device).float()]
        M_list  = [torch.from_numpy(M).to(device).float()]
        print("→ Loaded single input graph and mask for sampling.")
    else:
        raise ValueError("Provide (--sample_pkl AND --mask_pkl) or (--input_graph AND --mask_npy).")

    print(f"[IO] Loaded {len(A1_list)} graph(s) and {len(M_list)} mask matrix/matrices.", flush=True)

    use_external_prior = mode == 'prior'

    graphs_cpu = [a.detach().cpu() for a in A1_list]
    masks_cpu  = [m.detach().cpu() for m in M_list]

    if use_external_prior:
        prior_dir = getattr(args, 'prior_test_dir', None) or getattr(args, 'n2v_prior_test_dir', None)
        if not prior_dir:
            raise ValueError("Sampling with external priors requires --prior_test_dir.")
        print(f"⌛ Loading priors for sampling from: {prior_dir}")
        test_priors, test_z1d = load_priors_from_npy_dir(prior_dir, masks_cpu, args)
        print(f"[priors] Loaded {len(test_priors)} prior matrices for sampling.")
    else:
        test_priors = [None] * len(graphs_cpu)
        test_z1d = [None] * len(graphs_cpu)

    # Denoiser
    denoiser = DenoiseNetworkA(
        max_feat_num=1,
        max_node_num=args.max_graph_nodes,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,  
        # num_heads=args.num_heads, # optional (default 4)
        # conv=args.conv, # optional (default 'GCN')
    ).to(device)
    denoiser.load_state_dict(torch.load(args.ckpt, map_location=device))
    denoiser.eval()
    param_count = sum(p.numel() for p in denoiser.parameters())
    print(f"[Model] Denoiser loaded from {args.ckpt} ({param_count/1e6:.2f}M params).", flush=True)
    if cuda_available:
        assert next(denoiser.parameters()).is_cuda, "Model not on CUDA!"
    
    # Diffusion step grid (override with --sample_nsteps)
    n_steps_values = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100]
    custom_steps = getattr(args, 'sample_nsteps', '')
    if custom_steps:
        try:
            n_steps_values = [int(s) for s in custom_steps.split(',') if s.strip()]
        except ValueError as exc:
            raise ValueError('--sample_nsteps must be comma-separated integers') from exc
    if not n_steps_values:
        raise ValueError('No valid n_steps provided; supply at least one integer')
    print(f"[Schedule] n_steps grid: {n_steps_values}", flush=True)
    print(f"🚀 Sampling with n_steps grid: {n_steps_values}")

    final_recons_by_steps = {n: [] for n in n_steps_values}
    true_graphs = []
    plot_paths = []

    # evaluator masks:
    edge_mask_list = [] # evaluator scores on masked region (M==0)
    aobs_mask_list = [] # evaluator scores on zeros of A_obs (union)
    atrue_mask_list = [] # evaluator scores on true-zero region

    initial_raws = []
    eval_csv_paths = []

    print(f"[Loop] Sampling configuration: prior_init={mode}, scope={region_scope}", flush=True)
    print(f"[Loop] Beginning per-graph sampling over {len(A1_list)} graph(s).", flush=True)
    
    for i, (A1, edge_mask) in enumerate(zip(A1_list, M_list)):
        print(f"[Graph {i+1}/{len(A1_list)}] start (N={A1.size(0)})", flush=True)
        node_mask = torch.ones(A1.size(0), dtype=torch.bool, device=device)
        Y_prior = test_priors[i].to(device).float() if test_priors[i] is not None else None
        if test_z1d[i] is not None:
            z_full = test_z1d[i].to(device).float()
        else:
            z_full = torch.linspace(0.0, 1.0, A1.size(0), device=device, dtype=A1.dtype)
        # Align by z_full (monotone coordinate)
        p     = torch.argsort(z_full, dim=0)
        p_inv = invert_perm(p)

        z_full   = z_full.index_select(0, p)
        A1_perm  = permute_square(A1,       p)
        edge_mask_perm = permute_square(edge_mask.float(), p)
        node_mask = node_mask.index_select(0, p)
        Y_prior_perm  = permute_square(Y_prior,  p) if Y_prior is not None else None

        edge_mask_perm = sym_zero_diag_valid(edge_mask_perm, node_mask)
        A_obs = sym_zero_diag_valid(A1_perm * edge_mask_perm, node_mask)
        if masked_only:
            Omega = sym_zero_diag_valid(1.0 - edge_mask_perm, node_mask)
            noise_mask = edge_mask_perm
            A0_clean = build_initial_A0_lp(
                args,
                A_true=A1_perm,
                edge_mask=edge_mask_perm,
                node_mask=node_mask,
                prior=Y_prior_perm,
            )
        else:
            Omega = sym_zero_diag_valid(1.0 - A_obs, node_mask)
            noise_mask = A_obs
            A0_clean = build_initial_A0(
                args,
                A_obs=A_obs,
                node_mask=node_mask,
                prior=Y_prior_perm,
                noise_std=0.0,
            )
        A_anchor = A_obs
        A0_noisy = A0_clean.clone()
        if args.noise_std > 0:
            A0_noisy = add_masked_symmetric_noise(
                M=A0_noisy, node_mask=node_mask, edge_mask=noise_mask,
                sigma=args.noise_std, clip01=True
            )
        print(f"\n--- Processing Sample {i}: Generated common A0 (clean + noisy) ---")

        # saving reconstructions
        A0_unperm_clean = permute_square(A0_clean, p_inv)
        A0_unperm_noisy = permute_square(A0_noisy, p_inv)

        np.save(os.path.join(rounded_raw_dir, f"{prefix}_sample{i}_A0raw.npy"),        A0_unperm_noisy.cpu().numpy()) 
        np.save(os.path.join(rounded_raw_dir, f"{prefix}_sample{i}_A0raw_clean.npy"),  A0_unperm_clean.cpu().numpy()) 

        A0_rounded_unperm = (A0_unperm_noisy > 0.5).float() 
        np.save(os.path.join(rounded_dir,     f"{prefix}_sample{i}_A0rounded.npy"), A0_rounded_unperm.cpu().numpy())
        initial_raws.append(A0_unperm_clean.cpu().clone())

        # sampling start from the same noisy A0
        A = A0_noisy.clone()
        A_obs_unperm = permute_square(A_obs, p_inv)
        aobs_mask_list.append(A_obs_unperm.cpu().clone())   # union zeros of A_obs
        atrue_mask_list.append(A1.cpu().clone())  
        row_A0 = compute_metric_row_single(
            A_true_t=A1,
            A_rec_t=A0_unperm_clean, # report the CLEAN A0 in CSV
            mask_t=edge_mask,
            plot_path="",
            score_mode="raw",
            variant="A0raw",
            sample_idx=i,
            n_steps=0
        )
        if row_A0: all_metric_rows.append(row_A0)
        true_graphs.append(A1.cpu().clone()) 
        edge_mask_list.append(edge_mask.cpu().clone()) 
        
        # optional trajectory graphs
        if args.save_plots and args.traj_plot and i < args.traj_max_samples:
            traj_k = args.traj_k or args.n_steps
            traj_dir  = os.path.join(run_dir, f"traj_k{traj_k}", f"sample{i:03d}")
            traj1_dir = os.path.join(run_dir, "traj_1step", f"sample{i:03d}")
            os.makedirs(traj_dir, exist_ok=True)
            os.makedirs(traj1_dir, exist_ok=True)

    
            save_five_panel(
                A_true=A1, edge_mask=edge_mask,
                A_step=A0_unperm_clean,
                outpath=os.path.join(traj_dir,  "step000_A0.png"),
                recon_title="Recon (t=0 / A0)"
            )
            save_five_panel(
                A_true=A1, edge_mask=edge_mask,
                A_step=A0_unperm_clean,
                outpath=os.path.join(traj1_dir, "step000_A0.png"),
                recon_title="Recon (t=0 / A0)"
            )
            
            
            row_A0_k = compute_metric_row_single(
                A_true_t=A1, A_rec_t=A0_unperm_clean, mask_t=edge_mask,
                plot_path=os.path.join(traj_dir, "step000_A0.png"),
                score_mode="raw", variant=f"traj_k{traj_k}_A0", sample_idx=i, n_steps=0
            )
            if row_A0_k: all_metric_rows.append(row_A0_k)

            row_A0_1 = compute_metric_row_single(
                A_true_t=A1, A_rec_t=A0_unperm_clean, mask_t=edge_mask,
                plot_path=os.path.join(traj1_dir, "step000_A0.png"),
                score_mode="raw", variant="traj_1step_A0", sample_idx=i, n_steps=0
            )
            if row_A0_1: all_metric_rows.append(row_A0_1)    

            # full k-step trajectory
            A = A0_noisy.clone()
            dt = 1.0 / traj_k
            x_feat = torch.zeros(1, z_full.shape[0], 1, device=device, dtype=z_full.dtype)

            for step in range(traj_k):
                inp = A.unsqueeze(0).unsqueeze(1)
                t_cur = torch.full((1,), step / float(traj_k), device=device)
                with torch.no_grad():
                    b = denoiser(x_feat, inp, node_mask.unsqueeze(0), t_cur).squeeze(0)
                    b = sym_zero_diag_valid(b, node_mask)
                b = b * Omega
                A = A + dt * b
                A.clamp_(0.0, 1.0)
                A = A_anchor + Omega * A
                A = sym_zero_diag_valid(A, node_mask)

                if (step + 1) % args.traj_every == 0 or step == traj_k - 1:
                    A_unperm = permute_square(A, p_inv)
                    panel_path = os.path.join(traj_dir, f"step{step+1:03d}.png")
                    save_five_panel(
                        A_true=A1, edge_mask=edge_mask,
                        A_step=A_unperm,
                        outpath=panel_path,
                        recon_title=f"Recon ({step+1}/{traj_k} steps)"
                    )
                    row = compute_metric_row_single(
                        A_true_t=A1, A_rec_t=A_unperm, mask_t=edge_mask,
                        plot_path=panel_path, score_mode="raw",
                        variant=f"traj_k{traj_k}_step{step+1:03d}", sample_idx=i, n_steps=step+1
                    )
                    if row: all_metric_rows.append(row)
                    
                
            # 1 step trajectory
            A1s = A0_noisy.clone()
            inp = A1s.unsqueeze(0).unsqueeze(1)
            t_one = torch.tensor([0.0], device=device)
            with torch.no_grad():
                b = denoiser(x_feat, inp, node_mask.unsqueeze(0), t_one).squeeze(0)
                b = sym_zero_diag_valid(b, node_mask)
            b = b * Omega
            A1s = A1s + 1.0 * b
            A1s = A_anchor + Omega * A1s
            A1s = sym_zero_diag_valid(A1s, node_mask)

            A1s_unperm = permute_square(A1s, p_inv)

            panel1 = os.path.join(traj1_dir, "step001_final.png")
            save_five_panel(A_true=A1, edge_mask=edge_mask, A_step=A1s_unperm, outpath=panel1, recon_title="Recon (1 step)")
            row_1final = compute_metric_row_single(
                A_true_t=A1, A_rec_t=A1s_unperm, mask_t=edge_mask,
                plot_path=panel1, score_mode="raw", variant="traj_1step_final", sample_idx=i, n_steps=1
            )
            if row_1final: all_metric_rows.append(row_1final)
        
            if args.make_pdf:
                traj_pdf_path  = os.path.join(traj_dir,  f"trajectory_k{traj_k}_sample{i:03d}.pdf")
                traj1_pdf_path = os.path.join(traj1_dir, f"trajectory_1step_sample{i:03d}.pdf")
                make_pdf_from_dir_with_metric_rows(traj_dir,  traj_pdf_path,  all_metric_rows)
                make_pdf_from_dir_with_metric_rows(traj1_dir, traj1_pdf_path, all_metric_rows)


        # looping over different n_step values
        A_final_for_plot = None 
        for current_n_steps in n_steps_values:
            # Start diffusion from the common A0
            A = A0_noisy.clone()
            dt = 1.0 / current_n_steps
            x_feat = torch.zeros(1, z_full.shape[0], 1, device=device, dtype=z_full.dtype)
            
            # Euler integration for the current number of steps
            for step in range(current_n_steps):
                inp = A.unsqueeze(0).unsqueeze(1)
                with torch.no_grad():
                    t = torch.full((1,), step * dt, device=device)
                    b = denoiser(x_feat, inp, node_mask.unsqueeze(0), t).squeeze(0)
                    b = sym_zero_diag_valid(b, node_mask)
                b = b * Omega

                A = A + dt * b
                A.clamp_(0.0, 1.0)
                A = A_anchor + Omega * A
                A = sym_zero_diag_valid(A, node_mask)

            # Store the final result for this run
            A_final_unperm = permute_square(A, p_inv)
            zero_diag_(A_final_unperm)
            final_recons_by_steps[current_n_steps].append(A_final_unperm.cpu().clone())
            print(f"  ✔️  Completed {current_n_steps}-step diffusion for sample {i}.")
            fname_final = f"{prefix}_sample{i}_{current_n_steps}steps_recon_raw.npy"
            out_final   = os.path.join(recon_raw_dir, fname_final)
            np.save(out_final, A_final_unperm.cpu().numpy())
            if current_n_steps == max(n_steps_values):
                 A_final_for_plot = A_final_unperm
            row_final = compute_metric_row_single(
                A_true_t=A1, A_rec_t=A_final_unperm, mask_t=edge_mask,
                plot_path="",  # no specific panel; leave empty
                score_mode="raw", variant=f"final_recon_{current_n_steps}steps",
                sample_idx=i, n_steps=current_n_steps
            )
            if row_final: all_metric_rows.append(row_final)
        reconstructed_A = (A_final_for_plot > 0.5).float()
        zero_diag_(reconstructed_A)
        diff = (reconstructed_A - A1).cpu() 

        if plot_dir:
            plot_path = os.path.join(plot_dir, f"{prefix}_sample{i}_plot.png")
            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
            axes[0].imshow(A1.cpu(), cmap='Greys');     axes[0].set_title("True Adjacency");     axes[0].axis("off")
            axes[1].imshow(edge_mask.cpu(), cmap='Greys'); axes[1].set_title("Edge Mask (kept)"); axes[1].axis("off")
            axes[2].imshow((A1 * edge_mask).cpu(), cmap='Greys'); axes[2].set_title("Masked Adjacency"); axes[2].axis("off")
            axes[3].imshow(reconstructed_A.cpu(), cmap='Greys'); axes[3].set_title(f"Recon ({max(n_steps_values)} steps)"); axes[3].axis("off")
            v = diff.abs().max().item() or 1e-6
            im = axes[4].imshow(diff.cpu(), cmap='bwr', vmin=-v, vmax=+v)
            axes[4].set_title("Raw Δ"); axes[4].axis("off")
            fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"✅ Plot saved to: {plot_path}")
            plot_paths.append(plot_path)
        else:
            plot_paths.append("")
        
        print(f"[Graph {i+1}] done (saved {len(n_steps_values)} recons)", flush=True)
        
    # evaluate the metrics
    include_extra_regions = not masked_only
    print("\n#############################################")
    print("📊 Evaluating final results for each n_steps value...")
    for n_steps, reconstructed_list in final_recons_by_steps.items():
        print(f"\n--- Evaluating for n_steps = {n_steps} ---")
        out = evaluate_and_save_real(
            args, true_graphs, reconstructed_list, edge_mask_list, plot_paths,
            st=f"final_recon_{n_steps}steps",
            score_mode="raw",
            compute_gw=True,
            gw_cost_mode="adj",
            gw_entropic=False,
            gw_epsilon=0.2,
            gw_max_iter=20000,
            gw_tol=1e-7,
            # MMD
            compute_mmd=True,
            mmd_kernel="rbf",
            mmd_sigma="median",     
            mmd_on="full_raw",
            mmd_max_samples=5000
        )

        if out:
            eval_csv_paths.append(out) 
        
        if include_extra_regions:
            # A_obs-zero union region (evaluate where A_obs==0)
            out_union = evaluate_and_save_real(
                args, true_graphs, reconstructed_list, aobs_mask_list, plot_paths,
                st=f"final_recon_{n_steps}steps_AobsZero",
                score_mode="raw",
                compute_gw=True, gw_cost_mode="adj",
                gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
                compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
                mmd_on="full_raw", mmd_max_samples=5000
            )
            if out_union: eval_csv_paths.append(out_union)

            # true-zero region (evaluate where A_true==0) 
            # AUC/AP may be NaN (no positives)
            out_truezero = evaluate_and_save_real(
                args, true_graphs, reconstructed_list, atrue_mask_list, plot_paths,
                st=f"final_recon_{n_steps}steps_trueZero",
                score_mode="raw",
                compute_gw=True, gw_cost_mode="adj",
                gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
                compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
                mmd_on="full_raw", mmd_max_samples=5000
            )
            if out_truezero: eval_csv_paths.append(out_truezero)


    print("\n#############################################")
    print("\n📊 Evaluating raw A0 (continuous) baseline on all regions…")

    # masked region (M==0)
    out_a0_masked = evaluate_and_save_real(
        args, true_graphs, initial_raws, edge_mask_list, plot_paths,
        st="A0raw",
        score_mode="raw",
        compute_gw=True, gw_cost_mode="adj",
        gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
        compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
        mmd_on="full_raw", mmd_max_samples=5000
    )
    if out_a0_masked: eval_csv_paths.append(out_a0_masked)

    if include_extra_regions:
        #A_obs-zero union region (A_obs == 0)
        out_a0_aobs = evaluate_and_save_real(
            args, true_graphs, initial_raws, aobs_mask_list, plot_paths,
            st="A0raw_AobsZero",
            score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )
        if out_a0_aobs: eval_csv_paths.append(out_a0_aobs)
        out_a0_tz = evaluate_and_save_real(
            args, true_graphs, initial_raws, atrue_mask_list, plot_paths,
            st="A0raw_trueZero",
            score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )
        if out_a0_tz: eval_csv_paths.append(out_a0_tz)

    master_paths = aggregate_last_rows_for_run(eval_csv_paths, args)
    if master_paths:
        master_paths = master_paths[-4:]

    for tmp in eval_csv_paths:
        try:
            os.remove(tmp)
        except OSError:
            pass

    # Print the aggregated CSV summaries
    if master_paths:
        print("\n=== Saved MASTER last-row summary CSVs ===")
        for p in master_paths:
            print(p)
    else:
        print("\n[ℹ️] No master CSV files were produced in this run.")

    print("\nAll sampling and evaluations complete.")

# -------------------------
# Sample Link Prediction
# -------------------------
def sample_LP(args):
    """Wrapper for link-prediction sampling restricted to masked edges."""
    return sample_expansion(args, masked_only=True)

# =========================
# Train denoising
# =========================
def train_denoise(args):
    """
    FAKE-EDGE training (opposite of link prediction):
      Region Ω := { (i,j) : A_obs[i,j] == 1 }, where A_obs = clip(A_true + R, [0,1]).
      We update ONLY on Ω (the 1s), zeros remain zeros.
      A0 := Ω ⊙ Prior  (+ noise only on Ω).
      Target drift on Ω: (A_true - A0).  Loss is masked to Ω via edge_mask=(1 - A_obs).
    """
    mode = getattr(args, 'fake_prior_init', 'prior')
    gauss_mean = getattr(args, 'fake_gauss_mean', 0.5)
    gauss_var = getattr(args, 'fake_gauss_var', 1.0)
    train_prior_dir = getattr(args, 'fake_prior_train_dir', None)
    val_prior_dir = getattr(args, 'fake_prior_val_dir', None)
    if mode == 'prior' and (not train_prior_dir or not val_prior_dir):
        print("⚠️  No fake prior directories provided; falling back to Gaussian initialization.")
        mode = 'gaussian'
        setattr(args, 'fake_prior_init', 'gaussian')

    print(f"======= FAKE-EDGE TRAIN (init={mode}) =======")
    print("• A_obs = clip(A_true + R, 0,1)")
    print("• Ω = ones(A_obs)  (update region = ones of observed graph)")
    if mode == 'prior':
        print("• A0 = Ω ⊙ Prior; Gaussian noise added ONLY on Ω")
    else:
        print(f"• A0 ~ N({gauss_mean}, {gauss_var}) on Ω; optional noise added on Ω")
    print("• Interpolant I_t = α(t)·A0 + β(t)·A_true; target drift = (A_true − A0)")
    print("• b_pred masked to Ω; loss computed on Ω via edge_mask = (1 − A_obs)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # load graphs
    print("📥 Loading pre-split graph datasets...")
    train_graphs = pickle.load(open(args.train_pkl, 'rb'))
    val_graphs   = pickle.load(open(args.val_pkl,   'rb'))
    print("📥 Loading fake-edge masks R...")
    train_R_np = pickle.load(open(args.train_fake_edge_mask_pkl, 'rb'))
    val_R_np   = pickle.load(open(args.val_fake_edge_mask_pkl,   'rb'))
    train_R = [torch.from_numpy(r).float() for r in train_R_np]
    val_R   = [torch.from_numpy(r).float() for r in val_R_np]

    if mode == 'prior':
        assert train_prior_dir and val_prior_dir, "Internal error: prior dirs missing despite prior mode."
        print(f"⌛ Loading priors:\n  train -> {train_prior_dir}\n  val   -> {val_prior_dir}")
        n2v_priors_train, n2v_z1d_train = load_priors_from_npy_dir(train_prior_dir, train_R, args)
        n2v_priors_val,   n2v_z1d_val   = load_priors_from_npy_dir(val_prior_dir,   val_R,   args)
    else:
        n2v_priors_train = [None] * len(train_R)
        n2v_z1d_train    = [None] * len(train_R)
        n2v_priors_val   = [None] * len(val_R)
        n2v_z1d_val      = [None] * len(val_R)

    # --- Dataset ---
    class GraphDenoiseDataset(Dataset):
        def __init__(self, graphs, R_list, priors=None, z1d=None):
            self.graphs = graphs
            self.R_list = R_list
            self.priors = priors
            self.z1d = z1d
        def __len__(self):
            return len(self.graphs)
        def __getitem__(self, i):
            G = self.graphs[i]
            A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32) if isinstance(G, nx.Graph) else G.float()
            prior = None
            if self.priors is not None:
                val = self.priors[i]
                if val is not None:
                    prior = val.float() if torch.is_tensor(val) else torch.tensor(val, dtype=torch.float32)
            z_feat = None
            if self.z1d is not None:
                z_val = self.z1d[i]
                if z_val is not None:
                    z_feat = z_val.float() if torch.is_tensor(z_val) else torch.tensor(z_val, dtype=torch.float32)
            return A, self.R_list[i].float(), prior, z_feat

    def collate_fn(batch):
        As, Rs, Ys, Zs = zip(*batch)
        A_batch, node_mask = collate_graphs(As)
        max_n = A_batch.size(1)
        B = len(Rs)
        R_pad = torch.zeros(B, max_n, max_n, dtype=torch.float32)
        for i, R in enumerate(Rs):
            n = R.size(0)
            R_pad[i, :n, :n] = R
        Y_pad = None
        if any(Y is not None for Y in Ys):
            Y_pad = torch.zeros(B, max_n, max_n, dtype=torch.float32)
            for i, Y in enumerate(Ys):
                if Y is None:
                    continue
                n = Y.size(0)
                Y_pad[i, :n, :n] = Y
        Z_pad = None
        if any(Z is not None for Z in Zs):
            Z_pad = torch.zeros(B, max_n, dtype=torch.float32)
            for i, Z in enumerate(Zs):
                if Z is None:
                    continue
                n = Z.size(0)
                Z_pad[i, :n] = Z
        return A_batch, node_mask, R_pad, Y_pad, Z_pad

    train_loader = DataLoader(
        GraphDenoiseDataset(train_graphs, train_R, n2v_priors_train, n2v_z1d_train),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        GraphDenoiseDataset(val_graphs, val_R, n2v_priors_val, n2v_z1d_val),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    def _size_of(g):
        if isinstance(g, torch.Tensor): return g.size(0)
        if isinstance(g, nx.Graph):     return g.number_of_nodes()
        raise TypeError("Unsupported graph type")
    Nmax = max([_size_of(g) for g in train_graphs])

    denoiser = DenoiseNetworkA(
        max_feat_num=1,
        max_node_num=Nmax,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,  
        # num_heads=args.num_heads,    # optional (default 4)
        # conv=args.conv,              # optional (default 'GCN')
    ).to(device)
    optimizer = optim.Adam(denoiser.parameters(), lr=args.lr)

    # Run dir
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(MODELS_DIR, f"time_n2v_fake_{args.name}_{args.flip_tag}_{run_ts}")
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        latest_path = os.path.join(MODELS_DIR, "latest_time_n2v_fake")
        if os.path.islink(latest_path) or os.path.exists(latest_path): os.remove(latest_path)
        os.symlink(ckpt_dir, latest_path)
    except OSError:
        pass
    def _sanitize_R_for_A(R, A, node_mask):
        R = torch.clamp(R, 0.0, 1.0)
        R = sym_zero_diag_valid(R, node_mask)
        R = R * (1.0 - (A > 0).float())
        return R

    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        sum_per_graphs, graphs_seen = 0.0, 0

        for A_batch, node_mask, R_batch, Y_prior, z1d in train_loader:
            A_batch   = A_batch.to(device)
            node_mask = node_mask.to(device)
            R_batch   = R_batch.to(device)
            Y_prior   = Y_prior.to(device) if Y_prior is not None else None
            z1d       = z1d.to(device) if z1d is not None else None
            B, N, _   = A_batch.size()

            for i in range(B):
                if mode == 'prior' and Y_prior is not None and z1d is not None:
                    p = torch.argsort(z1d[i], dim=0)
                    z1d[i] = z1d[i].index_select(0, p)
                else:
                    p = torch.arange(node_mask.size(1), device=device, dtype=torch.long)
                node_mask[i] = node_mask[i].index_select(0, p)
                A_batch[i]   = permute_square(A_batch[i], p)
                R_batch[i]   = permute_square(R_batch[i], p)
                if Y_prior is not None:
                    Y_prior[i] = permute_square(Y_prior[i], p)

            A0_list, Omega_list, Aobs_list = [], [], []
            for i in range(B):
                R_i   = _sanitize_R_for_A(R_batch[i], A_batch[i], node_mask[i])
                A_obs = sym_zero_diag_valid(torch.clamp(A_batch[i] + R_i, 0.0, 1.0), node_mask[i])
                Omega = A_obs
                if mode == 'prior' and Y_prior is not None:
                    A0_i = Omega * Y_prior[i]
                else:
                    A0_i = init_gaussian_on_omega(Omega, node_mask[i], gauss_mean, gauss_var)
                if args.train_noise_std > 0:
                    A0_i = add_masked_symmetric_noise(A0_i, node_mask[i], edge_mask=(Omega),
                                                       sigma=args.train_noise_std, clip01=True)
                A0_list.append(A0_i); Omega_list.append(Omega); Aobs_list.append(A_obs)

            A0          = torch.stack(A0_list,   dim=0)
            Omega_batch = torch.stack(Omega_list,dim=0)
            Aobs_batch  = torch.stack(Aobs_list, dim=0)

            t = torch.rand(B, device=device)
            alpha, beta, _, _ = linear_coeffs(t)
            I_t    = sym_zero_diag_valid(alpha.view(B,1,1) * A0 + beta.view(B,1,1) * A_batch, node_mask)
            target = sym_zero_diag_valid(A_batch - A0, node_mask)

            inp    = I_t.unsqueeze(1)
            x_feat = torch.zeros(B, N, 1, device=device)
            b_pred = denoiser(x_feat, inp, node_mask, t)
            b_pred = sym_zero_diag_valid(b_pred, node_mask)
            b_pred = b_pred * Omega_batch 

            loss = masked_upper_mse(
                b_pred,
                target,
                node_mask,
                loss_mask=(1.0 - Aobs_batch),
                reduction="per_graph",
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_per_graphs += float(loss.item()) * B
            graphs_seen    += B

        train_losses.append(sum_per_graphs / max(1, graphs_seen))

        # validation loop
        denoiser.eval()
        sum_per_graphs, graphs_seen = 0.0, 0
        with torch.no_grad():
            for A_batch, node_mask, R_batch, Y_prior, z1d in val_loader:
                A_batch   = A_batch.to(device)
                node_mask = node_mask.to(device)
                R_batch   = R_batch.to(device)
                Y_prior   = Y_prior.to(device) if Y_prior is not None else None
                z1d       = z1d.to(device) if z1d is not None else None
                B, N, _   = A_batch.size()

                for i in range(B):
                    if mode == 'prior' and Y_prior is not None and z1d is not None:
                        p = torch.argsort(z1d[i], dim=0)
                        z1d[i] = z1d[i].index_select(0, p)
                    else:
                        p = torch.arange(node_mask.size(1), device=device, dtype=torch.long)
                    node_mask[i] = node_mask[i].index_select(0, p)
                    A_batch[i]   = permute_square(A_batch[i], p)
                    R_batch[i]   = permute_square(R_batch[i], p)
                    if Y_prior is not None:
                        Y_prior[i] = permute_square(Y_prior[i], p)

                A0_list, Omega_list, Aobs_list = [], [], []
                for i in range(B):
                    R_i   = _sanitize_R_for_A(R_batch[i], A_batch[i], node_mask[i])
                    A_obs = sym_zero_diag_valid(torch.clamp(A_batch[i] + R_i, 0.0, 1.0), node_mask[i])
                    Omega = A_obs
                    if mode == 'prior' and Y_prior is not None:
                        A0_i = Omega * Y_prior[i]
                    else:
                        A0_i = init_gaussian_on_omega(Omega, node_mask[i], gauss_mean, gauss_var)
                    if args.val_noise_std > 0:
                        A0_i = add_masked_symmetric_noise(A0_i, node_mask[i], edge_mask=(Omega),
                                                           sigma=args.val_noise_std, clip01=True)
                    A0_list.append(A0_i); Omega_list.append(Omega); Aobs_list.append(A_obs)

                A0          = torch.stack(A0_list,   dim=0)
                Omega_batch = torch.stack(Omega_list,dim=0)
                Aobs_batch  = torch.stack(Aobs_list, dim=0)

                t = torch.rand(B, device=device)
                alpha, beta, _, _ = linear_coeffs(t)
                I_t    = sym_zero_diag_valid(alpha.view(B,1,1) * A0 + beta.view(B,1,1) * A_batch, node_mask)
                target = sym_zero_diag_valid(A_batch - A0, node_mask)

                inp    = I_t.unsqueeze(1)
                x_feat = torch.zeros(B, N, 1, device=device)
                b_pred = denoiser(x_feat, inp, node_mask, t)
                b_pred = sym_zero_diag_valid(b_pred, node_mask)
                b_pred = b_pred * Omega_batch

                l = masked_upper_mse(
                    b_pred,
                    target,
                    node_mask,
                    loss_mask=(1.0 - Aobs_batch),
                    reduction="per_graph",
                )
                sum_per_graphs += float(l.item()) * B
                graphs_seen    += B

        val_losses.append(sum_per_graphs / max(1, graphs_seen))
        print(f"[FAKE TRAIN | Ω=A_obs] Epoch {epoch}: train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")

        if (epoch % args.ckpt_every == 0) or (epoch == args.epochs):
            path = os.path.join(ckpt_dir, f"ep{epoch:04d}.pt")
            torch.save(denoiser.state_dict(), path)
            print(f"💾 [FAKE] Checkpoint saved @ epoch {epoch}: {path}")

    # Curves
    os.makedirs(LOSS_CURVE_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss (FAKE|Ω=A_obs)', marker='o')
    plt.plot(val_losses,   label='Validation Loss (FAKE|Ω=A_obs)', marker='x')
    plt.title("Training and Validation Loss (Fake-Edge Removal, Ω = ones of A_obs)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
    ts = time.strftime("%Y%m%d_%H%M%S")
    loss_plot_path = os.path.join(LOSS_CURVE_DIR, f"loss_curve_FAKE_{ts}.png")
    plt.savefig(loss_plot_path, dpi=300); plt.close()
    print(f"📉 Loss curve saved to: {loss_plot_path}")

# ===========================
# Sample denoising
# ===========================
def sample_denoise(args):
    """
    Inference for FAKE-EDGE detector (Ω = A_obs = A_true + R):
      • Moves ONLY on ones of A_obs to delete fake ones and keep true ones.
      • A0 = Ω ⊙ Prior (external) or Ω filled with Gaussian noise, with optional noise added on Ω.
      • Euler rollouts for a set of n_steps values.
      • Evaluates region-wise: Ω (A_obs==1), R-only (fake), A_true==1 (real), and optional A_obs==0.
    Required flags (mirrors train_fake IO):
      --ckpt
      AND either:
        (--sample_pkl  AND --fake_edge_mask_pkl)
        OR
        (--input_graph AND --fake_edge_mask_npy)
    Optional:
      --fake_prior_test_dir (required only when fake_prior_init=prior)
      --sample_nsteps "1,2,5,10,20,50,100"   # overrides default n_steps grid
      --noise_std (float)                    # sampling noise std for A0
      --hidden_dim, --num_layers, --num_linears, --c_init, --c_hid, --c_final
    """
    import os, time, pickle, numpy as np, torch, networkx as nx
    from torch.utils.data import DataLoader

    mode = getattr(args, 'fake_prior_init', 'prior')
    gauss_mean = getattr(args, 'fake_gauss_mean', 0.5)
    gauss_var = getattr(args, 'fake_gauss_var', 1.0)
    flow_label = "Sample Denoise"
    region_scope = "fake-edge (Ω=A_obs==1)"
    test_prior_dir = getattr(args, 'fake_prior_test_dir', None)
    if mode == 'prior' and not test_prior_dir:
        print("⚠️  No fake prior directory provided for sampling; falling back to Gaussian initialization.")
        mode = 'gaussian'
        setattr(args, 'fake_prior_init', 'gaussian')

    # setup
    print("[Device] Initializing device...", flush=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    base_dir = MMSE_FAKE_DIR
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_tag = getattr(args, "name", "FAKE") + f"_{ts}"
    run_dir = os.path.join(base_dir, run_tag)
    plot_dir        = os.path.join(run_dir, "plots")
    a0_raw_dir      = os.path.join(run_dir, "A0_raw")
    recon_raw_dir   = os.path.join(run_dir, "recon_raw")
    for d in (plot_dir, a0_raw_dir, recon_raw_dir):
        os.makedirs(d, exist_ok=True)
    prefix = os.path.basename(run_dir)
    
    print(f"[Run] Starting {flow_label} (prior_init={mode}, scope={region_scope}) → {run_tag}: {run_dir}", flush=True)
    print(f"[Device] CUDA available: {cuda_available} | Using device: {device}", flush=True)
    if cuda_available:
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    

    # load graphs from disk
    if getattr(args, "sample_pkl", None) and getattr(args, "fake_edge_mask_pkl", None):
        with open(args.sample_pkl, 'rb') as f:
            sample_graphs = pickle.load(f)
        A1_list = [
            torch.tensor(nx.to_numpy_array(g), dtype=torch.float32, device=device)
            if isinstance(g, nx.Graph) else g.to(dtype=torch.float32, device=device)
            for g in sample_graphs
        ]
        with open(args.fake_edge_mask_pkl, 'rb') as f:
            R_list_np = pickle.load(f)
        R_list = [torch.from_numpy(r).to(device).float() for r in R_list_np]
        print(f"→ Loaded {len(A1_list)} graphs and {len(R_list)} fake-edge mask matrices for sampling.")
    elif getattr(args, "input_graph", None) and getattr(args, "fake_edge_mask_npy", None):
        A = np.load(args.input_graph).astype(np.float32)
        R = np.load(args.fake_edge_mask_npy).astype(np.float32)
        A1_list = [torch.from_numpy(A).to(device)]
        R_list  = [torch.from_numpy(R).to(device)]
        print("→ Loaded single graph and fake-edge mask for sampling.")
    else:
        raise ValueError("Provide (--sample_pkl AND --fake_edge_mask_pkl) or (--input_graph AND --fake_edge_mask_npy).")

    #  prior setup
    if mode == 'prior':
        prior_dir = test_prior_dir
        assert prior_dir, "Internal error: prior dir missing despite prior mode."
        priors_cpu, z1d_cpu = load_priors_from_npy_dir(prior_dir, [r.detach().cpu() for r in R_list], args)
        Y_prior_list = [p.to(device).float() for p in priors_cpu]
        z1d_list     = [z.to(device).float() for z in z1d_cpu]
        print(f"[priors] Loaded {len(Y_prior_list)} prior matrices for sampling.")
    else:
        Y_prior_list = [None] * len(R_list)
        z1d_list     = [None] * len(R_list)

    print(f"[IO] Loaded {len(A1_list)} graph(s) and {len(R_list)} fake-edge mask matrix/matrices.", flush=True)


    def _n_of(x): return x.size(0) if torch.is_tensor(x) else x.number_of_nodes()
    maxN = max(_n_of(a) for a in A1_list)
    denoiser = DenoiseNetworkA(
        max_feat_num=1,
        max_node_num=maxN,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,  
        # num_heads=args.num_heads,    # optional (default 4)
        # conv=args.conv,              # optional (default 'GCN')
    ).to(device)
    denoiser.load_state_dict(torch.load(args.ckpt, map_location=device))
    denoiser.eval()
    
    param_count = sum(p.numel() for p in denoiser.parameters())
    print(f"[Model] Denoiser loaded from {args.ckpt} ({param_count/1e6:.2f}M params).", flush=True)
    if cuda_available:
        assert next(denoiser.parameters()).is_cuda, "Model not on CUDA!"


    # n_step_values
    default_grid = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100]
    custom_grid = getattr(args, 'sample_nsteps', '')
    if custom_grid:
        try:
            n_steps_values = [int(s) for s in custom_grid.split(',') if s.strip()]
        except ValueError as exc:
            raise ValueError('--sample_nsteps must be comma-separated integers') from exc
    else:
        n_steps_values = default_grid
    if not n_steps_values:
        raise ValueError('No valid n_steps provided; supply at least one integer')
    print(f"[Schedule] n_steps grid: {n_steps_values}", flush=True)
    print(f"🚀 Sampling with n_steps grid: {n_steps_values}")
    print("======= FAKE-EDGE INFERENCE =======")
    print("• A_obs = clip(A_true + R, 0,1)")
    print("• Ω = ones(A_obs)  (move ONLY on ones of observed graph)")
    if mode == 'prior':
        print("• A0 = Ω ⊙ Prior; Gaussian noise added ONLY on Ω")
    else:
        print(f"• A0 ~ N({gauss_mean}, {gauss_var}) on Ω; optional noise added on Ω")
    print("• Euler rollout; b_pred masked to Ω; outside-Ω clamped to 0")

    # data collectors
    final_recons_by_steps = {k: [] for k in n_steps_values}
    true_graphs    = []
    plot_paths     = []
    initial_raws   = []

    # Evaluator masks (score where mask==0):
    mask_Omega_list    = []   # Ω = A_obs==1, mask = 1 - A_obs
    mask_R_list        = []   # fake-only (R==1), mask = 1 - R
    mask_Atrue1_list   = []   # true ones, mask = 1 - A_true
    mask_AobsZero_list = []   # optional: A_obs==0, mask = A_obs

    print(f"[Loop] Sampling configuration: prior_init={mode}, scope={region_scope}", flush=True)
    print(f"[Loop] Beginning per-graph sampling over {len(A1_list)} graph(s).", flush=True)
    
    # sample per graph
    for i, (A1, R_in, Y_prior, z_full) in enumerate(zip(A1_list, R_list, Y_prior_list, z1d_list)):
        print(f"[Graph {i+1}/{len(A1_list)}] start (N={A1.size(0)})", flush=True)
        node_mask = torch.ones(A1.size(0), dtype=torch.bool, device=device)
        A1 = sym_zero_diag_valid(A1, node_mask)
        R  = sym_zero_diag_valid(R_in, node_mask) * (1.0 - (A1 > 0).float())

        if mode == 'prior' and Y_prior is not None and z_full is not None:
            p = torch.argsort(z_full, dim=0)
            p_inv = invert_perm(p)
            z_full = z_full.index_select(0, p)
            Y_prior_perm = permute_square(Y_prior, p)
        else:
            p = torch.arange(node_mask.size(0), device=device, dtype=torch.long)
            p_inv = invert_perm(p)
            Y_prior_perm = Y_prior
        A1p = permute_square(A1, p)
        Rp  = permute_square(R, p)
        node_mask_p = node_mask.index_select(0, p)

        A_obs = sym_zero_diag_valid(torch.clamp(A1p + Rp, 0.0, 1.0), node_mask_p)
        Omega = A_obs.clone()

        if mode == 'prior' and Y_prior_perm is not None:
            A0 = Omega * Y_prior_perm
        else:
            A0 = init_gaussian_on_omega(Omega, node_mask_p, gauss_mean, gauss_var)
        noise_std = getattr(args, 'noise_std', 0.01)
        if noise_std > 0:
            A0 = add_masked_symmetric_noise(
                M=A0, node_mask=node_mask_p, edge_mask=(1.0 - Omega),
                sigma=noise_std, clip01=True
            )

        A0_unperm    = permute_square(A0, p_inv)
        A_obs_unperm = permute_square(A_obs, p_inv)
        R_unperm     = permute_square(Rp,  p_inv)

        initial_raws.append(A0_unperm.detach().cpu().clone())
        true_graphs.append(A1.detach().cpu().clone())
        np.save(os.path.join(a0_raw_dir, f"{prefix}_sample{i}_A0raw.npy"), A0_unperm.detach().cpu().numpy())

        mask_Omega_list.append((1.0 - A_obs_unperm).detach().cpu().clone())
        mask_R_list.append((1.0 - R_unperm).detach().cpu().clone())
        mask_Atrue1_list.append((1.0 - A1).detach().cpu().clone())
        mask_AobsZero_list.append(A_obs_unperm.detach().cpu().clone())

        A_final_for_plot = None
        num_nodes = node_mask_p.size(0)
        for current_n in n_steps_values:
            print(f"  ├─ n_steps={current_n} (dt={1.0/float(current_n):.4f})", flush=True)
            A = A0.clone()
            dt = 1.0 / float(current_n)
            x_feat = torch.zeros(1, num_nodes, 1, device=device, dtype=A.dtype)

            for step in range(current_n):
                inp = A.unsqueeze(0).unsqueeze(1)
                with torch.no_grad():
                    t = torch.full((1,), step * dt, device=device)
                    b = denoiser(x_feat, inp, node_mask_p.unsqueeze(0), t).squeeze(0)
                    b = sym_zero_diag_valid(b, node_mask_p)
                b = b * Omega
                A = A + dt * b
                A.clamp_(0.0, 1.0)
                A = Omega * A
                A = sym_zero_diag_valid(A, node_mask_p)

            A_final_unperm = permute_square(A, p_inv)
            zero_diag_(A_final_unperm)
            final_recons_by_steps[current_n].append(A_final_unperm.detach().cpu().clone())
            np.save(os.path.join(recon_raw_dir, f"{prefix}_sample{i}_{current_n}steps_recon_raw.npy"),
                    A_final_unperm.detach().cpu().numpy())

            if current_n == max(n_steps_values):
                A_final_for_plot = A_final_unperm
        if A_final_for_plot is not None:
            rec_bin = (A_final_for_plot > 0.5).float()
            diff = (rec_bin - A1).cpu()
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(A1.cpu(), cmap='Greys');              axes[0].set_title("True");          axes[0].axis("off")
            axes[1].imshow(A_obs_unperm.cpu(), cmap='Greys');    axes[1].set_title("A_obs (Ω)");     axes[1].axis("off")
            axes[2].imshow(rec_bin.cpu(), cmap='Greys');         axes[2].set_title(f"Recon @{max(n_steps_values)}"); axes[2].axis("off")
            v = float(max(diff.abs().max().item(), 1e-6))
            im = axes[3].imshow(diff, cmap='bwr', vmin=-v, vmax=+v); axes[3].set_title("Raw Δ"); axes[3].axis("off")
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            plt.tight_layout()
            pth = os.path.join(plot_dir, f"{prefix}_sample{i}.png")
            plt.savefig(pth, dpi=300); plt.close()
            plot_paths.append(pth)
        else:
            plot_paths.append("")
            
        print(f"[Graph {i+1}] done (saved {len(n_steps_values)} recons)", flush=True)
        
    csv_paths = []

    for n_steps, recon_list in final_recons_by_steps.items():
        tag = f"fake_{n_steps}steps"

        # (A) Ω = A_obs==1 (candidate ones region)
        out_omega = evaluate_and_save_real(
            args, true_graphs, recon_list, mask_Omega_list, plot_paths,
            st=f"{tag}_Omega_AobsOne",
            score_mode="raw",
            compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
            gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        );  csv_paths += [out_omega] if out_omega else []

        # (B) R-only (fake edges)
        out_fake = evaluate_and_save_real(
            args, true_graphs, recon_list, mask_R_list, plot_paths,
            st=f"{tag}_R_fakeOnly",
            score_mode="raw",
            compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
            gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        );  csv_paths += [out_fake] if out_fake else []

        # (C) A_true==1 (true edges retention)
        out_true1 = evaluate_and_save_real(
            args, true_graphs, recon_list, mask_Atrue1_list, plot_paths,
            st=f"{tag}_AtrueOne",
            score_mode="raw",
            compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
            gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        );  csv_paths += [out_true1] if out_true1 else []

        # (D) Optional: A_obs==0 union
        out_aobs0 = evaluate_and_save_real(
            args, true_graphs, recon_list, mask_AobsZero_list, plot_paths,
            st=f"{tag}_AobsZero",
            score_mode="raw",
            compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
            gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        );  csv_paths += [out_aobs0] if out_aobs0 else []

    # A0 baselines on the same regions
    out_a0_omega = evaluate_and_save_real(
        args, true_graphs, initial_raws, mask_Omega_list, plot_paths,
        st="A0raw_Omega_AobsOne",
        score_mode="raw",
        compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
        gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
        compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
        mmd_on="full_raw", mmd_max_samples=5000
    );  csv_paths += [out_a0_omega] if out_a0_omega else []

    out_a0_fake = evaluate_and_save_real(
        args, true_graphs, initial_raws, mask_R_list, plot_paths,
        st="A0raw_R_fakeOnly",
        score_mode="raw",
        compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
        gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
        compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
        mmd_on="full_raw", mmd_max_samples=5000
    );  csv_paths += [out_a0_fake] if out_a0_fake else []

    out_a0_true1 = evaluate_and_save_real(
        args, true_graphs, initial_raws, mask_Atrue1_list, plot_paths,
        st="A0raw_AtrueOne",
        score_mode="raw",
        compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
        gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
        compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
        mmd_on="full_raw", mmd_max_samples=5000
    );  csv_paths += [out_a0_true1] if out_a0_true1 else []

    out_a0_aobs0 = evaluate_and_save_real(
        args, true_graphs, initial_raws, mask_AobsZero_list, plot_paths,
        st="A0raw_AobsZero",
        score_mode="raw",
        compute_gw=True, gw_cost_mode="adj", gw_entropic=False,
        gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
        compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
        mmd_on="full_raw", mmd_max_samples=5000
    );  csv_paths += [out_a0_aobs0] if out_a0_aobs0 else []
    
    master_paths = aggregate_last_rows_for_run(csv_paths, args)
    if csv_paths:
        print("\n=== Saved CSV summary files ===")
        for p in csv_paths:
            print(p)

    if master_paths:
        print("\n=== Saved MASTER last-row CSVs ===")
        for p in master_paths:
            print(p)

    print("\n[FAKE] Sampling + region-wise evaluation complete.")

# -------------------------
# MMD ablation Sampling
# -------------------------
def sample_kgrid(args):
    """
    Decoupled multi-run sampler on ONE graph:
      • Loads one graph+mask with same flags as sample(): 
          (--sample_pkl & --mask_pkl) OR (--input_graph & --mask_npy). 
        (Also supports --fake_edge_mask_pkl legacy synonym.)
      • Requires --fake_prior_test_dir (for external priors) and --ckpt like sample().
      • Runs N_short times with K=--k_short (default 1) and 
        N_long times with K=--k_long (default 100), each run fresh.
      • Saves every pred in K*/pred_runXXX.npy and each K's A_avg.npy.
      • Saves A_true.npy and mask.npy once at the graph root.
      • Evaluates averaged maps with evaluate_and_save_real().
      • Computes Consensus over missing edges (mask==0 assumed).
    """
    import os, time, json
    import numpy as np
    import torch
    import networkx as nx
    from copy import deepcopy

    # output dirs
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir = args.out_dir or KGRID_RAW_DIR
    run_name = getattr(args, "name", "Kgrid")
    out_root = os.path.join(base_dir, f"{run_name}_{ts}", f"graph_{getattr(args, 'graph_index', 0)}")
    k_short = int(getattr(args, "k_short", 1))
    k_long  = int(getattr(args, "k_long", 100))
    n_short = int(getattr(args, "n_short", 10))
    n_long  = int(getattr(args, "n_long", 10))
    k1_dir   = os.path.join(out_root, f"K{k_short}")
    k100_dir = os.path.join(out_root, f"K{k_long}")
    os.makedirs(k1_dir, exist_ok=True)
    os.makedirs(k100_dir, exist_ok=True)
    print(f"📁 Writing all outputs under: {out_root}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing device... {device}")

    # load graphs
    A1_list = []
    M_list  = []

    if getattr(args, "sample_pkl", None) and (getattr(args, "mask_pkl", None) or getattr(args, "fake_edge_mask_pkl", None)):
        with open(args.sample_pkl, 'rb') as f:
            sample_graphs = pickle.load(f)
        A1_list = [
            torch.tensor(nx.to_numpy_array(g), dtype=torch.float32, device=device)
            if isinstance(g, nx.Graph) else g.to(dtype=torch.float32, device=device)
            for g in sample_graphs
        ]

        mask_path = getattr(args, "mask_pkl", None) or getattr(args, "fake_edge_mask_pkl", None)
        with open(mask_path, 'rb') as f:
            sample_masks = pickle.load(f)
        M_list = [torch.from_numpy(m).to(device).float() for m in sample_masks]
        print(f"→ Loaded {len(A1_list)} graphs and masks for sampling.")
    elif getattr(args, "input_graph", None) and getattr(args, "mask_npy", None):
        A = np.load(args.input_graph).astype(np.float32)
        M = np.load(args.mask_npy).astype(np.float32)
        A1_list = [torch.from_numpy(A).to(device).float()]
        M_list  = [torch.from_numpy(M).to(device).float()]
        print("→ Loaded single input graph and mask for sampling.")
    else:
        raise ValueError("Provide (--sample_pkl AND --mask_pkl) or (--input_graph AND --mask_npy).")

    prior_dir = getattr(args, 'fake_prior_test_dir', None) or getattr(args, 'n2v_prior_test_dir', None)
    if not prior_dir:
        raise ValueError("Sampling with external priors requires --fake_prior_test_dir.")

    # load priors
    graphs_cpu = [a.detach().cpu() for a in A1_list]
    masks_cpu  = [m.detach().cpu() for m in M_list]
    print(f"⌛ Loading priors from: {prior_dir}")
    test_priors, test_z1d = load_priors_from_npy_dir(prior_dir, masks_cpu, args)
    print(f"[priors] Loaded {len(test_priors)} prior matrices.")

    
    denoiser = DenoiseNetworkA(
        max_feat_num=1,
        max_node_num=args.max_graph_nodes,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,
        # num_heads=args.num_heads,
        # conv=args.conv,
    ).to(device)
    denoiser.load_state_dict(torch.load(args.ckpt, map_location=device))
    denoiser.eval()

    # pick a graph
    gidx = int(getattr(args, "graph_index", 0))
    if gidx < 0 or gidx >= len(A1_list):
        raise IndexError(f"graph_index {gidx} out of range (0..{len(A1_list)-1})")

    A1          = A1_list[gidx]  
    edge_mask   = M_list[gidx]  
    Y_prior     = test_priors[gidx].to(device).float()
    z_full      = test_z1d[gidx].to(device).float()

    node_mask = torch.ones(A1.size(0), dtype=torch.bool, device=device)
    p     = torch.argsort(z_full, dim=0)
    p_inv = invert_perm(p)
    z_full          = z_full.index_select(0, p)
    A1_perm         = permute_square(A1,       p)
    edge_mask_perm  = permute_square(edge_mask, p)
    node_mask       = node_mask.index_select(0, p)
    Y_prior_perm    = permute_square(Y_prior,  p)

    # Observed adjacency & Omega
    A_obs  = sym_zero_diag_valid(A1_perm * edge_mask_perm, node_mask)
    Omega  = 1.0 - A_obs
    A0_clean = Omega * Y_prior_perm + A_obs  # same as in sample()
    
    A_obs_unp = permute_square(A_obs, p_inv).detach().cpu().numpy()

    # Save (unpermuted) GT & mask once
    A_true_unp = permute_square(A1_perm, p_inv).detach().cpu().numpy()
    mask_unp   = permute_square(edge_mask_perm, p_inv).detach().cpu().numpy()
    np.save(os.path.join(out_root, "A_true.npy"), A_true_unp)
    np.save(os.path.join(out_root, "mask.npy"),   mask_unp)

    # helpers
    def to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().float().numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return x

    def rollout_once(K_steps: int, subdir: str, run_idx: int):
        # fresh noise per run
        base = int(time.time()) % 10_000_000
        seed = base + (K_steps * 1000) + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))

        A0_noisy = add_masked_symmetric_noise(
            M=A0_clean, node_mask=node_mask, edge_mask=A_obs,  
            sigma=getattr(args, "noise_std", 0.1), clip01=True
        )

        A = A0_noisy.clone()
        dt = 1.0 / float(K_steps)
        x_feat = torch.zeros(1, z_full.shape[0], 1, device=device, dtype=z_full.dtype)

        for step in range(K_steps):
            inp = A.unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                t = torch.full((1,), step * dt, device=device)
                b = denoiser(x_feat, inp, node_mask.unsqueeze(0), t).squeeze(0)
                b = sym_zero_diag_valid(b, node_mask)
            b = b * Omega
            A = A + dt * b
            A.clamp_(0.0, 1.0)
            A = A_obs + Omega * A
            A = sym_zero_diag_valid(A, node_mask)

        A_unp = permute_square(A, p_inv)
        zero_diag_(A_unp)
        np_pred = to_np(A_unp)
        np.save(os.path.join(subdir, f"pred_run{run_idx:03d}.npy"), np_pred)
        return np_pred

    # K-short runs 
    print(f"▶️  Running K={k_short} for N={n_short} runs…")
    preds_short = []
    for r in range(n_short):
        preds_short.append(rollout_once(k_short, k1_dir, r))
    A_avg_short = np.mean(np.stack(preds_short, axis=0), axis=0).astype(np.float32)
    np.save(os.path.join(k1_dir, "A_avg.npy"), A_avg_short)

    # K-long runs
    print(f"▶️  Running K={k_long} for N={n_long} runs…")
    preds_long = []
    for r in range(n_long):
        preds_long.append(rollout_once(k_long, k100_dir, r))
    A_avg_long = np.mean(np.stack(preds_long, axis=0), axis=0).astype(np.float32)
    np.save(os.path.join(k100_dir, "A_avg.npy"), A_avg_long)

    # evaluate the metrics
    true_graphs     = [torch.from_numpy(A_true_unp).float()]
    plot_paths_stub = [""] * len(true_graphs) 
    mask_tensor     = torch.from_numpy(mask_unp).float()
    edge_mask_list  = [mask_tensor]  
    rec_short       = [torch.from_numpy(A_avg_short).float()]
    rec_long        = [torch.from_numpy(A_avg_long).float()]
    aobs_mask_t   = torch.from_numpy(A_obs_unp).float()
    aobs_mask_list = [aobs_mask_t]
    atrue_mask_list = [torch.from_numpy(A_true_unp).float()]


    if 'evaluate_and_save_real' in globals():
        print("📊 Evaluating averaged K_short …")
        evaluate_and_save_real(
            args, true_graphs, rec_short, edge_mask_list, plot_paths_stub,
            st=f"K{k_short}_avg", score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )
        print("📊 Evaluating averaged K_long …")
        evaluate_and_save_real(
            args, true_graphs, rec_long, edge_mask_list, plot_paths_stub,
            st=f"K{k_long}_avg", score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )
        
        print("📊 Evaluating averaged K_short on A_obs==0 …")
        evaluate_and_save_real(
            args, true_graphs, rec_short, aobs_mask_list, plot_paths_stub,
            st=f"K{k_short}_avg_AobsZero", score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )

        print("📊 Evaluating averaged K_long on A_obs==0 …")
        evaluate_and_save_real(
            args, true_graphs, rec_long, aobs_mask_list, plot_paths_stub,
            st=f"K{k_long}_avg_AobsZero", score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )
        
        
        print("📊 Evaluating averaged K_short on A_obs==0 …")
        evaluate_and_save_real(
            args, true_graphs, rec_short, aobs_mask_list, plot_paths_stub,
            st=f"K{k_short}_avg_AobsZero", score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )

        print("📊 Evaluating averaged K_long on A_obs==0 …")
        evaluate_and_save_real(
            args, true_graphs, rec_long, aobs_mask_list, plot_paths_stub,
            st=f"K{k_long}_avg_AobsZero", score_mode="raw",
            compute_gw=True, gw_cost_mode="adj",
            gw_entropic=False, gw_epsilon=0.2, gw_max_iter=20000, gw_tol=1e-7,
            compute_mmd=True, mmd_kernel="rbf", mmd_sigma="median",
            mmd_on="full_raw", mmd_max_samples=5000
        )
        
    # computing consensus on the masked region
    missing = (mask_unp == 0)
    y_true_masked = A_true_unp[missing]
    y_hat_masked  = A_avg_short[missing]
    cons_short = compute_consensus(y_true_masked, y_hat_masked)
    with open(os.path.join(k1_dir, "consensus.json"), "w") as f:
        json.dump({"K": k_short, "N": n_short, "consensus": cons_short}, f, indent=2)

    y_hat_masked = A_avg_long[missing]
    cons_long = compute_consensus(y_true_masked, y_hat_masked)
    with open(os.path.join(k100_dir, "consensus.json"), "w") as f:
        json.dump({"K": k_long, "N": n_long, "consensus": cons_long}, f, indent=2)
    summary = {
        "graph_index": gidx,
        "out_root": out_root,
        "K_short": {"K": k_short, "N": n_short, "consensus": cons_short},
        "K_long":  {"K": k_long,  "N": n_long,  "consensus": cons_long},
        "notes": "A_true.npy and mask.npy saved at graph root; individual runs saved in K*/pred_runXXX.npy; averaged in K*/A_avg.npy",
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Done.")
    print(f"📦 All artifacts saved under: {out_root}")


# -------------------------
# CLI
# -------------------------

def _configure_train_expansion_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--name', type=str, default='ext_graphs',
               help="Short identifier for dataset split; used in filenames.")
    parser.add_argument('--train_pkl', type=str, default=None, help="Path to train_graphs.pkl (required unless --subgraph_lp)")
    parser.add_argument('--val_pkl',   type=str, default=None, help="Path to val_graphs.pkl (required unless --subgraph_lp)")

    parser.add_argument('--train_noise_std', type=float, default=0.05,
               help='Gaussian std at t=0 ONLY on masked edges during training.')
    parser.add_argument('--val_noise_std', type=float, default=0.05,
               help='Gaussian std at t=0 ONLY on masked edges during validation.')

    parser.add_argument('--prior_init', choices=['prior', 'gaussian', 'baseline'], default='prior',
               help='Initial A0 policy: use external prior, Gaussian fill on Ω, or keep A_obs.')
    parser.add_argument('--init_gauss_mean', type=float, default=0.5,
               help='Mean for Gaussian A0 initialization when prior_init=gaussian.')
    parser.add_argument('--init_gauss_var', type=float, default=1.0,
               help='Variance for Gaussian A0 initialization when prior_init=gaussian.')

    # Node2Vec hyperparams
    parser.add_argument('--n2v_dim', type=int, default=64)
    parser.add_argument('--n2v_walk_length', type=int, default=30)
    parser.add_argument('--n2v_walks_per_node', type=int, default=10)
    parser.add_argument('--n2v_context', type=int, default=10)
    parser.add_argument('--n2v_epochs', type=int, default=300)
    parser.add_argument('--n2v_clf_epochs', type=int, default=300)
    parser.add_argument('--n2v_p', type=float, default=1.0)
    parser.add_argument('--n2v_q', type=float, default=1.0)

    # flow training
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--batch_size',   type=int, default=1)
    parser.add_argument('--lr',           type=float, default=0.0002)
    parser.add_argument('--drop_prob',    type=float, default=0.1)
    parser.add_argument('--hidden_dim',   type=int, default=32)
    parser.add_argument('--num_layers',   type=int, default=5)
    parser.add_argument('--num_linears',  type=int, default=2)
    parser.add_argument('--c_init',       type=int, default=2)
    parser.add_argument('--c_hid',        type=int, default=8)
    parser.add_argument('--c_final',      type=int, default=4)
    parser.add_argument('--num_heads',    type=int, default=4)
    parser.add_argument('--conv',         type=str, default='GCN')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--ckpt_every', type=int, default=100,
               help='Save a checkpoint every N epochs (and at the final epoch).')
    parser.add_argument('--val_posterior_every', type=int, default=0,
               help='If >0, run MMSE posterior eval on validation every K epochs.')
    parser.add_argument('--val_posterior_k', type=int, default=5,
               help='How many validation samples to run posterior on when triggered.')
    parser.add_argument('--val_save_steps', type=int, nargs='*',
               default=[2, 100, 300, 500, 700, 900],
               help='ODE steps to snapshot during validation posterior eval.')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--prior_train_dir', '--n2v_prior_train_dir', dest='prior_train_dir', type=str, default=None,
               help='Folder with per-graph prior matrices for training (required when prior_init=prior).')
    parser.add_argument('--prior_val_dir', '--n2v_prior_val_dir', dest='prior_val_dir', type=str, default=None,
               help='Folder with per-graph prior matrices for validation (required when prior_init=prior).')

    # Subgraph link-prediction flags
    parser.add_argument('--subgraph_lp', action='store_true', dest='subgraph_lp_mode',
               help='Enable SGDM subgraph link-prediction pipeline.')
    parser.add_argument('--no_subgraph_lp', action='store_false', dest='subgraph_lp_mode',
               help='Disable the SGDM subgraph link-prediction pipeline.')
    parser.add_argument('--subgraph_mode', action='store_true', dest='subgraph_lp_mode', help=argparse.SUPPRESS)
    parser.set_defaults(subgraph_lp_mode=False)
    parser.add_argument('--single_graph_path', type=str, default=None, help='Path to a single large graph adjacency (npy/pkl).')
    parser.add_argument('--split_seed', type=int, default=0, help='Random seed for edge-level train/val/test split in subgraph mode.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation edge ratio for subgraph mode.')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Test edge ratio for subgraph mode.')
    parser.add_argument('--sampler', type=str, default='egonet', choices=['egonet'], help='Subgraph sampler to use.')
    parser.add_argument('--k_hop', type=int, default=2, help='Hop count for ego-net sampler in subgraph mode.')
    parser.add_argument('--max_nodes', type=int, default=256, help='Maximum nodes per sampled subgraph.')
    parser.add_argument('--target_coverage', type=int, default=2, help='Minimum number of times each node must appear per epoch.')
    parser.add_argument('--lap_pe_dim', type=int, default=8, help='Laplacian positional encoding dimensionality.')
    parser.add_argument('--train_edge_drop_p', type=float, default=0.1, help='Self-supervised edge drop probability inside subgraphs.')
    parser.add_argument('--feature_adapter', dest='feature_adapter', action='store_true', help='Project subgraph features to a single channel.')
    parser.add_argument('--no_feature_adapter', dest='feature_adapter', action='store_false', help='Disable feature adapter.')
    parser.set_defaults(feature_adapter=True)
    parser.add_argument(
        '--train_edge_centered_subgraphs',
        action='store_true',
        help='If set, training subgraphs are k-hop ego-nets centered on each train edge with exactly one masked edge per subgraph.',
    )
    parser.add_argument('--local_context', dest='use_local_context', action='store_true',
               help='Include 2-D local context features (degree, normalized degree).')
    parser.add_argument('--no_local_context', dest='use_local_context', action='store_false',
               help='Disable the 2-D local context features.')
    parser.set_defaults(use_local_context=True)
    parser.add_argument(
        '--test_edge_centered_subgraphs',
        action='store_true',
        help='At inference (sample_expansion/sample_LP), build test subgraphs as k-hop unions around each test edge and mask the target edge.',
    )
    parser.add_argument(
        '--node_select_graph',
        type=str,
        default='train',
        choices=['train', 'val', 'test', 'full'],
        help="Adjacency used for k-hop subgraph sampling (default 'train' to avoid leaking held-out edges).",
    )
    parser.add_argument(
        '--subgraph_prior',
        choices=["node2vec", "zero", "lpformer", "graphsage", "graphsage_heart"],
        default='node2vec',
        help='Initialization prior inside subgraph mode (node2vec / LPFormer / GraphSAGE / zeros).'
    )
    parser.add_argument(
        '--subgraph_graphsage_emb_path',
        type=str,
        default=None,
        help='Path to precomputed GraphSAGE embeddings (.npy/.pt) of shape [N,D].'
    )
    parser.add_argument('--subgraph_graphsage_batch_size', type=int, default=256)
    parser.add_argument('--subgraph_graphsage_clf_epochs', type=int, default=30)
    parser.add_argument('--subgraph_graphsage_clf_lr', type=float, default=1e-2)
    parser.add_argument('--subgraph_graphsage_device', type=str, default='auto')
    parser.add_argument('--subgraph_sage_dim', type=int, default=64)
    parser.add_argument('--subgraph_sage_hidden_dim', type=int, default=64)
    parser.add_argument('--subgraph_sage_layers', type=int, default=2)
    parser.add_argument('--subgraph_sage_epochs', type=int, default=30)
    parser.add_argument('--subgraph_sage_lr', type=float, default=1e-2)
    parser.add_argument('--subgraph_sage_batch_size', type=int, default=1024)
    parser.add_argument('--subgraph_sage_agg', type=str, default='mean')

    # GraphSAGE-HEART prior hyperparameters
    parser.add_argument('--graphsage_heart_dim', type=int, default=128)
    parser.add_argument('--graphsage_heart_hidden_dim', type=int, default=128)
    parser.add_argument('--graphsage_heart_layers', type=int, default=2)
    parser.add_argument('--graphsage_heart_epochs', type=int, default=200)
    parser.add_argument('--graphsage_heart_lr', type=float, default=1e-2)
    parser.add_argument('--graphsage_heart_neg_ratio', type=float, default=1.0)
    parser.add_argument('--graphsage_heart_dropout', type=float, default=0.5)
    parser.add_argument('--subgraph_sage_heart_weight_decay', type=float, default=1e-4,
                        help='Weight decay for GraphSAGE-HEART optimizer.')
    parser.add_argument('--graphsage_heart_device', type=str, default='auto')

    parser.add_argument('--lpformer_ckpt', type=str, default=None,
               help='Path to a pretrained LPFormer checkpoint used when --subgraph_prior lpformer.')
    parser.add_argument('--lpformer_data_name', type=str, default=None,
               help='Dataset identifier passed to the LPFormer prior builder.')
    parser.add_argument('--lpformer_data_dir', type=str, default=None,
               help='Directory containing LPFormer dataset files (e.g., train_pos.txt).')
    parser.add_argument('--lpformer_prior_cache_dir', type=str, default=None,
               help='Directory where LPFormer subgraph priors are cached (precompute & reuse).')
    parser.add_argument('--lpformer_chunk_size', '--lpformer_edge_chunk_size', dest='lpformer_chunk_size',
               type=int, default=65536,
               help='Chunk size when querying the LPFormer prior for batched edge scores.')
    parser.add_argument('--subgraph_n2v_dim', type=int, default=32, help='Node2Vec embedding dimension for subgraph priors.')
    parser.add_argument('--subgraph_dataset_cfg', type=str, default='cfg/dataset.yaml',
               help='Path to YAML cfg with dataset.* block for SaGress sampling (empty to disable).')
    parser.add_argument('--subgraph_n2v_walk_length', type=int, default=10, help='Random walk length for Node2Vec.')
    parser.add_argument('--subgraph_n2v_walks_per_node', type=int, default=5, help='Number of walks per node for Node2Vec.')
    parser.add_argument('--subgraph_n2v_context_size', type=int, default=5, help='Context size for Node2Vec.')
    parser.add_argument('--subgraph_n2v_epochs', type=int, default=25, help='Training epochs for subgraph Node2Vec embeddings.')
    parser.add_argument('--subgraph_n2v_lr', type=float, default=0.01, help='Learning rate for Node2Vec.')
    parser.add_argument('--subgraph_n2v_batch_size', type=int, default=128, help='Batch size for Node2Vec random walk loader.')
    parser.add_argument('--subgraph_neg_ratio', type=float, default=1.0, help='Negative/positive ratio for Node2Vec logistic head.')
    parser.add_argument('--subgraph_clf_epochs', type=int, default=50, help='Epochs for Node2Vec logistic link predictor.')
    parser.add_argument('--subgraph_clf_lr', type=float, default=0.01, help='Learning rate for the Node2Vec logistic head.')
    parser.add_argument('--subgraph_n2v_device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
               help='Device to train Node2Vec priors on (auto selects CUDA when available).')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Optional checkpoint directory for subgraph training.')
    parser.add_argument('--subgraph_eval_batch_size', type=int, default=None,
               help='Batch size for validation/test subgraph evaluation (defaults to --batch_size).')
    return parser


def _configure_sample_expansion_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--prior_test_dir', '--n2v_prior_test_dir', dest='prior_test_dir', type=str, default=None,
               help='Folder with per-graph prior matrices for sampling (required when prior_init=prior).')

    parser.add_argument('--noise_std', type=float, default=0.05,
               help='Gaussian std at t=0 ONLY on masked edges (symmetric, zero-diag).')

    parser.add_argument('--prior_init', choices=['prior', 'gaussian', 'baseline'], default='prior',
               help='Initial A0 policy for sampling.')
    parser.add_argument('--init_gauss_mean', type=float, default=0.5,
               help='Mean for Gaussian A0 initialization when prior_init=gaussian.')
    parser.add_argument('--init_gauss_var', type=float, default=1.0,
               help='Variance for Gaussian A0 initialization when prior_init=gaussian.')
    parser.add_argument('--save_plots', dest='save_plots', action='store_true',
               help='If set, save diagnostic PNGs for each sampled graph (default: enabled).')
    parser.add_argument('--no_save_plots', dest='save_plots', action='store_false',
               help='Disable saving diagnostic PNGs during sampling.')
    parser.set_defaults(save_plots=True)

    parser.add_argument('--sample_pkl', type=str,
               help="List of test graphs (pickled list of nx.Graph or np/torch matrices).")
    parser.add_argument('--mask_pkl', type=str,
               help="List of masks aligned to --sample_pkl (1=observed, 0=masked).")
    parser.add_argument('--input_graph',  type=str,
               help="Path to a single .npy adjacency (if not using --sample_pkl).")
    parser.add_argument('--mask_npy',     type=str,
               help="Path to a single .npy mask (aligned with --input_graph).")

    # Node2Vec hyperparams (must match training cache for reuse)
    parser.add_argument('--n2v_dim', type=int, default=64)
    parser.add_argument('--n2v_walk_length', type=int, default=30)
    parser.add_argument('--n2v_walks_per_node', type=int, default=10)
    parser.add_argument('--n2v_context', type=int, default=10)
    parser.add_argument('--n2v_epochs', type=int, default=300)
    parser.add_argument('--n2v_clf_epochs', type=int, default=300)
    parser.add_argument('--n2v_p', type=float, default=1.0)
    parser.add_argument('--n2v_q', type=float, default=1.0)

    parser.add_argument('--name', type=str, default='ext_graphs')
    parser.add_argument('--epochs',       type=int, default=100)  # only used in file names
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--sample_nsteps', type=str, default='',
               help='Comma-separated list of diffusion step counts to evaluate (default 1,2,5,10,20,30,40,50,75,100).')
    parser.add_argument('--ckpt',         type=str, required=True,
               help="Path to trained denoiser checkpoint (.pt).")
    parser.add_argument('--max_graph_nodes', type=int, default=20,
               help="Max number of nodes used to initialize the denoiser")
    parser.add_argument('--drop_prob',    type=float, default=0.1)
    parser.add_argument('--hidden_dim',   type=int, default=32)
    parser.add_argument('--num_layers',   type=int, default=5)
    parser.add_argument('--num_linears',  type=int, default=2)
    parser.add_argument('--c_init',       type=int, default=2)
    parser.add_argument('--c_hid',        type=int, default=8)
    parser.add_argument('--c_final',      type=int, default=4)
    parser.add_argument('--num_heads',    type=int, default=4)
    parser.add_argument('--conv',         type=str, default='GCN')
    parser.add_argument('--seed',         type=int, default=42)

    # Subgraph link-prediction flags
    parser.add_argument('--subgraph_lp', action='store_true', dest='subgraph_lp_mode',
               help='Enable SGDM subgraph link-prediction inference.')
    parser.add_argument('--no_subgraph_lp', action='store_false', dest='subgraph_lp_mode',
               help='Disable the SGDM subgraph link-prediction inference.')
    parser.add_argument('--subgraph_mode', action='store_true', dest='subgraph_lp_mode', help=argparse.SUPPRESS)
    parser.set_defaults(subgraph_lp_mode=False)
    parser.add_argument('--single_graph_path', type=str, default=None, help='Path to a single large graph adjacency (npy/pkl).')
    parser.add_argument('--split_seed', type=int, default=0, help='Random seed for edge-level train/val/test split in subgraph mode.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation edge ratio for subgraph mode.')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Test edge ratio for subgraph mode.')
    parser.add_argument('--sampler', type=str, default='egonet', choices=['egonet'], help='Subgraph sampler to use.')
    parser.add_argument('--k_hop', type=int, default=2, help='Hop count for ego-net sampler in subgraph mode.')
    parser.add_argument('--max_nodes', type=int, default=256, help='Maximum nodes per sampled subgraph.')
    parser.add_argument('--target_coverage', type=int, default=2, help='Minimum number of times each node must appear per epoch.')
    parser.add_argument('--lap_pe_dim', type=int, default=8, help='Laplacian positional encoding dimensionality.')
    parser.add_argument('--batch_size', type=int, default=1, help='Subgraph batch size during inference.')
    parser.add_argument('--feature_adapter', dest='feature_adapter', action='store_true', help='Project subgraph features to a single channel.')
    parser.add_argument('--no_feature_adapter', dest='feature_adapter', action='store_false', help='Disable feature adapter.')
    parser.set_defaults(feature_adapter=True)
    parser.add_argument(
        '--subgraph_prior',
        choices=['node2vec', 'zero', 'lpformer', 'graphsage', 'graphsage_heart'],
        default='node2vec',
        help='Initialization prior inside subgraph sampling (node2vec / LPFormer / GraphSAGE / zeros).'
    )

    # GraphSAGE-specific CLI knobs mirrored from training for consistency
    parser.add_argument(
        '--subgraph_graphsage_emb_path',
        type=str,
        default=None,
        help='Path to precomputed GraphSAGE embeddings (.npy/.pt) of shape [N,D].'
    )
    parser.add_argument(
        '--subgraph_graphsage_batch_size',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--subgraph_graphsage_clf_epochs',
        type=int,
        default=30,
    )
    parser.add_argument(
        '--subgraph_graphsage_clf_lr',
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        '--subgraph_graphsage_device',
        type=str,
        default='auto',
    )
    parser.add_argument('--subgraph_sage_dim', type=int, default=64)
    parser.add_argument('--subgraph_sage_hidden_dim', type=int, default=64)
    parser.add_argument('--subgraph_sage_layers', type=int, default=2)
    parser.add_argument('--subgraph_sage_epochs', type=int, default=30)
    parser.add_argument('--subgraph_sage_lr', type=float, default=1e-2)
    parser.add_argument('--subgraph_sage_batch_size', type=int, default=1024)
    parser.add_argument('--subgraph_sage_agg', type=str, default='mean')
    parser.add_argument('--graphsage_heart_dim', type=int, default=128)
    parser.add_argument('--graphsage_heart_hidden_dim', type=int, default=128)
    parser.add_argument('--graphsage_heart_layers', type=int, default=2)
    parser.add_argument('--graphsage_heart_epochs', type=int, default=200)
    parser.add_argument('--graphsage_heart_lr', type=float, default=1e-2)
    parser.add_argument('--graphsage_heart_neg_ratio', type=float, default=1.0)
    parser.add_argument('--graphsage_heart_dropout', type=float, default=0.5)
    parser.add_argument('--subgraph_sage_heart_weight_decay', type=float, default=1e-4,
                        help='Weight decay for GraphSAGE-HEART optimizer.')
    parser.add_argument('--graphsage_heart_device', type=str, default='auto')
    parser.add_argument('--lpformer_ckpt', type=str, default=None,
               help='Path to a pretrained LPFormer checkpoint used when --subgraph_prior lpformer.')
    parser.add_argument('--lpformer_data_name', type=str, default=None,
               help='Dataset identifier passed to the LPFormer prior builder.')
    parser.add_argument('--lpformer_data_dir', type=str, default=None,
               help='Directory containing LPFormer dataset files (e.g., train_pos.txt).')
    parser.add_argument('--lpformer_prior_cache_dir', type=str, default=None,
               help='Directory where LPFormer subgraph priors are cached (precompute & reuse).')
    parser.add_argument('--lpformer_chunk_size', '--lpformer_edge_chunk_size', dest='lpformer_chunk_size',
               type=int, default=65536,
               help='Chunk size when querying the LPFormer prior for batched edge scores.')
    parser.add_argument('--subgraph_n2v_dim', type=int, default=32, help='Node2Vec embedding dimension for subgraph priors.')
    parser.add_argument('--subgraph_dataset_cfg', type=str, default='cfg/dataset.yaml',
               help='Path to YAML cfg with dataset.* block for SaGress sampling (empty to disable).')
    parser.add_argument('--subgraph_n2v_walk_length', type=int, default=10, help='Random walk length for Node2Vec.')
    parser.add_argument('--subgraph_n2v_walks_per_node', type=int, default=5, help='Number of walks per node for Node2Vec.')
    parser.add_argument('--subgraph_n2v_context_size', type=int, default=5, help='Context size for Node2Vec.')
    parser.add_argument('--subgraph_n2v_epochs', type=int, default=25, help='Training epochs for subgraph Node2Vec embeddings.')
    parser.add_argument('--subgraph_n2v_lr', type=float, default=0.01, help='Learning rate for Node2Vec.')
    parser.add_argument('--subgraph_n2v_batch_size', type=int, default=128, help='Batch size for Node2Vec random walk loader.')
    parser.add_argument('--subgraph_neg_ratio', type=float, default=1.0, help='Negative/positive ratio for Node2Vec logistic head.')
    parser.add_argument('--subgraph_clf_epochs', type=int, default=50, help='Epochs for Node2Vec logistic link predictor.')
    parser.add_argument('--subgraph_clf_lr', type=float, default=0.01, help='Learning rate for the Node2Vec logistic head.')
    parser.add_argument('--subgraph_n2v_device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
               help='Device to train Node2Vec priors on (auto selects CUDA when available).')
    parser.add_argument('--subgraph_eval_batch_size', type=int, default=128,
               help='Batch size for subgraph inference (defaults to 128).')
    parser.add_argument('--adapter_ckpt', type=str, default=None, help='Optional checkpoint for feature adapter parameters.')
    parser.add_argument('--local_context', dest='use_local_context', action='store_true',
               help='Include 2-D local context features (degree, normalized degree).')
    parser.add_argument('--no_local_context', dest='use_local_context', action='store_false',
               help='Disable the 2-D local context features.')
    parser.set_defaults(use_local_context=True)
    parser.add_argument(
        '--test_edge_centered_subgraphs',
        action='store_true',
        help='At inference (sample_expansion/sample_LP), build test subgraphs as k-hop unions around each test edge and mask the target edge.',
    )
    parser.add_argument(
        '--node_select_graph',
        type=str,
        default='train',
        choices=['train', 'val', 'test', 'full'],
        help="Adjacency used for k-hop subgraph sampling (default 'train' to avoid leaking held-out edges).",
    )
    
    # ---- Trajectory plotting (opt-in) ----
    parser.add_argument('--traj_plot', action='store_true',
                help='If set, save per-step adjacency plots A_t for a few samples.')
    parser.add_argument('--traj_k', type=int, default=None,
                help='Number of diffusion steps for the trajectory (default: --n_steps).')
    parser.add_argument('--traj_max_samples', type=int, default=2,
                help='How many test graphs to plot trajectories for.')
    parser.add_argument('--traj_every', type=int, default=1,
                help='Save every k-th step (default 1 = save all).')
    parser.add_argument('--subgraph_traj_plots', action='store_true',
                help='Enable rollout plots for subgraph inference (edge-centered mode only).')
    parser.add_argument('--subgraph_traj_max_samples', type=int, default=5,
                help='Number of subgraph rollouts to save when subgraph_traj_plots is enabled.')
    parser.add_argument('--make_pdf', action='store_true',
               help='If set, collate saved panel PNGs into PDFs and print their paths.')
    
    parser.add_argument('--sample_from_prior', action='store_true',
                help='If set, skip diffusion and just evaluate the provided prior on masked edges.')
    parser.add_argument('--sample_only', action='store_true',
                help='If set, skip evaluation/plots and only dump reconstructions and A0.')

    parser.add_argument('--clip_final', action='store_true',
                help='If set, clamp final recon to [0,1] before binarization.')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    # train
    p = _configure_train_expansion_parser(sub.add_parser('train_expansion', help='Train diffusion denoiser (union updates).'))
    lp_parser = _configure_train_expansion_parser(sub.add_parser('train_LP', help='Train diffusion denoiser on masked edges only.'))

    # sample
    s = _configure_sample_expansion_parser(sub.add_parser('sample_expansion', help='Sample diffusion reconstructions (union updates).'))
    slp_parser = _configure_sample_expansion_parser(sub.add_parser('sample_LP', help='Sample diffusion reconstructions on masked edges only.'))

    
    pf = sub.add_parser('train_denoise')
    pf.add_argument('--name', type=str, default='ext_graphs_fake',
                    help="Short identifier for FAKE training run.")
    pf.add_argument('--train_pkl', type=str, required=True, help="Path to train_graphs.pkl (GT).")
    pf.add_argument('--val_pkl',   type=str, required=True, help="Path to val_graphs.pkl (GT).")
    pf.add_argument('--train_fake_edge_mask_pkl', '--train_fake_mask_pkl', dest='train_fake_edge_mask_pkl',
                    type=str, required=True,
                    help="Pickle with list of fake-edge masks R for train (1 where a fake edge was injected).")
    pf.add_argument('--val_fake_edge_mask_pkl', '--val_fake_mask_pkl', dest='val_fake_edge_mask_pkl',
                    type=str, required=True,
                    help="Pickle with list of fake-edge masks R for val (1 where a fake edge was injected).")

    pf.add_argument('--train_noise_std', type=float, default=0.05,
                    help='Gaussian std at t=0 ONLY on FAKE pairs during training.')
    pf.add_argument('--val_noise_std', type=float, default=0.05,
                    help='Gaussian std at t=0 ONLY on FAKE pairs during validation.')
    pf.add_argument('--fake_prior_init', choices=['prior', 'gaussian'], default='gaussian',
                    help='Initialization for FAKE training: use external priors or Gaussian samples on Ω.')
    pf.add_argument('--fake_gauss_mean', type=float, default=0.5,
                    help='Mean for Gaussian FAKE initialization when fake_prior_init=gaussian.')
    pf.add_argument('--fake_gauss_var', type=float, default=1.0,
                    help='Variance for Gaussian FAKE initialization when fake_prior_init=gaussian.')

    # priors
    pf.add_argument('--n2v_dim', type=int, default=64)
    pf.add_argument('--n2v_walk_length', type=int, default=30)
    pf.add_argument('--n2v_walks_per_node', type=int, default=10)
    pf.add_argument('--n2v_context', type=int, default=10)
    pf.add_argument('--n2v_epochs', type=int, default=300)
    pf.add_argument('--n2v_clf_epochs', type=int, default=300)
    pf.add_argument('--n2v_p', type=float, default=1.0)
    pf.add_argument('--n2v_q', type=float, default=1.0)
    pf.add_argument('--fake_prior_train_dir', '--n2v_prior_train_dir', dest='fake_prior_train_dir',
                    type=str, default=None,
                    help="Folder with per-graph prior matrices for train (required when fake_prior_init=prior).")
    pf.add_argument('--fake_prior_val_dir', '--n2v_prior_val_dir', dest='fake_prior_val_dir',
                    type=str, default=None,
                    help="Folder with per-graph prior matrices for val (required when fake_prior_init=prior).")

    # Model & run
    pf.add_argument('--epochs',       type=int, default=100)
    pf.add_argument('--batch_size',   type=int, default=1)
    pf.add_argument('--lr',           type=float, default=0.0002)
    pf.add_argument('--hidden_dim',   type=int, default=32)
    pf.add_argument('--num_layers',   type=int, default=5)
    pf.add_argument('--num_linears',  type=int, default=2)
    pf.add_argument('--c_init',       type=int, default=2)
    pf.add_argument('--c_hid',        type=int, default=8)
    pf.add_argument('--c_final',      type=int, default=4)
    pf.add_argument('--seed',         type=int, default=42)
    pf.add_argument('--ckpt_every',   type=int, default=100)
    pf.add_argument('--flip_tag',     type=str, default='flip', help="Tag to identify the fake generator setting (e.g., rate0.05).")

    # denoising
    sf = sub.add_parser('sample_denoise')
    sf.add_argument('--name', type=str, default='ext_graphs_fake')
    sf.add_argument('--sample_pkl', type=str, required=True,
                    help="List of test GT graphs (pickled list).")
    sf.add_argument('--fake_edge_mask_pkl', '--fake_mask_pkl', dest='fake_edge_mask_pkl', type=str, required=True,
                    help="List of fake-edge masks R for test (1 where a fake edge was injected).")
    sf.add_argument('--fake_edge_mask_npy', '--fake_mask_npy', dest='fake_edge_mask_npy', type=str,
                    help="Path to a single fake-edge mask (.npy) when sampling one graph with --input_graph.")
    sf.add_argument('--fake_prior_test_dir', '--n2v_prior_test_dir', dest='fake_prior_test_dir',
                    type=str, default=None,
                    help="Folder with per-graph prior matrices for test (required when fake_prior_init=prior).")
    sf.add_argument('--ckpt', type=str, required=True, help="Path to trained FAKE denoiser checkpoint (.pt).")

    sf.add_argument('--noise_std', type=float, default=0.05,
                    help='Gaussian std at t=0 ONLY on FAKE pairs.')
    sf.add_argument('--fake_prior_init', choices=['prior', 'gaussian'], default='gaussian',
                    help='Initialization for FAKE sampling: use external priors or Gaussian samples on Ω.')
    sf.add_argument('--fake_gauss_mean', type=float, default=0.5,
                    help='Mean for Gaussian FAKE initialization when fake_prior_init=gaussian.')
    sf.add_argument('--fake_gauss_var', type=float, default=1.0,
                    help='Variance for Gaussian FAKE initialization when fake_prior_init=gaussian.')
    sf.add_argument('--n2v_dim', type=int, default=64)
    sf.add_argument('--n2v_walk_length', type=int, default=30)
    sf.add_argument('--n2v_walks_per_node', type=int, default=10)
    sf.add_argument('--n2v_context', type=int, default=10)
    sf.add_argument('--n2v_epochs', type=int, default=300)
    sf.add_argument('--n2v_clf_epochs', type=int, default=300)
    sf.add_argument('--n2v_p', type=float, default=1.0)
    sf.add_argument('--n2v_q', type=float, default=1.0)

    sf.add_argument('--epochs',       type=int, default=100)  # for filenames only
    sf.add_argument('--n_steps',      type=int, default=1000)
    sf.add_argument('--sample_nsteps', type=str, default='',
                    help='Comma-separated list of diffusion step counts (default 1,2,5,10,20,30,40,50,75,100).')
    sf.add_argument('--max_graph_nodes', type=int, default=20)
    sf.add_argument('--hidden_dim',   type=int, default=32)
    sf.add_argument('--num_layers',   type=int, default=5)
    sf.add_argument('--num_linears',  type=int, default=2)
    sf.add_argument('--c_init',       type=int, default=2)
    sf.add_argument('--c_hid',        type=int, default=8)
    sf.add_argument('--c_final',      type=int, default=4)
    sf.add_argument('--seed',         type=int, default=42)
    sf.add_argument('--flip_tag',     type=str, default='flip', help="Tag to identify the fake generator setting (e.g., rate0.05).")
    
    
    
    
    sp = sub.add_parser("sample_kgrid", help="Multi-run sampling on one graph with K-grid (decoupled).")
    sp.add_argument("--ckpt", type=str, required=True)

    # Same I/O as sample()
    sp.add_argument("--sample_pkl", type=str)
    sp.add_argument("--mask_pkl",   type=str)
    sp.add_argument("--fake_edge_mask_pkl", "--fake_mask_pkl", dest="fake_edge_mask_pkl", type=str)
    sp.add_argument("--input_graph", type=str)
    sp.add_argument("--mask_npy",    type=str)
    sp.add_argument("--fake_prior_test_dir", "--n2v_prior_test_dir", dest="fake_prior_test_dir", type=str, required=True)

    # Model / arch flags you already use
    sp.add_argument("--max_graph_nodes", type=int, required=True)
    sp.add_argument("--hidden_dim", type=int, required=True)
    sp.add_argument("--num_layers", type=int, required=True)
    sp.add_argument("--num_linears", type=int, required=True)
    sp.add_argument("--c_init", type=int, required=True)
    sp.add_argument("--c_hid",  type=int, required=True)
    sp.add_argument("--c_final",type=int, required=True)
    sp.add_argument("--noise_std", type=float, default=0.1)

    # K grid + reps
    sp.add_argument("--k_short", type=int, default=1)
    sp.add_argument("--k_long",  type=int, default=100)
    sp.add_argument("--n_short", type=int, default=10)
    sp.add_argument("--n_long",  type=int, default=10)

    # Which graph to use
    sp.add_argument("--graph_index", type=int, default=0)

    # Output naming
    sp.add_argument("--out_dir", type=str, default=None)
    sp.add_argument("--name",    type=str, default="Kgrid")

    sp.set_defaults(func=sample_kgrid)
    
    
    
    parser.add_argument(
        "--hidden_gaussian_prior",
        action="store_true",
        help="If set, initialize hidden entries with symmetric Gaussian noise instead of Node2Vec/zero."
    )
    parser.add_argument(
        "--hidden_gaussian_std",
        type=float,
        default=0.25,
        help="Std for Gaussian prior on hidden entries (mean is 0.5, then clamped to [0,1])."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Short identifier for this run; used to create timestamped checkpoint folders under outputs/."
    )
    
    # Which prior to use
    parser.add_argument(
        "--subgraph_prior",
        type=str,
        default="node2vec",
        choices=["node2vec", "zero", "lpformer", "graphsage", "graphsage_heart"],
    )


    # GraphSAGE embedding source
    parser.add_argument(
        "--subgraph_graphsage_emb_path",
        type=str,
        default=None,
        help="Path to GraphSAGE node embeddings (.npy/.pt) of shape [N, D]."
    ) 

    # Optional hyperparams for the logistic head on top of GraphSAGE embeddings
    parser.add_argument(
        "--subgraph_graphsage_batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--subgraph_graphsage_clf_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--subgraph_graphsage_clf_lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--subgraph_graphsage_device",
        type=str,
        default="auto",
    )
    
    
    parser.add_argument("--subgraph_sage_dim", type=int, default=64)
    parser.add_argument("--subgraph_sage_hidden_dim", type=int, default=64)
    parser.add_argument("--subgraph_sage_layers", type=int, default=2)
    parser.add_argument("--subgraph_sage_epochs", type=int, default=30)
    parser.add_argument("--subgraph_sage_lr", type=float, default=1e-2)
    parser.add_argument("--subgraph_sage_batch_size", type=int, default=1024)
    parser.add_argument("--subgraph_sage_agg", type=str, default="mean")
    
    parser.add_argument("--graphsage_heart_dim", type=int, default=128)
    parser.add_argument("--graphsage_heart_hidden_dim", type=int, default=128)
    parser.add_argument("--graphsage_heart_layers", type=int, default=2)
    parser.add_argument("--graphsage_heart_epochs", type=int, default=200)
    parser.add_argument("--graphsage_heart_lr", type=float, default=1e-2)
    parser.add_argument("--graphsage_heart_neg_ratio", type=float, default=1.0)
    parser.add_argument("--graphsage_heart_dropout", type=float, default=0.5)
    parser.add_argument("--subgraph_sage_heart_weight_decay", type=float, default=1e-4,
                        help="Weight decay for GraphSAGE-HEART optimizer.")
    parser.add_argument("--graphsage_heart_device", type=str, default="auto")

    parser.add_argument(
        "--test_edge_centered_subgraphs",
        action="store_true",
        help=(
            "At inference, build test subgraphs as k-hop unions "
            "around each target test edge instead of using SaGress sampling."
        ),
    )






    
    

    args = parser.parse_args()
    


    if args.mode == 'train_expansion':
        if getattr(args, 'subgraph_lp_mode', False):
            train_subgraph_lp(args)
        else:
            if args.train_pkl is None or args.val_pkl is None:
                raise ValueError("--train_pkl and --val_pkl are required unless --subgraph_lp is set.")
            train_expansion(args)
    elif args.mode == 'train_LP':
        if getattr(args, 'subgraph_lp_mode', False):
            train_subgraph_lp(args)
        else:
            if args.train_pkl is None or args.val_pkl is None:
                raise ValueError("--train_pkl and --val_pkl are required unless --subgraph_lp is set.")
            train_LP(args)
    elif args.mode == 'sample_expansion':
        if getattr(args, 'subgraph_lp_mode', False):
            infer_subgraph_lp(args)
        else:
            sample_expansion(args)
    elif args.mode == 'sample_LP':
        if getattr(args, 'subgraph_lp_mode', False):
            infer_subgraph_lp(args)
        else:
            sample_LP(args)
    elif args.mode == 'sample_kgrid':
        sample_kgrid(args)
    elif args.mode == 'train_denoise':
        train_denoise(args)
    elif args.mode == 'sample_denoise':
        sample_denoise(args)
