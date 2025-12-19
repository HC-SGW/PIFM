import os
import re
import time
from typing import List, Optional, Sequence

import numpy as np
import ot
import pandas as pd
import torch
from ignite.metrics import MaximumMeanDiscrepancy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .initialization import build_initial_A0, build_initial_A0_lp
from .paths import DIFFERENCE_DIR, VAL_MMSE_DIR
from .tensor_utils import (
    add_masked_symmetric_noise,
    maybe_subsample,
    permute_square,
    sym_zero_diag_valid,
    upper_triu_mask_batched,
    zero_diag_,
)


def aggregate_last_rows_for_run(csv_paths, args, master_dir: str = DIFFERENCE_DIR):
    """
    Reads each CSV in csv_paths, grabs the LAST ROW, annotates it with Region/Variant/n_steps/SourceCSV,
    and writes one master CSV per region. Returns list of created master paths.
    """
    os.makedirs(master_dir, exist_ok=True)

    def _detect_region(basename: str) -> str:
        if "Omega_AobsOne" in basename or "AobsOne" in basename:
            return "AobsOne"
        if "R_fakeOnly" in basename or "Ronly" in basename or "R_only" in basename:
            return "R_fakeOnly"
        if "AtrueOne" in basename:
            return "AtrueOne"
        if "AobsZero" in basename:
            return "AobsZero"
        if "trueZero" in basename:
            return "trueZero"
        return "masked"

    def _extract_n_steps(basename: str):
        m = re.search(r"(?:final_recon|fake)_(\d+)steps", basename)
        if m:
            return int(m.group(1))
        if "A0raw" in basename:
            return 0
        return None

    rows_by_region = {}

    for path in (csv_paths or []):
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            last = df.tail(1).copy()

            base = os.path.basename(path)
            region = _detect_region(base)
            n_steps = _extract_n_steps(base)

            variant = os.path.splitext(base)[0]
            if n_steps == 0 and "A0raw" in base:
                variant = "A0raw"

            last.insert(0, "Region", region)
            last.insert(1, "Variant", variant)
            last.insert(2, "n_steps", n_steps)
            last.insert(3, "SourceCSV", path)

            rows_by_region.setdefault(region, []).append(last)
        except Exception as e:
            print(f"[aggregate] skip {path}: {e}")

    if hasattr(args, "drop_prob"):
        mode_tag = f"{args.drop_prob}drop"
    elif hasattr(args, "flip_tag"):
        mode_tag = f"{args.flip_tag}"
    else:
        mode_tag = "modeNA"

    ts_master = time.strftime("%Y%m%d_%H%M%S")
    masters = []
    for region, chunks in rows_by_region.items():
        if not chunks:
            continue
        agg_df = pd.concat(chunks, ignore_index=True)
        master_name = (
            f"final_recon_ALL_lastrows_{args.name}_{region}_REAL_"
            f"{getattr(args,'epochs','NA')}epochs_{mode_tag}_{ts_master}.csv"
        )
        out_path = os.path.join(master_dir, master_name)
        agg_df.to_csv(out_path, index=False)
        masters.append(out_path)
        print(f"✅ Saved master last-row CSV for region '{region}' → {out_path}")

    return masters


def posterior_eval_on_val_samples(epoch, val_loader, denoiser, device, args, masked_only: bool = False):
    """
    Run MMSE-style posterior rollouts on up to K validation samples and save outputs.
    Uses the Node2Vec prior and z1d already provided by val_loader.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    mode = getattr(args, "prior_init", "prior")
    base = os.path.join(VAL_MMSE_DIR, f"{args.name}_ep{epoch:04d}_{ts}")
    plot_dir = os.path.join(base, "plots")
    rounded_dir = os.path.join(base, "rounded_A0")
    rounded_raw_dir = os.path.join(base, "rounded_A0_raw")
    recon_dir = os.path.join(base, "recon")
    recon_raw_dir = os.path.join(base, "recon_raw")

    for d in (plot_dir, rounded_dir, rounded_raw_dir, recon_dir, recon_raw_dir):
        os.makedirs(d, exist_ok=True)

    save_steps = args.val_save_steps
    raw_snapshots_by_step = {step: [] for step in save_steps}
    true_graphs, plot_paths, edge_mask_list, initial_raws = [], [], [], []

    processed = 0
    K = args.val_posterior_k

    denoiser.eval()
    with torch.no_grad():
        for A_batch, node_mask_batch, edge_mask_batch, Y_prior_batch, z1d_batch in val_loader:
            A_batch = A_batch.to(device)
            node_mask_batch = node_mask_batch.to(device)
            edge_mask_batch = edge_mask_batch.to(device)
            Y_prior_batch = Y_prior_batch.to(device) if Y_prior_batch is not None else None
            z1d_batch = z1d_batch.to(device) if z1d_batch is not None else None
            B = A_batch.size(0)

            for b in range(B):
                if processed >= K:
                    break

                A1 = A_batch[b]
                node_mask = node_mask_batch[b]
                edge_mask = edge_mask_batch[b]
                Y_prior = Y_prior_batch[b] if Y_prior_batch is not None else None
                z1d = z1d_batch[b] if z1d_batch is not None else None

                if getattr(args, "prior_init", "prior") == "prior":
                    if Y_prior is None or z1d is None:
                        raise RuntimeError(
                            "Validation loader missing priors/z-coordinates while prior_init='prior'."
                        )
                    p = torch.argsort(z1d, dim=0)
                    z1d = z1d.index_select(0, p)
                    node_mask = node_mask.index_select(0, p)
                    A1 = permute_square(A1, p)
                    edge_mask = permute_square(edge_mask, p)
                    Y_prior = permute_square(Y_prior, p)
                else:
                    p = torch.arange(A1.size(0), device=device)

                edge_mask = sym_zero_diag_valid(edge_mask, node_mask)
                A_obs = sym_zero_diag_valid(A1 * edge_mask, node_mask)

                if masked_only:
                    update_mask = sym_zero_diag_valid(1.0 - edge_mask, node_mask)
                    noise_edge_mask = edge_mask
                    A0_clean = build_initial_A0_lp(
                        args,
                        A_true=A1,
                        edge_mask=edge_mask,
                        node_mask=node_mask,
                        prior=Y_prior,
                    )
                    A_anchor = A_obs
                else:
                    update_mask = sym_zero_diag_valid(1.0 - A_obs, node_mask)
                    noise_edge_mask = A_obs
                    A0_clean = build_initial_A0(
                        args,
                        A_obs=A_obs,
                        node_mask=node_mask,
                        prior=Y_prior,
                        noise_std=0.0,
                    )
                    A_anchor = A_obs

                A0_noisy = A0_clean.clone()
                if args.val_noise_std > 0:
                    A0_noisy = add_masked_symmetric_noise(
                        M=A0_noisy,
                        node_mask=node_mask,
                        edge_mask=noise_edge_mask,
                        sigma=args.val_noise_std,
                        clip01=True,
                    )

                A0_rounded = (A0_noisy > 0.5).float()
                prefix = f"val_ep{epoch:04d}_sample{processed}"
                np.save(os.path.join(rounded_dir, f"{prefix}_A0rounded.npy"), A0_rounded.cpu().numpy())
                np.save(os.path.join(rounded_raw_dir, f"{prefix}_A0raw.npy"), A0_noisy.cpu().numpy())
                np.save(
                    os.path.join(rounded_raw_dir, f"{prefix}_A0raw_clean.npy"), A0_clean.cpu().numpy()
                )

                initial_raws.append(A0_clean.cpu().clone())

                A = A0_noisy.clone()

                dt = 1.0 / args.n_steps
                xfeat = torch.zeros(1, A1.size(0), 1, device=device, dtype=A1.dtype)
                for step in range(args.n_steps):
                    inp = A.unsqueeze(0).unsqueeze(1)
                    t = torch.full((1,), step * dt, device=device)

                    b = denoiser(xfeat, inp, node_mask.unsqueeze(0), t).squeeze(0)
                    b = sym_zero_diag_valid(b, node_mask)
                    b = b * update_mask

                    A = A + dt * b
                    A = A_anchor + update_mask * A
                    A = sym_zero_diag_valid(A, node_mask)

                    if step in save_steps:
                        A_step_raw = A.cpu().clone()
                        raw_snapshots_by_step[step].append(A_step_raw)

                zero_diag_(A)
                np.save(os.path.join(recon_raw_dir, f"{prefix}_reconstructed_raw.npy"), A.cpu().numpy())
                reconstructed_A = (A > 0.5).float()
                zero_diag_(reconstructed_A)

                diff = (reconstructed_A - A1).cpu()
                plot_path = os.path.join(plot_dir, f"{prefix}_plot.png")
                fig, axes = plt.subplots(1, 5, figsize=(16, 4))
                axes[0].imshow(A1.cpu(), cmap="Greys")
                axes[0].set_title("True")
                axes[0].axis("off")
                axes[1].imshow(edge_mask.cpu(), cmap="Greys")
                axes[1].set_title("Edge Mask")
                axes[1].axis("off")
                axes[2].imshow((A1 * edge_mask).cpu(), cmap="Greys")
                axes[2].set_title("Masked A1")
                axes[2].axis("off")
                axes[3].imshow(reconstructed_A.cpu(), cmap="Greys")
                axes[3].set_title("Reconstructed")
                axes[3].axis("off")
                v = diff.abs().max().item() or 1e-6
                im = axes[4].imshow(diff.cpu(), cmap="bwr", vmin=-v, vmax=+v)
                axes[4].set_title("Raw Δ")
                axes[4].axis("off")
                fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
                plt.suptitle(f"Val MMSE (init={mode}) @ epoch {epoch}", fontsize=10)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(plot_path, dpi=300)
                plt.close()

                np.save(os.path.join(recon_dir, f"{prefix}_reconstructed.npy"), reconstructed_A.cpu().numpy())

                true_graphs.append(A1.cpu().clone())
                edge_mask_list.append(edge_mask.cpu().clone())
                plot_paths.append(plot_path)

                processed += 1

            if processed >= K:
                break

    outpaths = []
    for step in save_steps:
        out = evaluate_and_save_real(
            args,
            true_graphs,
            raw_snapshots_by_step[step],
            edge_mask_list,
            plot_paths,
            st=step,
            score_mode="raw",
        )
        outpaths.append(out)

    evaluate_and_save_real(
        args,
        true_graphs,
        initial_raws,
        edge_mask_list,
        plot_paths,
        st="A0RAW",
    )
    print(f"[Val posterior @ epoch {epoch}] Outputs → {base}")


def masked_upper_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    loss_mask: torch.Tensor,   # ← positive mask: 1 on supervised entries
    reduction: str = "global",
) -> torch.Tensor:
    """
    Batch MSE on exactly the entries selected by `loss_mask` (no inversion).
    Works when `loss_mask` is boolean or {0,1}. Returns 0 if mask is empty.
    reduction:
      - "global": average over all supervised entries in the batch
      - "per_graph": average per-graph, then mean across graphs (legacy behavior)
    """
    B, N, _ = pred.shape
    ut_valid = upper_triu_mask_batched(node_mask)
    eff = ut_valid & (loss_mask > 0)

    if reduction == "per_graph":
        per_graph = []
        for i in range(B):
            mu = eff[i]
            if mu.any():
                per_graph.append((pred[i][mu] - target[i][mu]).pow(2).mean())
            else:
                per_graph.append(pred.new_tensor(0.0))
        return torch.stack(per_graph).mean()
    if reduction != "global":
        raise ValueError('reduction must be "global" or "per_graph"')

    diff2 = (pred - target) ** 2
    num = (diff2 * eff.float()).sum()
    denom = eff.float().sum()

    if denom.item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return num / denom


def evaluate_and_save_real(
    args,
    A1_list,
    reconstructed_list,
    edge_masks,
    plot_paths,
    st,
    score_mode: str = "auto",
    compute_gw: bool = False,
    gw_cost_mode: str = "adj",
    gw_entropic: bool = True,
    gw_epsilon: float = 5e-3,
    gw_max_iter: int = 200,
    gw_tol: float = 1e-9,
    compute_mmd: bool = True,
    mmd_kernel: str = "rbf",
    mmd_sigma="median",
    mmd_on: str = "masked_raw",
    mmd_max_samples: int = 5000,
    mmd_seed: int = 0,
):
    rows = []
    inferred_mode = None

    for i, (A_true, A_rec, mask) in enumerate(zip(A1_list, reconstructed_list, edge_masks)):
        A_true_np = A_true.cpu().numpy()
        A_rec_np = A_rec.cpu().numpy()
        mask_np = mask.cpu().numpy()

        n = A_true_np.shape[0]
        iu = np.triu_indices(n, k=1)
        masked_upper = (1.0 - mask_np)[iu] == 1
        if not masked_upper.any():
            print(f"[⚠️] Sample {i} has 0 masked upper-triangle edges — skipping metrics")
            continue

        y_true_masked = A_true_np[iu][masked_upper]
        y_hat_masked = A_rec_np[iu][masked_upper]

        if inferred_mode is None:
            if score_mode.lower() == "raw":
                inferred_mode = "RAW"
            elif score_mode.lower() == "bin":
                inferred_mode = "BIN"
            else:
                is_binary = np.all((y_hat_masked == 0) | (y_hat_masked == 1))
                inferred_mode = "BIN" if is_binary else "RAW"

        mae = mean_absolute_error(y_true_masked, y_hat_masked)
        mse = mean_squared_error(y_true_masked, y_hat_masked)
        frob = np.linalg.norm(y_true_masked - y_hat_masked)

        y_bin = (y_true_masked > 0.5).astype(int)
        yhat_bin = (y_hat_masked > 0.5).astype(int)
        rec = recall_score(y_bin, yhat_bin, zero_division=0)
        f1 = f1_score(y_bin, yhat_bin, zero_division=0)

        try:
            auc = roc_auc_score(y_bin, y_hat_masked)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_bin, y_hat_masked)
        except ValueError:
            ap = float("nan")

        TP = int(np.logical_and(y_bin == 1, yhat_bin == 1).sum())
        FN = int(np.logical_and(y_bin == 1, yhat_bin == 0).sum())
        FP = int(np.logical_and(y_bin == 0, yhat_bin == 1).sum())
        TN = int(np.logical_and(y_bin == 0, yhat_bin == 0).sum())
        fn_denom = FN + TP
        fp_denom = FP + TN
        fn_rate = (FN / fn_denom) if fn_denom > 0 else float("nan")
        fp_rate = (FP / fp_denom) if fp_denom > 0 else float("nan")

        GW2, GW = float("nan"), float("nan")
        if compute_gw:
            try:
                GW2, GW = gw_distance(A_true_np, A_rec_np)
            except Exception as e:
                print(f"[GW] sample {i} failed: {e}")
                GW2, GW = np.nan, np.nan

        MMD2_un, MMD2_b = float("nan"), float("nan")

        if compute_mmd:
            iu = np.triu_indices(n, k=1)

            if mmd_on == "masked_raw":
                x = A_true_np[iu][masked_upper].astype(np.float32)
                y = A_rec_np[iu][masked_upper].astype(np.float32)
            elif mmd_on == "full_raw":
                x = A_true_np[iu].astype(np.float32)
                y = A_rec_np[iu].astype(np.float32)
            else:
                raise ValueError(f"Unknown mmd_on='{mmd_on}' (use 'masked_raw' or 'full_raw').")

            if x.size > mmd_max_samples:
                x = maybe_subsample(x, mmd_max_samples, seed=mmd_seed)
            if y.size > mmd_max_samples:
                y = maybe_subsample(y, mmd_max_samples, seed=mmd_seed + 1)

            x_t = torch.from_numpy(x).view(-1, 1)
            y_t = torch.from_numpy(y).view(-1, 1)
            mmd_metric = MaximumMeanDiscrepancy(var=1.0)
            mmd_metric.update((x_t, y_t))
            mmd2_val = mmd_metric.compute()
            mmd2_val = float(mmd2_val.item() if isinstance(mmd2_val, torch.Tensor) else mmd2_val)

            MMD2_un = mmd2_val
            MMD2_b = mmd2_val

        rows.append(
            {
                "sample": i,
                "PredMode": inferred_mode,
                "NumMasked": int(masked_upper.sum()),
                "MAE": mae,
                "MSE": mse,
                "FrobNorm": frob,
                "AP": ap,
                "Rec(0.5)": rec,
                "F1(0.5)": f1,
                "ROC_AUC": auc,
                "MMD2_unbiased_maskedRAW": MMD2_un,
                "MMD2_biased_maskedRAW": MMD2_b,
                "FN_rate": fn_rate,
                "FP_rate": fp_rate,
                "GW2": GW2,
                "GW": GW,
                "PlotPath": plot_paths[i],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows to save (all samples had 0 masked edges).")
        return None

    avg_row = df.mean(numeric_only=True)
    avg_row["sample"] = "average"
    pred_mode_for_header = df["PredMode"].iloc[0] if "PredMode" in df.columns else "RAW"
    avg_row["PredMode"] = pred_mode_for_header
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    renames = {
        "MAE": f"MAE [pred={pred_mode_for_header}]",
        "MSE": f"MSE [pred={pred_mode_for_header}]",
        "FrobNorm": f"FrobNorm [pred={pred_mode_for_header}]",
        "ROC_AUC": f"ROC_AUC [pred={pred_mode_for_header}]",
        "AP": f"AveragePrecision [pred={pred_mode_for_header}]",
        "Rec(0.5)": "Rec@0.5 [pred=BIN]",
        "F1(0.5)": "F1@0.5 [pred=BIN]",
        "MMD2_unbiased_maskedRAW": f"MMD^2 UNbiased [masked-raw, kernel={mmd_kernel}, sigma={mmd_sigma}]",
        "MMD2_biased_maskedRAW": f"MMD^2 biased [masked-raw, kernel={mmd_kernel}, sigma={mmd_sigma}]",
        "GW2": f"GW2 [cost={gw_cost_mode}]",
        "GW": f"GW [sqrt, cost={gw_cost_mode}]",
    }

    renames.update(
        {
            "MMD2_unbiased_maskedRAW": f"MMD^2 (unbiased) [kernel={mmd_kernel}, sigma={mmd_sigma}]",
            "MMD2_biased_maskedRAW": f"MMD^2 (biased) [kernel={mmd_kernel}, sigma={mmd_sigma}]",
        }
    )

    df.rename(columns=renames, inplace=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    epochs_tag = getattr(args, "epochs", "NA")
    steps_tag = getattr(args, "n_steps", "NA")
    if hasattr(args, "drop_prob"):
        mode_tag = f"{args.drop_prob}drop"
    elif hasattr(args, "flip_tag"):
        mode_tag = f"fake{args.flip_tag}"
    else:
        mode_tag = "modeNA"
    prior_mode = getattr(args, "prior_init", "prior")
    name = (
        f"{st}_{args.name}_{prior_mode}Init_REAL_{epochs_tag}epochs_{steps_tag}steps_{mode_tag}_{ts}.csv"
    )
    outpath = os.path.join(DIFFERENCE_DIR, name)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"✅ [REAL] Saved masked-edge-only summary CSV → {outpath}")
    return outpath


def gw_distance(gt: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((gt.shape[0],)) / gt.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = "square_loss"
    dw2 = ot.gromov.gromov_wasserstein2(gt, estimation, p, q, loss_fun, log=False, armijo=False)
    return dw2, np.sqrt(dw2)


def evaluate_and_save_real_init(args, A1_list, reconstructed_list, edge_masks, plot_paths):
    rows = []
    for i, (A_true, A_rec, mask) in enumerate(zip(A1_list, reconstructed_list, edge_masks)):
        A_true_np = A_true.cpu().numpy()
        A_rec_np = A_rec.cpu().numpy()
        mask_np = mask.cpu().numpy()

        n = A_true_np.shape[0]
        iu = np.triu_indices(n, k=1)

        mask_comp = (1.0 - mask_np)
        masked_upper = (mask_comp[iu] == 1)
        num_masked_edges = int(masked_upper.sum())
        if num_masked_edges == 0:
            print(f"[⚠️] Sample {i} has 0 masked upper-triangle edges — skipping metrics")
            continue

        y_true_masked = A_true_np[iu][masked_upper]
        y_hat_masked = A_rec_np[iu][masked_upper]

        mae = mean_absolute_error(y_true_masked, y_hat_masked)
        mse = mean_squared_error(y_true_masked, y_hat_masked)
        frob = np.linalg.norm(y_true_masked - y_hat_masked)

        y_bin = (y_true_masked > 0.5).astype(int)
        yhat_bin = (y_hat_masked > 0.5).astype(int)
        rec = recall_score(y_bin, yhat_bin, zero_division=0)
        f1 = f1_score(y_bin, yhat_bin, zero_division=0)

        try:
            auc = roc_auc_score(y_bin, y_hat_masked)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_bin, y_hat_masked)
        except ValueError:
            ap = float("nan")

        rows.append(
            {
                "sample": i,
                "NumMaskedEdges": num_masked_edges,
                "MAE": mae,
                "MSE": mse,
                "FrobNorm": frob,
                "AP": ap,
                "Rec(0.5)": rec,
                "F1(0.5)": f1,
                "ROC_AUC": auc,
                "PlotPath": plot_paths[i],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[evaluate_init] No rows to save (all samples had 0 masked edges).")
        return None

    avg_row = df.mean(numeric_only=True)
    avg_row["sample"] = "average"
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(DIFFERENCE_DIR, f"A0_init_metrics_{args.name}_{ts}.csv")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"✅ Saved A0 init metrics CSV → {outpath}")
    return outpath


def compute_metric_row_single(
    A_true_t,
    A_rec_t,
    mask_t,
    plot_path,
    score_mode: str,
    variant: str,
    sample_idx: int,
    n_steps: Optional[int],
):
    A_true_np = A_true_t.cpu().numpy()
    A_rec_np = A_rec_t.cpu().numpy()
    mask_np = mask_t.cpu().numpy()

    n = A_true_np.shape[0]
    iu = np.triu_indices(n, k=1)
    masked_upper = (1.0 - mask_np)[iu] == 1
    if not masked_upper.any():
        return None

    y_true_masked = A_true_np[iu][masked_upper]
    y_hat_masked = A_rec_np[iu][masked_upper]

    if score_mode == "auto":
        is_binary = np.all((y_hat_masked == 0) | (y_hat_masked == 1))
        score_mode = "BIN" if is_binary else "RAW"

    mae = mean_absolute_error(y_true_masked, y_hat_masked)
    mse = mean_squared_error(y_true_masked, y_hat_masked)
    frob = np.linalg.norm(y_true_masked - y_hat_masked)

    y_bin = (y_true_masked > 0.5).astype(int)
    yhat_bin = (y_hat_masked > 0.5).astype(int)
    rec = recall_score(y_bin, yhat_bin, zero_division=0)
    f1 = f1_score(y_bin, yhat_bin, zero_division=0)

    try:
        auc = roc_auc_score(y_bin, y_hat_masked)
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(y_bin, y_hat_masked)
    except ValueError:
        ap = float("nan")

    TP = int(np.logical_and(y_bin == 1, yhat_bin == 1).sum())
    FN = int(np.logical_and(y_bin == 1, yhat_bin == 0).sum())
    FP = int(np.logical_and(y_bin == 0, yhat_bin == 1).sum())
    TN = int(np.logical_and(y_bin == 0, yhat_bin == 0).sum())
    fn_denom = FN + TP
    fp_denom = FP + TN
    fn_rate = (FN / fn_denom) if fn_denom > 0 else float("nan")
    fp_rate = (FP / fp_denom) if fp_denom > 0 else float("nan")

    return {
        "sample": sample_idx,
        "variant": variant,
        "PredMode": score_mode,
        "n_steps": n_steps,
        "NumMasked": int(masked_upper.sum()),
        "MAE": mae,
        "MSE": mse,
        "FrobNorm": frob,
        "AP": ap,
        "Rec(0.5)": rec,
        "F1(0.5)": f1,
        "ROC_AUC": auc,
        "FN_rate": fn_rate,
        "FP_rate": fp_rate,
        "PlotPath": plot_path,
    }


def metrics_table_figure_from_row(row: dict, title: str = ""):
    if row is None:
        return None
    cols_pref = [
        "sample",
        "variant",
        "PredMode",
        "n_steps",
        "NumMasked",
        "MAE",
        "MSE",
        "FrobNorm",
        "AP",
        "Rec(0.5)",
        "F1(0.5)",
        "ROC_AUC",
        "FN_rate",
        "FP_rate",
    ]
    cols = [c for c in cols_pref if c in row] + [c for c in row.keys() if c not in cols_pref]
    view = pd.DataFrame([[row.get(c, "") for c in cols]], columns=cols).T.reset_index()
    view.columns = ["Metric", "Value"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18, weight="bold", pad=12)

    tbl = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    base_fs = 12
    tbl.set_fontsize(base_fs)
    tbl.scale(1.4, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(base_fs + 1)

    fig.tight_layout(pad=0.5)
    return fig


def make_pdf_from_dir_with_metric_rows(img_dir: str, out_pdf: str, metric_rows: List[dict], suffixes=(".png", ".jpg", ".jpeg")):
    imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(suffixes))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for p in imgs:
            try:
                img = plt.imread(p)
            except Exception as e:
                print(f"[skip] {p} ({e})")
                continue
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 120.0, h / 120.0), dpi=120)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(img)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for row in metric_rows:
            try:
                p = row.get("PlotPath", "")
                if isinstance(p, str) and os.path.dirname(p) == img_dir:
                    fig = metrics_table_figure_from_row(row, title=f"Metrics for {os.path.basename(p)}")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            except Exception as e:
                print(f"[warn] failed to render metrics page: {e}")

    print(f"📄 PDF saved → {out_pdf}")


def save_five_panel(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    A_step: torch.Tensor,
    outpath: str,
    recon_title: str,
):
    A1 = A_true.detach().cpu().numpy().astype(np.float32)
    M = edge_mask.detach().cpu().numpy().astype(np.float32)
    Arec = A_step.detach().cpu().numpy().astype(np.float32)
    Arec_bin = (Arec > 0.5).astype(np.float32)

    diff_raw = Arec - A1
    diff_bin = Arec_bin - A1

    fig = plt.figure(figsize=(18, 4.2))
    gs = plt.GridSpec(
        1,
        8,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 0.06, 1, 0.06],
        wspace=0.25,
        left=0.03,
        right=0.995,
        bottom=0.07,
        top=0.88,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[0, 4])
    cax4 = fig.add_subplot(gs[0, 5])
    ax5 = fig.add_subplot(gs[0, 6])
    cax5 = fig.add_subplot(gs[0, 7])

    im_kwargs = dict(cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax0.imshow(A1, **im_kwargs)
    ax0.set_title("True Adjacency")
    ax0.axis("off")
    ax1.imshow(M, **im_kwargs)
    ax1.set_title("Edge Mask (kept)")
    ax1.axis("off")
    ax2.imshow(Arec_bin, **im_kwargs)
    ax2.set_title("Binarized Recon")
    ax2.axis("off")
    ax3.imshow(Arec, **im_kwargs)
    ax3.set_title("Raw Recon")
    ax3.axis("off")

    diff_kwargs = dict(cmap="bwr", vmin=-1.0, vmax=+1.0, interpolation="nearest")
    im4 = ax4.imshow(diff_raw, **diff_kwargs)
    ax4.set_title("Raw Δ")
    ax4.axis("off")
    fig.colorbar(im4, cax=cax4)

    im5 = ax5.imshow(diff_bin, **diff_kwargs)
    ax5.set_title("BIN Δ")
    ax5.axis("off")
    fig.colorbar(im5, cax=cax5)

    fig.suptitle(recon_title, fontsize=13, weight="bold")

    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def compute_consensus(y_true_masked, y_hat_masked):
    """Fraction of masked edges predicted correctly after thresholding at 0.5."""
    y_bin = (y_true_masked > 0.5).astype(int)
    yhat_bin = (y_hat_masked > 0.5).astype(int)
    return float(np.mean(y_bin == yhat_bin))


__all__ = [
    "aggregate_last_rows_for_run",
    "posterior_eval_on_val_samples",
    "masked_upper_mse",
    "evaluate_and_save_real",
    "evaluate_and_save_real_init",
    "gw_distance",
    "compute_metric_row_single",
    "metrics_table_figure_from_row",
    "make_pdf_from_dir_with_metric_rows",
    "save_five_panel",
    "compute_consensus",
]
