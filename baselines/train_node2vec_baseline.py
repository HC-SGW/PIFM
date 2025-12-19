#!/usr/bin/env python3
"""
GPU-enabled per-graph Node2Vec link prediction + reconstruction evaluation.

NEW:
- --data_root <run_folder> will automatically find:
    <run_folder>/masks_drop-<drop_rate>/{train,val,test}_graphs.pkl
    <run_folder>/masks_drop-<drop_rate>/{train,val,test}_masks.pkl
  and run ALL THREE splits in one run (train, val, test).
- Prints the three RAW reconstruction folders at the end.
- Backwards compatible: you can still pass --adj_file/--mask_file to run one set.

Saves per-graph reconstructions as .npy files under:
  <OUTPUT_ROOT>/output_inter/n2vdiff_priors/N2V_<name>_<drop>drop_<ts>/<SPLIT>/graphs_npy/{raw,binary,true,mask}
which defaults to <repo>/outputs unless overridden via the PIFM_OUTPUT_ROOT env var.
"""

import argparse
import json
import random
import os
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.getenv("PIFM_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "outputs"))
OUTPUT_INTER_DIR = os.path.join(OUTPUT_ROOT, "output_inter")
DIFFERENCE_DIR = os.path.join(OUTPUT_INTER_DIR, "difference_inter")
N2V_PRIOR_ROOT = os.path.join(OUTPUT_INTER_DIR, "n2vdiff_priors")

import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def load_matrix(path):
    path = Path(path)
    if path.suffix in (".npz", ".npy"):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = list(arr.keys())
            if len(keys) == 1:
                return arr[keys[0]]
            for k in ("adj", "mask", "array"):
                if k in arr:
                    return arr[k]
            raise ValueError(f"Ambiguous .npz file {path}")
        return arr
    else:
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)

        # Single Graph
        if isinstance(obj, nx.Graph):
            return [nx.to_numpy_array(obj, dtype=int)]

        # List of Graphs
        if isinstance(obj, list) and all(isinstance(g, nx.Graph) for g in obj):
            return [nx.to_numpy_array(g, dtype=int) for g in obj]

        # Already arrays
        if isinstance(obj, list) and all(hasattr(g, "shape") for g in obj):
            return [np.array(g) for g in obj]

        return obj


def sample_neg(adj: np.ndarray, num: int, seed: int=0):
    np.random.seed(seed)
    n = adj.shape[0]
    possible_negatives = (n * (n - 1) // 2) - (np.count_nonzero(adj) // 2)
    if num > possible_negatives:
        num = possible_negatives

    neg = set()
    while len(neg) < num:
        u, v = np.random.randint(0, n), np.random.randint(0, n)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if adj[a, b] != 0:
            continue
        neg.add((a, b))
    return list(neg)

def sample_neg_by_region(A: np.ndarray, M: np.ndarray, num: int, *, region="observed", seed: int = 0):
    """
    region = "observed" -> draw negatives from zeros where M==1 (training)
    region = "masked"   -> draw negatives from zeros where M==0 (val/test)
    Returns list of (i,j) with i<j.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    target = 1 if region == "observed" else 0

    cand = [(i, j)
            for i in range(n) for j in range(i+1, n)
            if (M[i, j] == target) and (A[i, j] == 0)]
    if not cand:
        return []
    num = min(num, len(cand))
    picks = rng.choice(len(cand), size=num, replace=False)
    return [cand[k] for k in picks]



def split_masked(edges, frac: float):
    random.shuffle(edges)
    k = int(len(edges)*frac)
    return edges[:k], edges[k:]


# ---- Evaluate only on upper-triangle masked entries ----
def evaluate_and_save_real(args, A_true_list, A_rec_list, masks, plot_paths, st):
    rows=[]
    for i,(A_true,A_rec,mask,plot) in enumerate(zip(A_true_list,A_rec_list,masks,plot_paths)):
        A_true_np = np.array(A_true); A_rec_np = np.array(A_rec); mask_np = np.array(mask)
        n = A_true_np.shape[0]
        iu = np.triu_indices(n, k=1)
        mask_comp = (1.0 - mask_np)
        masked_upper = (mask_comp[iu] == 1)
        num_masked = int(masked_upper.sum())
        if num_masked == 0:
            print(f"[⚠️] Sample {i} has 0 masked upper-triangle edges — skipping metrics")
            continue

        y_t = A_true_np[iu][masked_upper]
        y_h = A_rec_np[iu][masked_upper]

        mae = mean_absolute_error(y_t,y_h)
        mse = mean_squared_error(y_t,y_h)
        frob= np.linalg.norm(y_t-y_h)
        yb=(y_t>0.5).astype(int); yhb=(y_h>0.5).astype(int)
        prec=precision_score(yb,yhb,zero_division=0)
        rec= recall_score(yb,yhb,zero_division=0)
        f1 = f1_score(yb,yhb,zero_division=0)
        try: auc=roc_auc_score(yb,y_h)
        except: auc=float('nan')
        fn=int(((yb==1)&(yhb==0)).sum())
        fp=int(((yb==0)&(yhb==1)).sum())
        rows.append({
            'sample':i,'NumMasked':num_masked,
            'MAE':mae,'MSE':mse,'FrobNorm':frob,
            'Prec(0.5)':prec,'Rec(0.5)':rec,'F1(0.5)':f1,
            'AUC':auc,'FN_count':fn,'FP_count':fp,
            'PlotPath':plot,
        })
    if not rows:
        print("[evaluate] No rows produced; skipping CSV.")
        return None

    df = pd.DataFrame(rows)
    avg = df.mean(numeric_only=True)
    avg['sample']='average'
    df = pd.concat([df, avg.to_frame().T], ignore_index=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{st}_{args.name}_REAL_{args.n2v_epochs}ep_{args.dimensions}dim_{args.walk_length}wl_{args.walks_per_node}wn_{ts}.csv"
    out = Path(DIFFERENCE_DIR)/name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out,index=False)
    print(f"Saved REAL-only summary CSV → {out}")
    return out

# ---------------------------
# Split runner
# ---------------------------

def run_one_split(split_name: str,
                  adjs_list,
                  masks_list,
                  args,
                  device,
                  global_run_root: Path,
                  prefix_base: str,
                  drop_rate_str: str):
    """
    Run the full training+reconstruction on a (graphs, masks) list for a given split.
    Returns the RAW recon folder path.
    """
    # Per-split run folder
    run_dir = global_run_root / split_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Recon subfolders
    recon_dir = run_dir / "graphs_npy"
    raw_dir   = recon_dir / "raw"
    bin_dir   = recon_dir / "binary"
    true_dir  = recon_dir / "true"
    mask_dir  = recon_dir / "mask"
    for d in (recon_dir, raw_dir, bin_dir, true_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{split_name}] Saving reconstructions under: {recon_dir}")
    print(f"    raw   → {raw_dir}")
    print(f"    binary→ {bin_dir}")
    print(f"    true  → {true_dir}")
    print(f"    mask  → {mask_dir}")

    base_out = Path(args.output_dir); base_out.mkdir(parents=True, exist_ok=True)
    A_true_list=[]; A_rec_list=[]; mask_list=[]; plot_paths=[]

    # Loop graphs
    for idx,(A_raw,M_raw) in enumerate(zip(adjs_list,masks_list)):
        A=np.array(A_raw); M=np.array(M_raw)
        n=A.shape[0]
        st = f"{args.name}_{split_name}_g{idx}_drop{drop_rate_str}"
        out_dir = base_out / st
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{split_name} | Graph {idx}] n={n}")

        # Build observed graph
        G=nx.Graph(); G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1,n):
                if M[i,j] and A[i,j]!=0: G.add_edge(i,j)

        # Adjust walk params
        walk_length = min(args.walk_length, max(2, n - 1))
        walks_per_node = max(1, args.walks_per_node)
        context_size = min(args.context_size, walk_length)

        data=from_networkx(G); edge_index=data.edge_index.to(device)

        # Node2Vec
        n2v = Node2Vec(
            edge_index,
            embedding_dim=args.dimensions,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=args.p,
            q=args.q,
            num_nodes=n,
            sparse=True
        ).to(device)
        loader=n2v.loader(batch_size=128,shuffle=True)
        opt_n2v=torch.optim.SparseAdam(list(n2v.parameters()),lr=0.01)
        n2v.train()
        for e in range(args.n2v_epochs):
            total = 0
            for pos, neg in loader:
                opt_n2v.zero_grad()
                loss = n2v.loss(pos.to(device), neg.to(device)).mean()
                loss.backward()
                opt_n2v.step()
                total += loss.item()
            if e%100==0:
                print(f"  [N2V] epo {e}/{args.n2v_epochs} loss={total/len(loader):.4f}")

        n2v.eval()
        with torch.no_grad():
            Z=n2v.embedding.weight.cpu().numpy()
        print(f"[{split_name} | Graph {idx}] Node2Vec done. Building pos/masked sets…")

        # Edges
        pos=[]; masked=[]
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j]!=0:
                    (pos if M[i,j] else masked).append((i,j))
        val_pos, test_pos = split_masked(masked, args.val_frac)

        # FIX: no label leakage during training; align val/test with masked region
        neg_tr  = sample_neg_by_region(A, M, len(pos),      region="observed", seed=idx)
        neg_val = sample_neg_by_region(A, M, len(val_pos),  region="masked",   seed=idx+100)
        neg_te  = sample_neg_by_region(A, M, len(test_pos), region="masked",   seed=idx+200)
        
        def _assert_region(pairs, M, target):
            for (i,j) in pairs:
                assert M[i,j] == target, f"Negative ({i},{j}) not in expected region M=={target}"

        _assert_region(neg_tr,  M, 1)  # observed
        _assert_region(neg_val, M, 0)  # masked
        _assert_region(neg_te,  M, 0)  # masked



        # Edge features
        def mk_X(pos, neg):
            if len(pos) == 0 and len(neg) == 0:
                return np.empty((0, Z.shape[1])), np.array([])
            feats=[]
            for u,v in pos: feats.append(Z[u]*Z[v])
            labs=[1]*len(pos)
            for u,v in neg:
                feats.append(Z[u]*Z[v])
                labs.append(0)
            return np.vstack(feats).astype(np.float32), np.array(labs, dtype=np.float32)

        Xtr,ytr=mk_X(pos,neg_tr)
        Xval,yval=mk_X(val_pos,neg_val)
        Xte,yte  =mk_X(test_pos,neg_te)

        # Classifier
        clf=LinkPredictor(args.dimensions).to(device)
        opt_c=torch.optim.Adam(clf.parameters(),lr=0.01); bce=nn.BCELoss()
        Xtr_t=torch.tensor(Xtr,device=device,dtype=torch.float32); ytr_t=torch.tensor(ytr,device=device,dtype=torch.float32)
        for e in range(args.clf_epochs):
            clf.train(); opt_c.zero_grad()
            pr=clf(Xtr_t)
            loss=bce(pr,ytr_t)
            loss.backward(); opt_c.step()
            if e%100==0: print(f"  [CLF] epo {e}/{args.clf_epochs} loss={loss.item():.4f}")

        # Full reconstruction
        w = clf.linear.weight.detach().cpu().numpy().reshape(-1)  # (D,)
        b = float(clf.linear.bias.detach().cpu())
        T = Z * w
        S = T @ Z.T
        A_full = 1.0 / (1.0 + np.exp(-(S + b)))
        np.fill_diagonal(A_full, 0.0)

        # keep scores only on masked entries
        A_rec = np.zeros_like(A_full, dtype=float)
        mask_comp = (1 - M).astype(bool)
        iu = np.triu_indices(n, 1)
        sel = mask_comp[iu]
        A_rec[iu[0][sel], iu[1][sel]] = A_full[iu[0][sel], iu[1][sel]]
        A_rec = A_rec + A_rec.T

        # Save
        prefix_split = f"{prefix_base}_{split_name}"
        raw_name = f"{prefix_split}_g{idx}_recon.npy"
        raw_path = raw_dir / raw_name
        np.save(raw_path, A_rec)
        print(f"✔️  [{split_name}] Saved raw reconstruction {idx} → {raw_path}")

        A_binary = (A_rec > 0.5).astype(np.float32)
        bin_name = f"{prefix_split}_g{idx}_recon_binary.npy"
        bin_path = bin_dir / bin_name
        np.save(bin_path, A_binary)

        true_name = f"{prefix_split}_g{idx}_true.npy"
        # keep npy for true & mask (consistent with others)
        np.save(true_dir / true_name, A)
        np.save(mask_dir / f"{prefix_split}_g{idx}_mask.npy", M)

        # Collect for REAL eval
        A_true_list.append(A); A_rec_list.append(A_rec)
        mask_list.append(M); plot_paths.append("")

        # Optional: per-graph metrics on train/val/test classification sets
        per_metrics={'train':{},'val':{},'test':{}}
        for tag,(X,y) in zip(['train','val','test'],[(Xtr,ytr),(Xval,yval),(Xte,yte)]):
            if y.size>0:
                p=clf(torch.tensor(X,device=device)).detach().cpu().numpy()
                per_metrics[tag]=dict(
                    roc_auc=roc_auc_score(y,p) if len(np.unique(y))>1 else None,
                    ap=average_precision_score(y,p) if len(np.unique(y))>1 else None
                )
        mpath=(base_out/f"{prefix_split}_g{idx}_metrics.json")
        with open(mpath,"w") as f:
            json.dump(per_metrics, f, indent=2)

    # Per-split REAL evaluation CSV
    evaluate_and_save_real(args, A_true_list, A_rec_list, mask_list, plot_paths, st=f"{args.name}_{split_name}")

    # Save a small index for the split
    idx_path = run_dir / f"{prefix_base}_{split_name}_save_index.json"
    with open(idx_path, "w") as f:
        json.dump({
            "run_dir": str(run_dir),
            "raw_dir": str(raw_dir),
            "binary_dir": str(bin_dir),
            "true_dir": str(true_dir),
            "mask_dir": str(mask_dir),
            "split": split_name
        }, f, indent=2)

    return str(raw_dir)

def _drop_variants_str(drop_rate):
    """
    Generate robust string variants for a drop rate, preserving minus signs.
    Examples for -1.0: {"-1.0", "-1"}
             for  1.0: {"1.0", "1"}
    """
    variants = set()
    # exact CLI representation first (preserve minus sign & decimals if present)
    variants.add(str(drop_rate))

    # float-normalized variants (if convertible)
    try:
        f = float(drop_rate)
        variants.add(f"{f}")          # e.g., -1.0
        variants.add(f"{f:.1f}")      # e.g., -1.0
        variants.add(f"{f:.2f}")      # e.g., -1.00
        if f.is_integer():
            variants.add(f"{int(f)}") # e.g., -1
    except Exception:
        pass

    # strip trailing zeros for the decimal variants, but NEVER strip the minus
    cleaned = set()
    for s in variants:
        if "." in s:
            # keep sign, strip trailing zeros & dot after the numeric part
            sign = "-" if s.startswith("-") else ""
            body = s[1:] if sign else s
            body = body.rstrip("0").rstrip(".") or "0"
            cleaned.add(sign + body)
        else:
            cleaned.add(s)
    # Keep originals too
    return list(variants | cleaned)


from pathlib import Path

def find_data_dir(base_dir: Path, drop_rate):
    """
    Accept either:
      base_dir / 'masks_drop<drop>'      (preferred; no hyphen)
      base_dir / 'masks_drop-<drop>'     (legacy; with hyphen)
    or base_dir already == that masks folder.

    If no exact match, falls back to the only subdir that startswith 'masks_drop'.
    """
    base_dir = Path(base_dir)

    # If user already points to a masks folder, trust it
    if base_dir.is_dir() and base_dir.name.startswith("masks_drop"):
        return base_dir

    # Try explicit variants
    for v in _drop_variants_str(drop_rate):
        for pat in (f"masks_drop{v}", f"masks_drop-{v}"):
            cand = base_dir / pat
            if cand.is_dir():
                return cand

    # Fallback: auto-pick if exactly one masks_drop* exists
    subdirs = [d for d in base_dir.iterdir()
               if d.is_dir() and d.name.startswith("masks_drop")]
    if len(subdirs) == 1:
        print(f"[auto] Using the only masks_drop* directory found: {subdirs[0]}")
        return subdirs[0]

    # Helpful error
    found = [d.name for d in subdirs]
    raise FileNotFoundError(
        f"Could not locate masks_drop<drop> under {base_dir} for drop={drop_rate}. "
        f"Checked variants={_drop_variants_str(drop_rate)} with and without hyphen. "
        f"Found candidates={found}."
    )



# --- leave find_data_dir() as you have it ---

def _pick_file(dir_path: Path, split: str, kind: str) -> Path:
    """
    Find one file for {split}_{kind} in dir_path. Prefer .pkl, then .npz, then .npy.
    """
    dir_path = Path(dir_path).resolve()
    preferred = dir_path / f"{split}_{kind}.pkl"
    if preferred.exists():
        return preferred

    # Try alternates
    candidates = sorted(dir_path.glob(f"{split}_{kind}.*"))
    if candidates:
        ext_order = [".pkl", ".npz", ".npy"]
        for ext in ext_order:
            for p in candidates:
                if p.suffix.lower() == ext:
                    print(f"[auto] Using {p.name} in {dir_path} for {split}_{kind}")
                    return p
        print(f"[auto] Using {candidates[0].name} in {dir_path} for {split}_{kind}")
        return candidates[0]

    # Not found → print a short directory listing
    listing = "\n".join(f"  - {p.name}" for p in sorted(dir_path.iterdir()))
    raise FileNotFoundError(
        f"Missing {split}_{kind}.pkl|npz|npy in {dir_path}.\n"
        f"Directory listing:\n{listing}"
    )

def load_split_lists_from_root(data_root: Path, drop_rate: float):
    """
    Expected structure (always respected):
      <run_dir>/
        train_graphs.pkl   val_graphs.pkl   test_graphs.pkl
        masks_drop<rate>/
          train_masks.pkl  val_masks.pkl    test_masks.pkl
    (Also supports legacy 'masks_drop-<rate>' and if you pass the masks folder directly.)
    """
    # Locate the masks directory first (accepts base as <run_dir> or the masks folder itself)
    masks_dir = find_data_dir(data_root, drop_rate)   # e.g., <run_dir>/masks_drop0.5
    graphs_dir = masks_dir.parent                     # graphs live ONE LEVEL ABOVE masks

    print(f"[auto] graphs dir → {graphs_dir}")
    print(f"[auto] masks  dir → {masks_dir}")

    out = {}
    for split in ("train", "val", "test"):
        # graphs come from the parent folder; masks come from masks_drop<rate>
        adj_p  = _pick_file(graphs_dir, split, "graphs")
        mask_p = _pick_file(masks_dir,  split, "masks")
        print(f"[{split}] graphs → {adj_p}")
        print(f"[{split}] masks  → {mask_p}")
        out[split] = (load_matrix(adj_p), load_matrix(mask_p))

    # Return the top-level run directory for logging
    return out, graphs_dir



# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU-per-graph Node2Vec link prediction + REAL eval")
    # Mode A: single set (backwards compatible)
    parser.add_argument('--adj_file',    help="List of graphs (.pkl/.npy/.npz)")
    parser.add_argument('--mask_file',   help="List of masks (.pkl/.npy/.npz)")
    # Mode B: auto all splits
    parser.add_argument('--data_root',   help="Folder that contains masks_drop-<drop>/… files (or the masks_drop-<drop> folder itself)")
    # Shared
    parser.add_argument('--name',        required=True)
    parser.add_argument('--drop_rate',   type=float, required=True)
    parser.add_argument('--p',           type=float, default=1.0)
    parser.add_argument('--q',           type=float, default=1.0)
    parser.add_argument('--dimensions',  type=int,   default=64)
    parser.add_argument('--walk_length', type=int,   default=30)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--context_size',   type=int, default=10)
    parser.add_argument('--n2v_epochs',   type=int,   default=1000)
    parser.add_argument('--clf_epochs',   type=int,   default=1000)
    parser.add_argument('--val_frac',     type=float, default=0.5)
    parser.add_argument('--output_dir',   default='results_per_graph_gpu')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Device: {device}")

    # Timestamp + global run root
    ts = time.strftime("%Y%m%d_%H%M%S")
    drop_rate_str = str(args.drop_rate)
    prefix_base = f"N2V_{args.name}_{drop_rate_str}drop_{ts}"
    global_run_root = Path(N2V_PRIOR_ROOT) / prefix_base
    global_run_root.mkdir(parents=True, exist_ok=True)

    # If --data_root is provided: run ALL splits (train/val/test)
    raw_dirs_summary = {}
    if args.data_root:
        data_root = Path(args.data_root)
        split_map, data_dir_used = load_split_lists_from_root(data_root, args.drop_rate)
        print(f"[auto] Using data from: {data_dir_used}")

        for split in ("train","val","test"):
            adjs_list, masks_list = split_map[split]
            if not isinstance(adjs_list, list) or not isinstance(masks_list, list):
                raise ValueError(f"[{split}] Expect lists of graphs/masks")
            if len(adjs_list) != len(masks_list):
                raise ValueError(f"[{split}] Length mismatch between graphs and masks")
            raw_dir = run_one_split(split, adjs_list, masks_list,
                                    args, device, global_run_root, prefix_base, drop_rate_str)
            raw_dirs_summary[split] = raw_dir

    else:
        # Backwards compatible single-set mode
        if not args.adj_file or not args.mask_file:
            raise ValueError("Provide either --data_root OR both --adj_file and --mask_file.")
        adjs  = load_matrix(args.adj_file)
        masks = load_matrix(args.mask_file)
        if not isinstance(adjs, list) or not isinstance(masks, list):
            raise ValueError("Expect lists of graphs and masks")
        if len(adjs) != len(masks):
            raise ValueError("Length mismatch between graphs and masks")

        raw_dir = run_one_split("custom", adjs, masks,
                                args, device, global_run_root, prefix_base, drop_rate_str)
        raw_dirs_summary["custom"] = raw_dir

    # --------------- Final summary print ---------------
    if raw_dirs_summary:
        print("\n==================== SUMMARY ====================")
        print("RAW reconstruction folders (collect these):")
        for split, p in raw_dirs_summary.items():
            print(f"  {split:>5}: {p}")
        print("=================================================\n")

        # Save a top-level index too
        with open(global_run_root / f"{prefix_base}_all_splits_index.json", "w") as f:
            json.dump({
                "run_root": str(global_run_root),
                "raw_dirs": raw_dirs_summary,
                "timestamp": ts,
                "experiment_name": args.name,
                "drop_rate": args.drop_rate
            }, f, indent=2)


if __name__=='__main__':
    main()
