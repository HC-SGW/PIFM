#!/usr/bin/env bash
# GraphSAGE-HEART prior run for Cora, edge-centered k=1 (feature adapter ON)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

RUN_NAME="CORA_pifm_GraphSAGEHeartPrior_edgeCentered_k1"
EPOCHS=500             # PIFM epochs
HEART_EPOCHS=1000      # GraphSAGE-HEART epochs (full-graph pretrain)
LOG_DIR="logs"
TS="$(date +"%Y%m%d_%H%M%S")"
CKPT_DIR="outputs/checkpoints/${TS}_${RUN_NAME}"
LOG_FILE="${LOG_DIR}/${TS}_${RUN_NAME}.log"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

nohup bash -c "
  echo '=== START (GraphSAGE-HEART prior, edge-centered k=1) ' \$(date) ' ==='
  echo 'Run: ${RUN_NAME}'
  echo 'Log: ${LOG_FILE}'
  echo 'CKPT_DIR: ${CKPT_DIR}'
  echo 'EPOCHS: ${EPOCHS}'
  echo

  echo '--- [1/2] TRAIN (edge-centered subgraphs) ----------------------'
  python -u main.py --run_name ${RUN_NAME} train_expansion \
    --subgraph_dataset_cfg cfg/dataset_unif.yaml \
    --subgraph_lp \
    --single_graph_path dataset/cora/cora_adj.npy \
    --split_seed 0 \
    --val_ratio 0.05 --test_ratio 0.1 \
    --k_hop 1 --max_nodes 96 --target_coverage 1 \
    --lap_pe_dim 8 --train_edge_drop_p 0.15 \
    --epochs ${EPOCHS} \
    --batch_size 128 \
    --hidden_dim 64 --num_layers 4 --num_linears 2 \
    --c_init 2 --c_hid 16 --c_final 8 \
    --num_heads 4 --conv GCN \
    --lr 2e-4 --seed 0 \
    --subgraph_prior graphsage_heart \
    --graphsage_heart_dim 128 \
    --graphsage_heart_hidden_dim 64 \
    --graphsage_heart_layers 2 \
    --graphsage_heart_epochs ${HEART_EPOCHS} \
    --graphsage_heart_lr 1e-4 \
    --graphsage_heart_neg_ratio 1.0 \
    --graphsage_heart_dropout 0.5 \
    --subgraph_sage_heart_weight_decay 1e-4 \
    --graphsage_heart_device cuda \
    --train_edge_centered_subgraphs \
    --node_select_graph train \
    --ckpt_dir ${CKPT_DIR}

  echo
  echo '--- [2/2] INFERENCE (edge-centered subgraphs) --------'
  python -u main.py sample_expansion \
    --subgraph_dataset_cfg cfg/dataset_unif.yaml \
    --subgraph_lp \
    --single_graph_path dataset/cora/cora_adj.npy \
    --split_seed 0 \
    --val_ratio 0.05 --test_ratio 0.1 \
    --k_hop 1 --max_nodes 96 --target_coverage 1 \
    --lap_pe_dim 8 \
    --batch_size 128 \
    --hidden_dim 64 --num_layers 4 --num_linears 2 \
    --c_init 2 --c_hid 16 --c_final 8 \
    --num_heads 4 --conv GCN \
    --n_steps 100 \
    --seed 0 \
    --subgraph_prior graphsage_heart \
    --graphsage_heart_dim 128 \
    --graphsage_heart_hidden_dim 64 \
    --graphsage_heart_layers 2 \
    --graphsage_heart_epochs ${HEART_EPOCHS} \
    --graphsage_heart_lr 1e-4 \
    --graphsage_heart_neg_ratio 1.0 \
    --graphsage_heart_dropout 0.5 \
    --subgraph_sage_heart_weight_decay 1e-4 \
    --graphsage_heart_device cuda \
    --test_edge_centered_subgraphs \
    --node_select_graph train \
    --ckpt ${CKPT_DIR}/subgraph_epoch\$(printf '%04d' ${EPOCHS}).pt

  echo
  echo '=== END (GraphSAGE-HEART prior, edge-centered k=1) ' \$(date) ' ==='
" > "$LOG_FILE" 2>&1 &

echo "Launched edge-centered k=1 run. PID: $!"
echo "Tail logs with:"
echo "  tail -f \"$LOG_FILE\""
