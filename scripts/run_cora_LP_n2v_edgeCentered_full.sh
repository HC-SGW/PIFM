#!/usr/bin/env bash
# End-to-end training + inference with edge-centered subgraphs (Node2Vec prior, Cora)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

RUN_NAME="CORA_pifm_Node2VecPrior_edgeCentered_k3_noFeat"
EPOCHS=500            # PIFM epochs
LOG_DIR="logs"
TS="$(date +"%Y%m%d_%H%M%S")"
CKPT_DIR="outputs/checkpoints/${TS}_${RUN_NAME}"
LOG_FILE="${LOG_DIR}/${TS}_${RUN_NAME}.log"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

nohup bash -c "
  echo '=== START (Node2Vec prior, edge-centered train+infer) ' \$(date) ' ==='
  echo 'Run: ${RUN_NAME}'
  echo 'Log: ${LOG_FILE}'
  echo 'CKPT_DIR: ${CKPT_DIR}'
  echo 'EPOCHS: ${EPOCHS}'
  echo

  echo '--- [1/2] TRAIN (edge-centered subgraphs, Node2Vec prior) ---'
  python -u main.py --run_name ${RUN_NAME} train_expansion \
    --subgraph_dataset_cfg cfg/dataset_unif.yaml \
    --subgraph_lp \
    --single_graph_path dataset/cora/cora_adj.npy \
    --split_seed 0 \
    --val_ratio 0.05 --test_ratio 0.1 \
    --k_hop 3 --max_nodes 128 --target_coverage 1 \
    --lap_pe_dim 8 --train_edge_drop_p 0.15 \
    --epochs ${EPOCHS} \
    --batch_size 128 \
    --hidden_dim 64 --num_layers 4 --num_linears 2 \
    --c_init 2 --c_hid 16 --c_final 8 \
    --num_heads 4 --conv GCN \
    --lr 2e-4 --seed 0 \
--subgraph_prior node2vec \
    --subgraph_n2v_dim 32 \
    --subgraph_n2v_walk_length 8 \
    --subgraph_n2v_walks_per_node 4 \
    --subgraph_n2v_context_size 4 \
    --subgraph_n2v_epochs 1000 \
    --subgraph_clf_epochs 1000 \
    --subgraph_n2v_batch_size 256 \
    --subgraph_neg_ratio 1.0 \
    --subgraph_clf_lr 1e-2 \
    --subgraph_n2v_device auto \
    --no_feature_adapter \
    --train_edge_centered_subgraphs \
    --node_select_graph train \
    --ckpt_dir ${CKPT_DIR}

  echo
  echo '--- [2/2] INFERENCE (edge-centered subgraphs, Node2Vec prior) ---'
  python -u main.py sample_expansion \
    --subgraph_dataset_cfg cfg/dataset_unif.yaml \
    --subgraph_lp \
    --single_graph_path dataset/cora/cora_adj.npy \
    --split_seed 0 \
    --val_ratio 0.05 --test_ratio 0.1 \
    --k_hop 3 --max_nodes 128 --target_coverage 1 \
    --lap_pe_dim 8 \
    --batch_size 128 \
    --hidden_dim 64 --num_layers 4 --num_linears 2 \
    --c_init 2 --c_hid 16 --c_final 8 \
    --num_heads 4 --conv GCN \
    --n_steps 100 \
    --seed 0 \
--subgraph_prior node2vec \
    --subgraph_n2v_dim 32 \
    --subgraph_n2v_walk_length 8 \
    --subgraph_n2v_walks_per_node 4 \
    --subgraph_n2v_context_size 4 \
    --subgraph_n2v_epochs 1000 \
    --subgraph_clf_epochs 1000 \
    --subgraph_n2v_batch_size 256 \
    --subgraph_neg_ratio 1.0 \
    --subgraph_clf_lr 1e-2 \
    --subgraph_n2v_device auto \
    --no_feature_adapter \
    --test_edge_centered_subgraphs \
    --node_select_graph train \
    --ckpt ${CKPT_DIR}/subgraph_epoch\$(printf '%04d' ${EPOCHS}).pt

  echo
  echo '=== END (Node2Vec prior, edge-centered train+infer) ' \$(date) ' ==='
" > "$LOG_FILE" 2>&1 &

echo "Launched Node2Vec edge-centered train+infer run. PID: $!"
echo "Tail logs with:"
echo "  tail -f \"$LOG_FILE\""
