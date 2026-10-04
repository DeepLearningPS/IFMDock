#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPUS=${IFMDOCK_GPUS:-4,5,6,7}
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
python -c 'import lightning, torch_geometric' >/dev/null || {
  echo "Activate the IFMDock2 environment before training." >&2; exit 2; }
CUDA_VISIBLE_DEVICES="$GPUS" python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  scripts/train/train_config.py configs/stage2_train.yaml "$@"
