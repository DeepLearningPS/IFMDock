#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

usage() {
  cat <<'EOF'
用法：
  run_pipeline.sh DATASET_DIR [OUTPUT_DIR] [选项]

未指定 OUTPUT_DIR 时，输出到项目根目录的 output/。

选项：
  --gpus GPU_CSV                 GPU 编号，例如 0,1,2,3（默认：0,1,2,3）
  --distance-candidate-selection MODE
                                 距离矩阵选择：model_default（默认）或 balanced_random
  --rdkit-initial-pose            使用配对RDKit坐标定位距离缓存；模型1仍从高斯分布采样
  --no-rdkit-initial-pose         不使用配对RDKit坐标
  --rdkit-initial-pose-root DIR   使用与距离缓存相同的RDKit坐标侧车目录
  -h, --help                     显示帮助

EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then usage; exit 0; fi
DATA=${1:?缺少 DATASET_DIR；使用 --help 查看用法}
shift
OUT="$ROOT/output"
if (($#)) && [[ "$1" != -* ]]; then
  OUT=$1
  shift
fi

POSITIONAL=()
GPUS=""
DISTANCE_CANDIDATE_SELECTION="model_default"
RDKIT_INITIAL_POSE="true"
RDKIT_INITIAL_POSE_ROOT=""
RDKIT_INITIAL_POSE_MODE="paired"
while (($#)); do
  case "$1" in
    --gpus) GPUS=${2:?--gpus 缺少参数}; shift 2 ;;
    --distance-candidate-selection) DISTANCE_CANDIDATE_SELECTION=${2:?参数缺失}; shift 2 ;;
    --rdkit-initial-pose) RDKIT_INITIAL_POSE="true"; shift ;;
    --no-rdkit-initial-pose) RDKIT_INITIAL_POSE="false"; shift ;;
    --rdkit-initial-pose-mode) RDKIT_INITIAL_POSE_MODE=${2:?参数缺失}; shift 2 ;;
    --rdkit-initial-pose-root) RDKIT_INITIAL_POSE_ROOT=${2:?参数缺失}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; while (($#)); do POSITIONAL+=("$1"); shift; done ;;
    -*) echo "未知选项：$1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

GPUS=${GPUS:-${POSITIONAL[0]:-0,1,2,3}}
IFMSCORE_CKPT=${IFMDOCK_IFMSCORE_CKPT:-$ROOT/IFMScore/trained_models/ifmscore.pth}
WORKERS=${IFMDOCK_PB_WORKERS:-24}
PB_TIMEOUT=${IFMDOCK_PB_TIMEOUT:-120}
LIMIT=${IFMDOCK_LIMIT:-0}
SAMPLES=${IFMDOCK_SAMPLES:-40}
STEPS=${IFMDOCK_STEPS:-15}
STAGE1_EPOCH=${IFMDOCK_STAGE1_EPOCH:-200}
STAGE2_EPOCH=${IFMDOCK_STAGE2_EPOCH:-40}
MODEL1_CKPT=${IFMDOCK_MODEL1_CKPT:-$ROOT/checkpoints/model1.pt}
MODEL2_CKPT=${IFMDOCK_MODEL2_CKPT:-$ROOT/checkpoints/model2.pt}
MODEL1_CONFIG=${IFMDOCK_MODEL1_CONFIG:-$ROOT/checkpoints/model1.yml}
MODEL2_CONFIG=${IFMDOCK_MODEL2_CONFIG:-$ROOT/checkpoints/model2.yml}
python - "$MODEL1_CKPT" "$MODEL1_CONFIG" "$STAGE1_EPOCH" \
  "$MODEL2_CKPT" "$MODEL2_CONFIG" "$STAGE2_EPOCH" <<'PY'
import sys
import torch
from omegaconf import OmegaConf

for model_number, offset, source_mode in ((1, 1, "gaussian"), (2, 4, "stage2_pose")):
    checkpoint, config_path, expected_epoch = sys.argv[offset:offset + 3]
    config = OmegaConf.load(config_path)
    actual_mode = config.transforms.get("cartesian_source_mode", "stage2_pose")
    if not config.transforms.get("cartesian_flow", False) or not config.model.get("cartesian_refinement", False):
        raise SystemExit(f"model{model_number}: checkpoint configuration is not Cartesian flow")
    if actual_mode != source_mode:
        raise SystemExit(f"model{model_number}: expected Cartesian source {source_mode}, got {actual_mode}")
    actual_epoch = torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch")
    if actual_epoch != int(expected_epoch):
        raise SystemExit(f"model{model_number}: expected epoch {expected_epoch}, got {actual_epoch}")
PY
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export IFMDOCK_DISTANCE_MODEL_ROOT="$ROOT/ifmdock/utils/third_party/distance_model"

# Stage 1: Cartesian Gaussian flow samples, 15 integration steps.
STAGE1_EXTRA=()
if [[ "$RDKIT_INITIAL_POSE" == "true" ]]; then
  STAGE1_EXTRA+=(--rdkit-initial-pose --rdkit-initial-pose-mode "$RDKIT_INITIAL_POSE_MODE")
else
  STAGE1_EXTRA+=(--no-rdkit-initial-pose)
fi
if [[ -n "$RDKIT_INITIAL_POSE_ROOT" ]]; then
  STAGE1_EXTRA+=(--rdkit-initial-pose-root "$RDKIT_INITIAL_POSE_ROOT")
fi
python scripts/inference/evaluate_rigid_pocket_posebusters.py \
  --epoch "$STAGE1_EPOCH" --run-dir "$ROOT/checkpoints" --checkpoint "$MODEL1_CKPT" \
  --model-config "$MODEL1_CONFIG" \
  --data-dir "$DATA" --output-root "$OUT/stage1" --samples-per-complex "$SAMPLES" \
  --inference-steps "$STEPS" --batch-size 8 --gpu-ids "$GPUS" \
  --limit-complexes "$LIMIT" --distance-candidate-selection "$DISTANCE_CANDIDATE_SELECTION" \
  "${STAGE1_EXTRA[@]}" --no-update-best-model --force
S1=$(printf '%s/stage1/epoch_%04d' "$OUT" "$STAGE1_EPOCH")

# Stage 2: Cartesian local flow.  Quick geometry/contact checks are recorded by
# the refiner but do not replace Stage-2 coordinates unless the legacy opt-in
# flag --quick-physical-fallback is passed directly to that script.
S2=$(printf '%s/stage2/epoch_%04d' "$OUT" "$STAGE2_EPOCH")
mkdir -p "$S2/predictions" "$S2/model"
cp "$S1/inputs.csv" "$S2/inputs.csv"
cp "$MODEL2_CKPT" "$S2/model/checkpoint.pt"
cp "$MODEL2_CONFIG" "$S2/model/model_parameters.yml"
IFS=',' read -ra GPU_ARRAY <<<"$GPUS"
PIDS=()
for SHARD in "${!GPU_ARRAY[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$SHARD]} python scripts/inference/refine_posebusters.py \
    --inputs "$S2/inputs.csv" --initial-root "$S1/predictions" \
    --output-root "$S2/predictions" --model-dir "$S2/model" \
    --checkpoint "$S2/model/checkpoint.pt" --steps "$STEPS" --pose-batch-size 8 \
    --distance-candidate-selection "$DISTANCE_CANDIDATE_SELECTION" \
    --no-physical-fallback \
    ${RDKIT_INITIAL_POSE_ROOT:+--rdkit-initial-pose-root "$RDKIT_INITIAL_POSE_ROOT"} \
    --shard-index "$SHARD" --num-shards "${#GPU_ARRAY[@]}" \
    >"$S2/refine_gpu_${GPU_ARRAY[$SHARD]}.log" 2>&1 & PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
python scripts/inference/evaluate_rigid_pocket_posebusters.py \
  --epoch "$STAGE2_EPOCH" --run-dir "$ROOT/checkpoints" --checkpoint "$MODEL2_CKPT" \
  --data-dir "$DATA" --output-root "$OUT/stage2" --samples-per-complex "$SAMPLES" \
  --inference-steps "$STEPS" --limit-complexes "$LIMIT" --export-only --no-update-best-model --force

# Rank the unmodified Stage-2 candidates. Shards share restartable JSON outputs.
PIDS=()
for SHARD in "${!GPU_ARRAY[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$SHARD]} python scripts/evaluation/evaluate_posebusters_ifmscore.py score \
    --epoch-dir "$S2" --data-dir "$DATA" \
    --checkpoint "$IFMSCORE_CKPT" --shard-index "$SHARD" --num-shards "${#GPU_ARRAY[@]}" \
    >"$S2/ifmscore_gpu_${GPU_ARRAY[$SHARD]}.log" 2>&1 & PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
python scripts/inference/export_ranked_poses.py \
  --epoch-dir "$S2" --data-dir "$DATA" --output-dir "$OUT"
# Final physical metrics require only the two reported selections: IFMScore
# Top-1 and RMSD Best-1; these checks report validity without changing poses.
python scripts/evaluation/evaluate_posebusters_ifmscore.py finalize \
  --epoch-dir "$S2" --data-dir "$DATA" --workers "$WORKERS" \
  --check-best-and-ifmscore-top1 --force-posebusters \
  --posebusters-timeout "$PB_TIMEOUT"
cp "$S2/full_evaluation_summary.json" "$S2/full_evaluation_per_complex.json" \
  "$S2/full_evaluation_per_complex.csv" "$OUT/"
echo "Final metrics: $S2/full_evaluation_summary.json"
