#!/usr/bin/env bash
set -euo pipefail

# pipeline.sh - converted from pipeline.ipynb
# Usage: ./pipeline.sh      # runs full pipeline with defaults
#        AUG_TYPE=TPG ./pipeline.sh   # override variables via env

PYTHON=${PYTHON:-python3}

# --- configuration (defaults taken from notebook) ---
aug_type=${AUG_TYPE:-TPGBIR}
src_dir=${SRC_DIR:-hybrid}
label_file=${LABEL_FILE:-"${src_dir}.csv"}
dist_dir=${DIST_DIR:-"${src_dir}_${aug_type}"}
n_augmentations=${N_AUGMENTATIONS:-1}

# model settings
model_state=${MODEL_STATE:-}
model_choice=${MODEL_CHOICE:-openai_whisper_large_v3_turbo_4bit}
model_base_path=${MODEL_BASE_PATH:-"model/${model_choice}"}
eval_function=${EVAL_FUNCTION:-lev}

usage(){
	cat <<EOF
Usage: $0 [options]

Environment overrides accepted (examples):
	AUG_TYPE, SRC_DIR, LABEL_FILE, DIST_DIR, N_AUGMENTATIONS
	MODEL_STATE, MODEL_CHOICE, MODEL_BASE_PATH, EVAL_FUNCTION

Example: AUG_TYPE=TPGB ./pipeline.sh
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

echo "[pipeline] aug_type=$aug_type src_dir=$src_dir dist_dir=$dist_dir n_augmentations=$n_augmentations"

# Run preprocessing (dataset augmentation)
echo "[pipeline] Running preprocess.py..."
# $PYTHON preprocess.py --aug_type "$aug_type" --src_dir "$src_dir" --label_file "$label_file" --dist_dir "$dist_dir" --n_augmentations "$n_augmentations"

# Run training
echo "[pipeline] Running train.py..."
if [[ -n "$model_state" ]]; then
	model_state_path="model/${model_choice}/${model_state}"
	$PYTHON train.py --dataset "$dist_dir" --model_choice "$model_choice" --model_state_path "$model_state_path" --eval_function "$eval_function"
else
	$PYTHON train.py --dataset "$dist_dir" --model_choice "$model_choice" --eval_function "$eval_function"
fi

# Find latest model directory (exclude names containing "checkpoint")
if [[ ! -d "$model_base_path" ]]; then
	echo "[pipeline] model base path '$model_base_path' does not exist. Exiting."
	exit 1
fi

latest_model_dir=""
latest_model_dir=$(find "$model_base_path" -maxdepth 1 -mindepth 1 -type d ! -iname '*checkpoint*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | awk '{print $2}') || true

if [[ -z "$latest_model_dir" ]]; then
	echo "[pipeline] No model directories found in $model_base_path (after training). Exiting."
	exit 1
fi

echo "[pipeline] Latest model directory: $latest_model_dir"

# Run prediction using latest model
echo "[pipeline] Running prediction.py with model_dir=$latest_model_dir"
$PYTHON prediction.py --model_dir "$latest_model_dir"

echo "[pipeline] Done."

