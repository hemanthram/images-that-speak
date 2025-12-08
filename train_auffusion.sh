#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for train_auffusion.py
# - Activates .venv if present
# - Hard-coded settings below (edit these lines as needed)
#
# Usage:
#   ./train_auffusion.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if available
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  PY=python
else
  PY=python3
fi

# -----------------------------
# Hard-coded settings
# -----------------------------
OUT_DIR="$SCRIPT_DIR/out/phoneme_lora_npz_slow_fin50"
REPO_ID="auffusion/auffusion-full-no-adapter"
DEVICE="cuda"          # cpu | cuda
USE_FP16=1             # 1 => enable --fp16; 0 => disable
SEED="none"

STEPS="20000"
BATCH="2"
LR="5e-4"
SAVE_EVERY="200"
PLOT_LOSS_EVERY="5"
RESUME=1

LORA_RANK="8"
LORA_ALPHA="16.0"
LORA_DROPOUT="0.0"

# Conditioning
COND_TYPE="phoneme"  # phoneme | text

# Phoneme mode inputs (NPZ-only)
PROMPTS_FILE="$SCRIPT_DIR/data/prompts_slow_50.txt"
SPEC_DIR="$SCRIPT_DIR/data/spectograms_slow_50"
PHONEME_NPZ="$SCRIPT_DIR/data/plbert_sentence_emb_slow_50.npz"

# Text mode input (manifest CSV with columns audio_prompt, mel_npy)
MANIFEST=""  # e.g., $SCRIPT_DIR/data/manifest.csv

MEL_MIN="-11.0"
MEL_MAX="0.0"

NUM_WORKERS="2"

# -----------------------------
# Build args
# -----------------------------
ARGS=( "$SCRIPT_DIR/train_auffusion.py"
  --out_dir "$OUT_DIR"
  --repo_id "$REPO_ID"
  --device "$DEVICE"
  --steps "$STEPS"
  --batch "$BATCH"
  --lr "$LR"
  --save_every "$SAVE_EVERY"
  --plot_loss_every "$PLOT_LOSS_EVERY"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_targets "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2"
  --unfreeze_patterns "mid_block,up_blocks.3"
  --base_lr "2e-5"
  --mel_min "$MEL_MIN"
  --mel_max "$MEL_MAX"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --cond_type "$COND_TYPE"
)

if [[ "$USE_FP16" == "1" ]]; then
  ARGS+=( --fp16 )
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=( --resume )
fi

if [[ "$COND_TYPE" == "phoneme" ]]; then
  # Validate essential inputs
  if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: PROMPTS_FILE not found: $PROMPTS_FILE" >&2
    exit 1
  fi
  if [[ ! -d "$SPEC_DIR" ]]; then
    echo "ERROR: SPEC_DIR not found: $SPEC_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$PHONEME_NPZ" ]]; then
    echo "ERROR: PHONEME_NPZ not found: $PHONEME_NPZ" >&2
    exit 1
  fi
  ARGS+=( --prompts_file "$PROMPTS_FILE" --spec_dir "$SPEC_DIR" --phoneme_npz "$PHONEME_NPZ" )
else
  # Text mode (manifest CSV)
  if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: Text mode requires setting MANIFEST path inside this script." >&2
    exit 1
  fi
  ARGS+=( --manifest "$MANIFEST" )
fi

exec "$PY" "${ARGS[@]}"


