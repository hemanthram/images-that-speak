#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for imprint_joint_denoise.py
# - Activates .venv if present
# - Hard-coded settings below (edit these lines as needed)
#
# Usage:
#   ./imprint_joint_denoise.sh
#   ./imprint_joint_denoise.sh "a corgi" "a dog barking"
#   ./imprint_joint_denoise.sh "a corgi" "a dog barking" out/my_output
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

# Prompts (can override with CLI arguments)
IMAGE_PROMPT="${1:-silhoutte of cats}"
AUDIO_PROMPT="${2:-hello world america austin texas}"

# Output directory (can override with third CLI argument)
OUT_DIR="${3:-$SCRIPT_DIR/out/imprint_joint_phoneme}"

# Model repos
SD_REPO_ID="runwayml/stable-diffusion-v1-5"
AUFFUSION_REPO_ID="auffusion/auffusion-full-no-adapter"

DEVICE="cuda"          # cpu | cuda
USE_FP16=1             # 1 => enable --fp16; 0 => disable
SEED=21

# Diffusion settings
STEPS="100"
IMAGE_GUIDANCE="7.5"
AUDIO_GUIDANCE="7.5"
IMAGE_START_STEP="10"
AUDIO_START_STEP="0"
AUDIO_WEIGHT="0.5"

# Image/spectrogram dimensions
IMG_HEIGHT="256"
IMG_WIDTH="1024"

# LoRA settings
LORA_PATH="$SCRIPT_DIR/out/phoneme_lora_npz_fin50/checkpoints/latest/lora.pt"
BASE_UNET_PATH="$SCRIPT_DIR/out/phoneme_lora_npz_fin50/checkpoints/latest/base_unet.pt"
LORA_TARGETS="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2"

# Conditioning type
COND_TYPE="phoneme"  # text | phoneme

# Phoneme mode settings (only used if COND_TYPE=phoneme)
ADAPTER_PATH="$SCRIPT_DIR/out/phoneme_lora_npz_fin50/checkpoints/latest/phoneme_adapter.pt"
PHONEME_NPZ=""   # e.g., "$SCRIPT_DIR/data/plbert_sentence_emb_50.npz"
PROMPT_INDEX=""  # Row index in NPZ (leave empty to use audio_prompt text)
PLBERT_MODEL_ID="papercup-ai/multilingual-pl-bert"
PH_LANG="en-us"

# Post-processing options
CUTOFF_LATENT=0    # 1 => enable; 0 => disable
CROP_IMAGE=0       # 1 => enable; 0 => disable
USE_COLORMAP=1     # 1 => save colormap version; 0 => skip

# -----------------------------
# Validation
# -----------------------------
if [[ ! -f "$LORA_PATH" ]]; then
  echo "ERROR: LORA_PATH not found: $LORA_PATH" >&2
  exit 1
fi

if [[ "$COND_TYPE" == "phoneme" && ! -f "$ADAPTER_PATH" ]]; then
  echo "ERROR: phoneme mode requires ADAPTER_PATH: $ADAPTER_PATH" >&2
  exit 1
fi

# -----------------------------
# Build args
# -----------------------------
ARGS=( "$SCRIPT_DIR/imprint_joint_denoise.py"
  --image_prompt "$IMAGE_PROMPT"
  --audio_prompt "$AUDIO_PROMPT"
  --out_dir "$OUT_DIR"
  --sd_repo_id "$SD_REPO_ID"
  --auffusion_repo_id "$AUFFUSION_REPO_ID"
  --lora_path "$LORA_PATH"
  --lora_targets "$LORA_TARGETS"
  --device "$DEVICE"
  --steps "$STEPS"
  --image_guidance "$IMAGE_GUIDANCE"
  --audio_guidance "$AUDIO_GUIDANCE"
  --image_start_step "$IMAGE_START_STEP"
  --audio_start_step "$AUDIO_START_STEP"
  --audio_weight "$AUDIO_WEIGHT"
  --img_height "$IMG_HEIGHT"
  --img_width "$IMG_WIDTH"
  --seed "$SEED"
  --cond_type "$COND_TYPE"
)

if [[ "$USE_FP16" == "1" ]]; then
  ARGS+=( --fp16 )
fi

if [[ -n "$BASE_UNET_PATH" && -f "$BASE_UNET_PATH" ]]; then
  ARGS+=( --base_unet_path "$BASE_UNET_PATH" )
fi

if [[ "$CUTOFF_LATENT" == "1" ]]; then
  ARGS+=( --cutoff_latent )
fi

if [[ "$CROP_IMAGE" == "1" ]]; then
  ARGS+=( --crop_image )
fi

if [[ "$USE_COLORMAP" == "1" ]]; then
  ARGS+=( --use_colormap )
fi

# Phoneme conditioning
if [[ "$COND_TYPE" == "phoneme" ]]; then
  ARGS+=( --adapter_path "$ADAPTER_PATH" )
  ARGS+=( --plbert_model_id "$PLBERT_MODEL_ID" --ph_lang "$PH_LANG" )
  
  if [[ -n "$PHONEME_NPZ" && -n "$PROMPT_INDEX" ]]; then
    if [[ ! -f "$PHONEME_NPZ" ]]; then
      echo "ERROR: PHONEME_NPZ not found: $PHONEME_NPZ" >&2
      exit 1
    fi
    ARGS+=( --phoneme_npz "$PHONEME_NPZ" --prompt_index "$PROMPT_INDEX" )
  fi
fi

echo "=============================================="
echo "Joint Image + Spectrogram Generation"
echo "=============================================="
echo "Image prompt: $IMAGE_PROMPT"
echo "Audio prompt: $AUDIO_PROMPT"
echo "Output dir:   $OUT_DIR"
echo "Steps:        $STEPS"
echo "Audio weight: $AUDIO_WEIGHT"
echo "=============================================="

"$PY" "${ARGS[@]}"

# -----------------------------
# Decode spectrogram to audio
# -----------------------------
SPEC_NPY="$OUT_DIR/spec.npy"
if [[ -f "$SPEC_NPY" ]]; then
  echo ""
  echo "[audio] Decoding spectrogram to audio..."
  DECODE_ARGS=(
    "$SCRIPT_DIR/decode_spec_to_audio.py"
    --spec_path "$SPEC_NPY"
    --repo_id "$AUFFUSION_REPO_ID"
    --device "$DEVICE"
  )
  if [[ "$USE_FP16" == "1" ]]; then
    DECODE_ARGS+=( --fp16 )
  fi
  "$PY" "${DECODE_ARGS[@]}"
  echo "[done] Audio saved to $OUT_DIR/audio.wav"
else
  echo "[warn] spec.npy not found, skipping audio generation"
fi
