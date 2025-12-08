#!/usr/bin/env bash
set -euo pipefail

# Batch generate image+spectrogram pairs with random prompts
#
# Usage:
#   ./imprint_joint_batch.sh [NUM_SAMPLES] [OUT_DIR]
#   ./imprint_joint_batch.sh 100 out/bulk_joint
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
# Settings
# -----------------------------

NUM_SAMPLES="${1:-100}"
OUT_DIR="${2:-$SCRIPT_DIR/out/bulk_joint_1}"

WORDS_FILE="$SCRIPT_DIR/data/words.txt"
IMAGE_PROMPTS_FILE="$SCRIPT_DIR/data/image_prompts.txt"
NUM_WORDS_FOR_AUDIO=6

# Model repos
SD_REPO_ID="runwayml/stable-diffusion-v1-5"
AUFFUSION_REPO_ID="auffusion/auffusion-full-no-adapter"

DEVICE="cuda"
USE_FP16=1

# Diffusion settings
STEPS="200"
IMAGE_GUIDANCE="10"
AUDIO_GUIDANCE="10"
IMAGE_START_STEP="10"
AUDIO_START_STEP="0"
AUDIO_WEIGHT="0.4"

IMG_HEIGHT="256"
IMG_WIDTH="1024"

# LoRA settings
LORA_PATH="$SCRIPT_DIR/out/phoneme_lora_npz_fin50/checkpoints/latest/lora.pt"
BASE_UNET_PATH="$SCRIPT_DIR/out/phoneme_lora_npz_fin50/checkpoints/latest/base_unet.pt"
LORA_TARGETS="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2"

# Conditioning
COND_TYPE="phoneme"
ADAPTER_PATH="$SCRIPT_DIR/out/phoneme_lora_npz_fin50/checkpoints/latest/phoneme_adapter.pt"
PLBERT_MODEL_ID="papercup-ai/multilingual-pl-bert"
PH_LANG="en-us"

# -----------------------------
# Validation
# -----------------------------
if [[ ! -f "$WORDS_FILE" ]]; then
  echo "ERROR: Words file not found: $WORDS_FILE" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PROMPTS_FILE" ]]; then
  echo "ERROR: Image prompts file not found: $IMAGE_PROMPTS_FILE" >&2
  exit 1
fi

if [[ ! -f "$LORA_PATH" ]]; then
  echo "ERROR: LORA_PATH not found: $LORA_PATH" >&2
  exit 1
fi

if [[ "$COND_TYPE" == "phoneme" && ! -f "$ADAPTER_PATH" ]]; then
  echo "ERROR: ADAPTER_PATH not found: $ADAPTER_PATH" >&2
  exit 1
fi

# -----------------------------
# Setup
# -----------------------------
mkdir -p "$OUT_DIR"

# Read files into arrays
mapfile -t WORDS < "$WORDS_FILE"
mapfile -t IMAGE_PROMPTS < "$IMAGE_PROMPTS_FILE"

NUM_WORDS=${#WORDS[@]}
NUM_IMAGE_PROMPTS=${#IMAGE_PROMPTS[@]}

# CSV file for logging prompts
CSV_FILE="$OUT_DIR/prompts.csv"
if [[ ! -f "$CSV_FILE" ]]; then
  echo "index,image_prompt,audio_prompt" > "$CSV_FILE"
fi

# Temp directory for intermediate outputs
TMP_DIR="$OUT_DIR/.tmp"
mkdir -p "$TMP_DIR"

# -----------------------------
# Build base args for imprint_joint_denoise.py
# -----------------------------
BASE_ARGS=(
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
  --cond_type "$COND_TYPE"
)

if [[ "$USE_FP16" == "1" ]]; then
  BASE_ARGS+=( --fp16 )
fi

if [[ -n "$BASE_UNET_PATH" && -f "$BASE_UNET_PATH" ]]; then
  BASE_ARGS+=( --base_unet_path "$BASE_UNET_PATH" )
fi

if [[ "$COND_TYPE" == "phoneme" ]]; then
  BASE_ARGS+=( --adapter_path "$ADAPTER_PATH" )
  BASE_ARGS+=( --plbert_model_id "$PLBERT_MODEL_ID" --ph_lang "$PH_LANG" )
fi

# -----------------------------
# Generate samples
# -----------------------------
echo "=============================================="
echo "Batch Image + Spectrogram Generation"
echo "=============================================="
echo "Samples to generate: $NUM_SAMPLES"
echo "Output directory: $OUT_DIR"
echo "Words file: $WORDS_FILE ($NUM_WORDS words)"
echo "Image prompts: $IMAGE_PROMPTS_FILE ($NUM_IMAGE_PROMPTS prompts)"
echo "=============================================="

for i in $(seq 1 "$NUM_SAMPLES"); do
  PADDED=$(printf "%06d" "$i")
  
  # Check if already exists
  if [[ -f "$OUT_DIR/${PADDED}.npy" && -f "$OUT_DIR/${PADDED}_img.png" ]]; then
    echo "[$PADDED/$NUM_SAMPLES] SKIP (already exists)"
    continue
  fi

  # Random image prompt
  IMG_IDX=$((RANDOM % NUM_IMAGE_PROMPTS))
  IMG_PROMPT="${IMAGE_PROMPTS[$IMG_IDX]}"

  # Random audio prompt (9 words)
  AUDIO_PROMPT=""
  for j in $(seq 1 $NUM_WORDS_FOR_AUDIO); do
    WORD_IDX=$((RANDOM % NUM_WORDS))
    WORD="${WORDS[$WORD_IDX]}"
    if [[ -z "$AUDIO_PROMPT" ]]; then
      AUDIO_PROMPT="$WORD"
    else
      AUDIO_PROMPT="$AUDIO_PROMPT $WORD"
    fi
  done

  echo "[$PADDED/$NUM_SAMPLES] Generating..."
  echo "  Image: ${IMG_PROMPT:0:50}..."
  echo "  Audio: ${AUDIO_PROMPT:0:50}..."

  # Run generation
  "$PY" "$SCRIPT_DIR/imprint_joint_denoise.py" \
    --image_prompt "$IMG_PROMPT" \
    --audio_prompt "$AUDIO_PROMPT" \
    --out_dir "$TMP_DIR" \
    "${BASE_ARGS[@]}"

  # Move outputs to numbered files
  mv "$TMP_DIR/spec.npy" "$OUT_DIR/${PADDED}.npy"
  mv "$TMP_DIR/img.png" "$OUT_DIR/${PADDED}_img.png"
  mv "$TMP_DIR/spec.png" "$OUT_DIR/${PADDED}_spec.png"
  
  # Decode to audio if spec exists
  if [[ -f "$OUT_DIR/${PADDED}.npy" ]]; then
    DECODE_ARGS=(
      "$SCRIPT_DIR/decode_spec_to_audio.py"
      --spec_path "$OUT_DIR/${PADDED}.npy"
      --repo_id "$AUFFUSION_REPO_ID"
      --device "$DEVICE"
    )
    if [[ "$USE_FP16" == "1" ]]; then
      DECODE_ARGS+=( --fp16 )
    fi
    "$PY" "${DECODE_ARGS[@]}"
    # decode_spec_to_audio.py saves as audio.wav in spec's parent dir, rename it
    if [[ -f "$OUT_DIR/audio.wav" ]]; then
      mv "$OUT_DIR/audio.wav" "$OUT_DIR/${PADDED}.wav"
    fi
  fi

  # Log to CSV (escape quotes in prompts)
  IMG_PROMPT_ESC="${IMG_PROMPT//\"/\"\"}"
  AUDIO_PROMPT_ESC="${AUDIO_PROMPT//\"/\"\"}"
  echo "$PADDED,\"$IMG_PROMPT_ESC\",\"$AUDIO_PROMPT_ESC\"" >> "$CSV_FILE"

  echo "[$PADDED] Done!"
done

# Cleanup temp directory
rm -rf "$TMP_DIR"

echo "=============================================="
echo "Batch generation complete!"
echo "Output: $OUT_DIR"
echo "Prompts log: $CSV_FILE"
echo "=============================================="
