#!/usr/bin/env bash
set -euo pipefail

# Orchestrates the full pipeline:
# 1) Generate prompts from words (L prompts, W words each)
# 2) Generate TTS audio into raw_audio/
# 3) Resize audio into resized_audio/
# 4) Generate spectrograms into spectograms/ (optionally save PNGs)
#
# Defaults are chosen to match the Python scripts in this folder.
# Configure everything below; this script does not accept CLI args.
#
# How to use:
#   1) Edit the variables in the "Defaults" section below.
#   2) Run: ./run.sh

source ../.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
L="50"
W="9"
PROMPTS_FILE="$SCRIPT_DIR/prompts_50.txt"
WORDS_FILE="$SCRIPT_DIR/words.txt"
TTS_MODEL="tts_models/en/ljspeech/tacotron2-DDC"
TTS_OUT_DIR="$SCRIPT_DIR/raw_audio_50"
RESIZED_OUT_DIR="$SCRIPT_DIR/resized_audio_50"
SPEC_OUT_DIR="$SCRIPT_DIR/spectograms_50"
PHONEMES_OUT_FILE="$SCRIPT_DIR/plbert_sentence_emb_50.npz"
TTS_START="1"
TTS_END=""             # empty => process to end
TTS_SKIP_EXISTING="0"  # 1 => pass --skip-existing
SAVE_PNGS="1"          # 1 => pass --save_pngs

echo "=== Step 1: Generating prompts ==="
echo "Words file     : $WORDS_FILE"
echo "Prompts file   : $PROMPTS_FILE"
echo "L (count)      : $L"
echo "W (per prompt) : $W"
python3 "$SCRIPT_DIR/create_prompts_from_words.py" "$L" "$W" -o "$PROMPTS_FILE" -w "$WORDS_FILE"

echo "=== Step 2: Generating TTS audio ==="
echo "TTS model      : $TTS_MODEL"
echo "Prompts        : $PROMPTS_FILE"
echo "Out dir        : $TTS_OUT_DIR"
echo "Start..End     : ${TTS_START}..${TTS_END:-last}"
TTS_ARGS=( --prompts "$PROMPTS_FILE" --out-dir "$TTS_OUT_DIR" --model "$TTS_MODEL" --start "$TTS_START" )
if [[ -n "$TTS_END" ]]; then
  TTS_ARGS+=( --end "$TTS_END" )
fi
if [[ "$TTS_SKIP_EXISTING" == "1" ]]; then
  TTS_ARGS+=( --skip-existing )
fi
python3 "$SCRIPT_DIR/generate_tts.py" "${TTS_ARGS[@]}"

echo "=== Step 3: Resizing audio into resized_audio/ ==="
python3 "$SCRIPT_DIR/resize_audio.py" --in_dir "$TTS_OUT_DIR" --out_dir "$RESIZED_OUT_DIR"

echo "=== Step 4: Generating spectrograms ==="
SPEC_ARGS=( --in_dir "$RESIZED_OUT_DIR" --out_dir "$SPEC_OUT_DIR" )
if [[ "$SAVE_PNGS" == "1" ]]; then
  SPEC_ARGS+=( --save_pngs )
fi
python3 "$SCRIPT_DIR/generate_spectograms.py" "${SPEC_ARGS[@]}"

python3 "$SCRIPT_DIR/generate_phonemes.py" --prompts "$PROMPTS_FILE" --out_npz "$PHONEMES_OUT_FILE"

echo "All done."


