from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Resize and time-stretch audio files.")
parser.add_argument("--in_dir", type=Path, default=Path("raw_audio"), help="Input directory containing audio files.")
parser.add_argument("--out_dir", type=Path, default=Path("resized_audio"), help="Output directory for resized and processed audio files.")
args = parser.parse_args()

IN_DIR = args.in_dir
OUT_DIR = args.out_dir
TARGET_SEC = 10.0
TARGET_SR = 44100
TEMPO = 0.5  # quarter tempo => 4x slower, longer

EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in tqdm(sorted(IN_DIR.rglob("*"))):
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue

        out = (OUT_DIR / f.relative_to(IN_DIR)).with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)

        # Load audio as mono at target sample rate
        try:
            y, sr = librosa.load(str(f), sr=TARGET_SR, mono=True)
        except Exception as e:
            print("skip (load error):", f, e)
            continue

        # Ensure a float64 array for processing
        y = np.asarray(y, dtype=np.float64, order="C")

        if y.size == 0 or sr <= 0:
            print("skip (empty or invalid):", f)
            continue

        # Time-stretch using WSOLA via audiotsm (TEMPO < 1 slows down, preserving pitch)
        try:
            from audiotsm import wsola
            from audiotsm.io.array import ArrayReader, ArrayWriter
            reader = ArrayReader(y.reshape(1, -1))
            writer = ArrayWriter(1)
            tsm = wsola(channels=1, speed=TEMPO)
            tsm.run(reader, writer)
            y_slow = np.asarray(writer.data[0], dtype=np.float64, order="C")
        except Exception as e:
            print("skip (stretch error):", f, e)
            continue

        # Duration after slowing
        dur2 = float(y_slow.shape[0]) / float(sr)
        if dur2 <= 0.01:
            print("skip:", f)
            continue

        # Trim to TARGET_SEC and/or pad with zeros to exactly TARGET_SEC
        target_len = int(round(TARGET_SEC * sr))
        if y_slow.shape[0] > target_len:
            y_final = y_slow[:target_len]
        elif y_slow.shape[0] < target_len:
            pad_len = target_len - y_slow.shape[0]
            y_final = np.pad(y_slow, (0, pad_len), mode="constant")
        else:
            y_final = y_slow

        pad_sec = max(0.0, TARGET_SEC - dur2)

        # Write WAV (16-bit PCM)
        try:
            sf.write(str(out), y_final, sr, subtype="PCM_16")
        except Exception as e:
            print("write failed:", out, e)
            continue

        # print(f"{f.name}: slowed={dur2:.2f}s -> {TARGET_SEC:.2f}s (pad {pad_sec:.2f}s)")

if __name__ == "__main__":
    main()
