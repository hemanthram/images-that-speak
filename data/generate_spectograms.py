import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from PIL import Image


EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


def ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: [C, N] -> [1, N]
    return wav.mean(dim=0, keepdim=True) if wav.dim() == 2 else wav.unsqueeze(0)


def pad_or_trim(wav: torch.Tensor, target_samples: int) -> torch.Tensor:
    # wav: [1, N]
    n = wav.shape[-1]
    if n == target_samples:
        return wav
    if n > target_samples:
        return wav[..., :target_samples]
    pad = target_samples - n
    return torch.nn.functional.pad(wav, (0, pad))


def fix_frames(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
    # mel: [n_mels, T]
    T = mel.shape[-1]
    if T == target_frames:
        return mel
    if T > target_frames:
        return mel[..., :target_frames]
    # pad with the minimum value (silence-ish in log space)
    pad_val = mel.min()
    pad_amt = target_frames - T
    pad = pad_val.expand(mel.shape[0], pad_amt)
    return torch.cat([mel, pad], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="resized_audio", help="input folder of audio files")
    ap.add_argument("--out_dir", default="spectograms", help="output folder for mel.npy files")
    ap.add_argument("--save_pngs", action="store_true", help="also save spectograms as PNGs in out_dir/imgs/")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--target_sec", type=float, default=10.24)   # Auffusion pads to 10.24s :contentReference[oaicite:1]{index=1}
    ap.add_argument("--target_frames", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=160)
    ap.add_argument("--fmax", type=int, default=8000)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "imgs"
    if args.save_pngs:
        img_dir.mkdir(parents=True, exist_ok=True)

    target_samples = int(round(args.sr * args.target_sec))

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=0.0,
        f_max=float(args.fmax),
        power=2.0,
        center=True,
        pad_mode="reflect",
    )

    for f in sorted(in_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue

        wav, sr_in = torchaudio.load(str(f))
        wav = ensure_mono(wav)

        if sr_in != args.sr:
            wav = torchaudio.functional.resample(wav, sr_in, args.sr)

        wav = pad_or_trim(wav, target_samples)

        # mel: [1, n_mels, T]
        mel = mel_tf(wav)
        mel = mel.squeeze(0)  # [n_mels, T]

        # log-mel (stable)
        mel = torch.log(mel.clamp_min(1e-5))

        # enforce exact (256, 1024)
        mel = fix_frames(mel, args.target_frames)
        assert mel.shape == (args.n_mels, args.target_frames), mel.shape

        out_path = (out_dir / f.relative_to(in_dir)).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mel.cpu().numpy().astype(np.float32))

        print(f"{f.name}: saved {out_path}  shape={tuple(mel.shape)}")

        if args.save_pngs:
            # normalize log-mel to [0, 255] per-sample and flip vertically for typical spectrogram view
            mel_np = mel.cpu().numpy()
            vmin = float(mel_np.min())
            vmax = float(mel_np.max())
            denom = max(vmax - vmin, 1e-12)
            mel_norm = (mel_np - vmin) / denom
            mel_img = (mel_norm[::-1, :] * 255.0).clip(0, 255).astype(np.uint8)
            # mel_img = np.flip(mel_img, axis=1)

            png_path = (img_dir / f.relative_to(in_dir)).with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mel_img, mode="L").save(png_path)
            print(f"{f.name}: saved {png_path}  image_shape={mel_img.shape}")

    print("done.")


if __name__ == "__main__":
    main()