import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.components.auffusion_converter import Generator, denormalize_spectrogram


@torch.no_grad()
def decode_spec_file_to_audio(
    spec_path: str | Path,
    repo_id: str = "auffusion/auffusion-full-no-adapter",
    device: str = "cuda",
    fp16: bool = True,
) -> np.ndarray:
    """
    Load a spectrogram file and synthesize audio using the same approach as in src:
      - Load vocoder from HF with Generator.from_pretrained(repo_id, subfolder='vocoder')
      - If .npy (log-mel [256,1024], values ~[-11,0]) pass directly
      - If an image, first denormalize via denormalize_spectrogram
    Returns float32 waveform as a 1D numpy array (16 kHz expected in this repo).
    """
    spec_path = Path(spec_path)
    use_cuda = device == "cuda" and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if (fp16 and use_cuda) else torch.float32

    # Load vocoder like in src/guidance/auffusion.py
    vocoder = Generator.from_pretrained(repo_id, subfolder="vocoder")
    vocoder = vocoder.to(device=torch_device, dtype=dtype).eval()

    # Prepare mel tensor [1, 256, 1024] in log-mel domain
    if spec_path.suffix.lower() == ".npy":
        mel = np.load(spec_path)
        if mel.ndim != 2:
            raise ValueError(f"Expected 2D mel array (256,1024). Got shape={mel.shape}")
        
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device=torch_device, dtype=dtype)
        mel_t = torch.flip(mel_t, [1])
    else:
        img = Image.open(spec_path).convert("RGB")
        x = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C] in 0..1
        x = x.permute(2, 0, 1)  # [C, H, W]
        denorm = denormalize_spectrogram(x)  # [256, 1024] in log-mel domain
        mel_t = denorm.unsqueeze(0).to(device=torch_device, dtype=dtype)  # [1, 256, 1024]

    wav = vocoder.inference(mel_t)[0].astype("float32")  # float32 1D numpy array
    return wav


if __name__ == "__main__":
    import argparse
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_path", type=str, default=None, help="Path to spectrogram .npy (preferred) or image.")
    ap.add_argument("--out_dir", type=str, default=None, help="If set, writes audio.wav here; defaults to spec_path parent.")
    ap.add_argument("--repo_id", type=str, default="auffusion/auffusion-full-no-adapter", help="HF repo id to load vocoder from")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    if args.spec_path is None and args.out_dir is not None:
        # Default to spec.npy inside out_dir (as produced by infer_from_prompt.py)
        spec_path = Path(args.out_dir) / "spec.npy"
    elif args.spec_path is not None:
        spec_path = Path(args.spec_path)
    else:
        raise SystemExit("Provide --spec_path or --out_dir (will look for out_dir/spec.npy).")

    out_dir = Path(args.out_dir) if args.out_dir is not None else spec_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = decode_spec_file_to_audio(spec_path, repo_id=args.repo_id, device=args.device, fp16=args.fp16)

    # Normalize to avoid clipping (same style as infer_from_prompt.py)
    m = float(np.max(np.abs(wav)) + 1e-9)
    if m > 0:
        wav = wav * min(1.0, 0.98 / m)

    sr = 16000
    out_path = out_dir / "audio.wav"
    sf.write(str(out_path), wav, sr, subtype="PCM_16")
    print(f"[audio] wrote {out_path} sr={sr} len={wav.shape[0]} max_abs={np.max(np.abs(wav)):.4f}")

