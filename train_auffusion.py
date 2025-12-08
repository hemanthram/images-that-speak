#!/usr/bin/env python3
"""
LoRA-train Auffusion UNet to map text (audio_prompt) -> mel-spectrogram image (mel_npy).

Fixes:
- Keep LoRA weights in FP32 even when UNet is FP16 → avoids GradScaler error:
  "Attempting to unscale FP16 gradients."
- LoRA branch runs in FP32, then cast back to base dtype for addition.
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re
import csv as _csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import DiffusionPipeline


def set_seed(seed: int | None):
    if seed is None:
        return  # Skip seeding for non-reproducible randomness
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    if not rows:
        raise RuntimeError(f"Empty manifest: {csv_path}")
    return rows

def smart_resolve(manifest_dir: Path, p: str) -> Path:
    p = str(p).strip()
    if os.path.isabs(p):
        return Path(p)
    rel = Path(p)

    c1 = manifest_dir / rel
    if c1.exists():
        return c1

    c2 = Path.cwd() / rel
    if c2.exists():
        return c2

    return c1  # best guess for error path

def count_params(params):
    return sum(int(p.numel()) for p in params)

def find_latest_step(out_dir: Path) -> int:
    ck_root = out_dir / "checkpoints"
    if not ck_root.exists():
        return 0
    max_step = 0
    for p in ck_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"step_(\d+)$", p.name)
        if m:
            try:
                s = int(m.group(1))
                if s > max_step:
                    max_step = s
            except Exception:
                pass
    return max_step

def append_loss_csv(loss_csv: Path, step: int, loss_val: float):
    new_file = not loss_csv.exists()
    with loss_csv.open("a", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow(["step", "loss"])
        w.writerow([int(step), float(loss_val)])

def plot_loss_curve(loss_csv: Path, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception:
        return
    steps: List[int] = []
    losses: List[float] = []
    try:
        with loss_csv.open("r", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                steps.append(int(row.get("step", 0)))
                losses.append(float(row.get("loss", 0.0)))
    except Exception:
        return
    if not steps:
        return
    plt.figure(figsize=(7.5, 4.5))
    # Plot faded raw losses at every 5 steps
    raw_idx = [i for i, s in enumerate(steps) if (s % 5) == 0]
    if raw_idx:
        plt.plot([steps[i] for i in raw_idx],
                 [losses[i] for i in raw_idx],
                 linestyle="-", linewidth=0.8, color="#999999", alpha=0.35, label="loss (every 5 steps)")
    # Moving average (window=50) as main line
    window = 500
    if len(losses) >= window:
        kernel = _np.ones(window, dtype=_np.float64) / float(window)
        ma = _np.convolve(_np.array(losses, dtype=_np.float64), kernel, mode="valid")
        ma_steps = steps[window - 1:]
        plt.plot(ma_steps, ma, linewidth=2.0, color="#1f77b4", label=f"loss (MA {window})")
    else:
        # Not enough points; fall back to plotting all losses prominently
        plt.plot(steps, losses, linewidth=2.0, color="#1f77b4", label="loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(str(out_png))
    finally:
        plt.close()


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: str, cond_col="audio_prompt", spec_col="mel_npy"):
        self.manifest_path = Path(manifest_csv).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.rows = read_rows(self.manifest_path)

        if cond_col not in self.rows[0]:
            raise RuntimeError(f"Manifest missing '{cond_col}'. Columns: {list(self.rows[0].keys())}")
        if spec_col not in self.rows[0]:
            raise RuntimeError(f"Manifest missing '{spec_col}'. Columns: {list(self.rows[0].keys())}")

        self.cond_col = cond_col
        self.spec_col = spec_col

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = str(r[self.cond_col])

        p = smart_resolve(self.manifest_dir, r[self.spec_col])
        if not p.exists():
            raise FileNotFoundError(f"Missing mel/spec npy: {p}")

        x = np.load(str(p))
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim != 2:
            raise RuntimeError(f"Expected 2D mel/spec array, got {x.shape} at {p}")

        # allow stored as (T, n_mels)
        if x.shape == (1024, 256):
            x = x.T

        mel = torch.from_numpy(x.astype(np.float32))  # (256,1024)
        return mel, text, {"id": r.get("id", str(idx)), "path": str(p)}

def collate_fn(batch):
    mels = torch.stack([b[0] for b in batch], dim=0)  # (B,256,1024)
    conds = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    # If conditions are tensors (e.g., phoneme embeddings), stack them
    if isinstance(conds[0], torch.Tensor):
        conds = torch.stack(conds, dim=0)
    return mels, conds, metas


def mel_to_vae_image(mel: torch.Tensor, in_channels: int, mel_min: float, mel_max: float) -> torch.Tensor:
    """
    mel: (B,256,1024) -> (B,C,256,1024) in [-1,1]
    """
    x = mel.clamp(mel_min, mel_max)
    x = (x - mel_min) / (mel_max - mel_min + 1e-8)  # [0,1]
    x = x * 2.0 - 1.0                               # [-1,1]
    x = x.unsqueeze(1)                               # (B,1,H,W)
    if in_channels == 1:
        return x
    return x.repeat(1, in_channels, 1, 1)


class LoRALinear(nn.Module):
    """
    Frozen base nn.Linear + trainable low-rank adapters in FP32.
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.rank)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # IMPORTANT: A/B will be moved to device but kept in FP32
        self.A = nn.Linear(base.in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x, *args, **kwargs):
        # base path (likely fp16)
        base_out = self.base(x)

        # lora path in fp32 for stable grads -> then cast back
        x32 = self.drop(x).float()
        lora_out = self.B(self.A(x32)) * self.scale
        lora_out = lora_out.to(dtype=base_out.dtype)

        return base_out + lora_out


def _set_module(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_lora_into_unet(
    unet: nn.Module,
    device: torch.device,
    rank=8,
    alpha=16.0,
    dropout=0.0,
    targets=("to_q","to_k","to_v","to_out.0"),
):
    """
    Replace targeted nn.Linear modules inside UNet with LoRALinear wrappers.

    CRITICAL: Keep LoRA A/B in FP32 on the right device (prevents GradScaler error).
    """
    for p in unet.parameters():
        p.requires_grad = False

    replace: List[Tuple[str, nn.Linear]] = []
    for name, mod in unet.named_modules():
        if isinstance(mod, nn.Linear):
            for t in targets:
                if name.endswith(t):
                    replace.append((name, mod))
                    break

    if not replace:
        raise RuntimeError("No target Linear layers found in UNet (to_q/to_k/to_v/to_out.0).")

    for name, base in replace:
        wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)

        # ✅ move ONLY LoRA params to gpu, keep them FP32
        wrapped.A = wrapped.A.to(device=device, dtype=torch.float32)
        wrapped.B = wrapped.B.to(device=device, dtype=torch.float32)

        _set_module(unet, name, wrapped)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Injected LoRA but found 0 trainable parameters.")
    return trainable


def unfreeze_unet_by_patterns(unet: nn.Module, patterns: List[str]) -> List[nn.Parameter]:
    """
    Set requires_grad=True for parameters of UNet modules whose qualified name contains
    any of the provided patterns. If a module is a LoRALinear, only unfreeze the
    underlying base Linear parameters (not the LoRA A/B) to avoid optimizer duplicates.
    Newly unfrozen parameters are cast to float32 to avoid AMP unscale errors on fp16 grads.
    """
    selected: List[nn.Parameter] = []
    pats = [p for p in patterns if p]
    if not pats:
        return selected

    for name, mod in unet.named_modules():
        if any(pat in name for pat in pats):
            if isinstance(mod, LoRALinear):
                for p in mod.base.parameters():
                    if not p.requires_grad:
                        # cast to fp32 to keep grads in fp32 with GradScaler
                        p.data = p.data.to(dtype=torch.float32)
                        p.requires_grad = True
                        selected.append(p)
            else:
                for p in mod.parameters():
                    if not p.requires_grad:
                        p.data = p.data.to(dtype=torch.float32)
                        p.requires_grad = True
                        selected.append(p)
    # Deduplicate while preserving order
    seen = set()
    unique: List[nn.Parameter] = []
    for p in selected:
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            unique.append(p)
    return unique


def extract_lora_state(unet: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, mod in unet.named_modules():
        if isinstance(mod, LoRALinear):
            state[f"{name}.A.weight"] = mod.A.weight.detach().cpu()
            state[f"{name}.B.weight"] = mod.B.weight.detach().cpu()
            state[f"{name}.alpha"] = torch.tensor(mod.alpha)
            state[f"{name}.rank"]  = torch.tensor(mod.rank)
    return state


class PhonemeNPZDataset(Dataset):
    """
    Loads sentence embeddings from a single NPZ file (key 'emb') and pairs with spectrogram .npy files.
    """
    def __init__(self, prompts_file: str, spec_dir: str, npz_path: str):
        self.prompts_path = Path(prompts_file).resolve()
        self.spec_dir = Path(spec_dir).resolve()
        self.npz_path = Path(npz_path).resolve()

        if not self.prompts_path.exists():
            raise FileNotFoundError(self.prompts_path)
        if not self.spec_dir.exists():
            raise FileNotFoundError(self.spec_dir)
        if not self.npz_path.exists():
            raise FileNotFoundError(self.npz_path)

        lines = [ln.strip() for ln in self.prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.prompts = lines

        self.spec_paths: List[Path] = sorted([p for p in self.spec_dir.rglob("*.npy") if p.is_file()])
        if not self.spec_paths:
            raise RuntimeError(f"No .npy spectrograms found under {self.spec_dir}")

        data = np.load(self.npz_path)
        if "emb" not in data:
            raise RuntimeError(f"NPZ {self.npz_path} missing key 'emb'")
        self.emb = data["emb"].astype(np.float32)  # (N, D)
        self.N, self.D = int(self.emb.shape[0]), int(self.emb.shape[1])

        if len(self.prompts) != self.N:
            raise RuntimeError(f"prompts count ({len(self.prompts)}) != embeddings rows ({self.N})")
        if len(self.spec_paths) < self.N:
            raise RuntimeError(f"spectrogram files ({len(self.spec_paths)}) < embeddings rows ({self.N})")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        emb = torch.from_numpy(self.emb[idx])  # (D,)
        p = self.spec_paths[idx]
        mel = np.load(str(p))
        if mel.ndim != 2:
            raise RuntimeError(f"Expected 2D mel/spec array, got {mel.shape} at {p}")
        if mel.shape == (1024, 256):
            mel = mel.T
        mel_t = torch.from_numpy(mel.astype(np.float32))  # (256,1024)
        meta = {"id": f"{idx:06d}", "path": str(p)}
        return mel_t, emb, meta


class PhonemeNPZDataset(Dataset):
    """
    Loads sentence embeddings from a single NPZ file (key 'emb') and pairs with spectrogram .npy files.
    """
    def __init__(self, prompts_file: str, spec_dir: str, npz_path: str):
        self.prompts_path = Path(prompts_file).resolve()
        self.spec_dir = Path(spec_dir).resolve()
        self.npz_path = Path(npz_path).resolve()

        if not self.prompts_path.exists():
            raise FileNotFoundError(self.prompts_path)
        if not self.spec_dir.exists():
            raise FileNotFoundError(self.spec_dir)
        if not self.npz_path.exists():
            raise FileNotFoundError(self.npz_path)

        lines = [ln.strip() for ln in self.prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.prompts = lines

        self.spec_paths: List[Path] = sorted([p for p in self.spec_dir.rglob("*.npy") if p.is_file()])
        if not self.spec_paths:
            raise RuntimeError(f"No .npy spectrograms found under {self.spec_dir}")

        data = np.load(self.npz_path)
        if "emb" not in data:
            raise RuntimeError(f"NPZ {self.npz_path} missing key 'emb'")
        self.emb = data["emb"].astype(np.float32)  # (N, D)
        self.N, self.D = int(self.emb.shape[0]), int(self.emb.shape[1])

        if len(self.prompts) != self.N:
            raise RuntimeError(f"prompts count ({len(self.prompts)}) != embeddings rows ({self.N})")
        if len(self.spec_paths) < self.N:
            raise RuntimeError(f"spectrogram files ({len(self.spec_paths)}) < embeddings rows ({self.N})")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        emb = torch.from_numpy(self.emb[idx])  # (D,)
        p = self.spec_paths[idx]
        mel = np.load(str(p))
        if mel.ndim != 2:
            raise RuntimeError(f"Expected 2D mel/spec array, got {mel.shape} at {p}")
        if mel.shape == (1024, 256):
            mel = mel.T
        mel_t = torch.from_numpy(mel.astype(np.float32))  # (256,1024)
        meta = {"id": f"{idx:06d}", "path": str(p)}
        return mel_t, emb, meta


class PhonemeAdapter(nn.Module):
    """
    Projects sentence-level phoneme embeddings (B, in_dim) to UNet cross-attn hidden states (B, seq_len, hidden_dim).
    """
    def __init__(self, in_dim: int, out_dim: int, seq_len: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.seq_len = int(seq_len)
        self.proj = nn.Linear(self.in_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> (B, seq_len, out_dim)
        y = self.proj(x.float())
        y = self.norm(y)
        y = y.unsqueeze(1).expand(-1, self.seq_len, -1)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--repo_id", default="auffusion/auffusion-full-no-adapter")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--cond_col", default="audio_prompt")
    ap.add_argument("--spec_col", default="mel_npy")
    ap.add_argument("--cond_type", choices=["text","phoneme"], default="text", help="conditioning type: text (tokenizer+encoder) or phoneme (sentence embeddings)")
    # Phoneme mode inputs (alternative to --manifest): align prompts with spectrogram directory and phoneme memmap
    ap.add_argument("--prompts_file", default=None, help="path to prompts.txt (1 per line) for indexing")
    ap.add_argument("--spec_dir", default=None, help="directory containing spectrogram .npy files (aligned with prompts)")
    ap.add_argument("--phoneme_npz", required=False, default=None, help="NPZ file with key 'emb' produced by data/generate_phonemes.py")

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--plot_loss_every", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in out_dir/checkpoints/")

    ap.add_argument("--seed", type=lambda x: None if x.lower() == "none" else int(x), default=1234,
                    help="Random seed (use 'none' for non-reproducible randomness)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    # Extend LoRA injection to MLP projections (feed-forward) and allow user overrides
    ap.add_argument(
        "--lora_targets",
        type=str,
        default="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2",
        help="Comma-separated Linear names to wrap with LoRA (matched by module name suffix)."
    )
    # Selectively unfreeze base UNet modules by name substrings (e.g., 'mid_block', 'up_blocks.3.attentions')
    ap.add_argument(
        "--unfreeze_patterns",
        type=str,
        default="",
        help="Comma-separated substrings; any UNet module name containing one is unfrozen (base weights)."
    )
    ap.add_argument(
        "--base_lr",
        type=float,
        default=2e-5,
        help="Learning rate for unfrozen base UNet parameters."
    )

    ap.add_argument("--mel_min", type=float, default=-11.0)
    ap.add_argument("--mel_max", type=float, default=0.0)

    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints")
    set_seed(args.seed)

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    # Pipeline weights can be fp16 for speed; LoRA weights remain fp32 by design.
    pipe_dtype = torch.float16 if (args.fp16 and use_cuda) else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        args.repo_id,
        torch_dtype=pipe_dtype,
        local_files_only=args.local_files_only,
    ).to(device)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    noise_scheduler = pipe.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_targets = tuple(s.strip() for s in str(args.lora_targets).split(",") if s.strip())
    trainable_lora = inject_lora_into_unet(
        unet,
        device=device,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=lora_targets,
    )
    print("[lora] injected LoRA into UNet attention linears (LoRA params fp32)")
    print(f"[lora] trainable params = {count_params(trainable_lora)}")

    # Optionally unfreeze additional base UNet params (e.g., middle block/late attentions/MLP)
    base_trainable: List[nn.Parameter] = []
    if args.unfreeze_patterns:
        patterns = [s.strip() for s in args.unfreeze_patterns.split(",") if s.strip()]
        base_trainable = unfreeze_unet_by_patterns(unet, patterns)
        if base_trainable:
            print(f"[unfreeze] additional UNet base params = {count_params(base_trainable)}")
        else:
            print("[unfreeze] no UNet modules matched patterns; keeping base frozen.")

    in_ch = int(getattr(vae.config, "in_channels", 3))
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    print(f"[vae] in_channels={in_ch} scaling_factor={scaling_factor}")

    # Build dataset
    if args.cond_type == "phoneme":
        if not (args.prompts_file and args.spec_dir and args.phoneme_npz):
            raise SystemExit("Phoneme mode requires --prompts_file, --spec_dir and --phoneme_npz")
        ds = PhonemeNPZDataset(args.prompts_file, args.spec_dir, args.phoneme_npz)
        in_dim = int(ds.D)
        # determine adapter sizes
        out_dim = int(getattr(text_encoder.config, "hidden_size", 768))
        seq_len = int(getattr(tokenizer, "model_max_length", 77))
        phoneme_adapter = PhonemeAdapter(in_dim=in_dim, out_dim=out_dim, seq_len=seq_len).to(device=device, dtype=torch.float32)
        # train adapter by default (in FP32)
        phoneme_adapter.train()
    else:
        if not args.manifest:
            raise SystemExit("Text mode requires --manifest CSV (with cond_col/spec_col).")
        ds = ManifestDataset(args.manifest, cond_col=args.cond_col, spec_col=args.spec_col)
        phoneme_adapter = None

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Optimizer over LoRA + (optional) phoneme adapter
    param_groups: List[Dict] = []
    if trainable_lora:
        param_groups.append({"params": list(trainable_lora), "lr": args.lr})
    if base_trainable:
        param_groups.append({"params": list(base_trainable), "lr": args.base_lr})
    if phoneme_adapter is not None:
        param_groups.append({"params": [p for p in phoneme_adapter.parameters() if p.requires_grad], "lr": args.lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    opt = torch.optim.AdamW(param_groups, lr=args.lr)

    # GradScaler is safe now because trainable params (LoRA) are FP32 → grads FP32
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and use_cuda))

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    step = 0
    # Optional resume from latest checkpoint
    if args.resume:
        latest_step = find_latest_step(out_dir)
        latest_dir = out_dir / "checkpoints" / "latest"
        if latest_step > 0 and latest_dir.exists():
            # load LoRA into injected wrappers
            lora_path = latest_dir / "lora.pt"
            try:
                sd = torch.load(str(lora_path), map_location="cpu")
                loaded = 0
                for k, v in sd.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    if k.endswith(".A.weight"):
                        modname = k[:-len(".A.weight")]
                        mod = dict(unet.named_modules()).get(modname, None)
                        if isinstance(mod, LoRALinear):
                            mod.A.weight.data.copy_(v.to(mod.A.weight.dtype)); loaded += 1
                    elif k.endswith(".B.weight"):
                        modname = k[:-len(".B.weight")]
                        mod = dict(unet.named_modules()).get(modname, None)
                        if isinstance(mod, LoRALinear):
                            mod.B.weight.data.copy_(v.to(mod.B.weight.dtype)); loaded += 1
                print(f"[resume] loaded LoRA tensors={loaded} from {lora_path}")
            except Exception as e:
                print(f"[resume] failed to load LoRA from {lora_path}: {e}")
            # load unfrozen base UNet weights if present
            base_path = latest_dir / "base_unet.pt"
            if base_path.exists():
                try:
                    bsd = torch.load(str(base_path), map_location="cpu")
                    named_params = dict(unet.named_parameters())
                    bloaded = 0
                    for k, v in bsd.items():
                        p = named_params.get(k, None)
                        if p is not None and isinstance(v, torch.Tensor):
                            with torch.no_grad():
                                p.copy_(v.to(dtype=p.dtype))
                            bloaded += 1
                    print(f"[resume] loaded base UNet params={bloaded} from {base_path}")
                except Exception as e:
                    print(f"[resume] failed to load base UNet from {base_path}: {e}")
            # load phoneme adapter if present
            if phoneme_adapter is not None:
                apath = latest_dir / "phoneme_adapter.pt"
                if apath.exists():
                    try:
                        asd = torch.load(str(apath), map_location="cpu")
                        if "state_dict" in asd:
                            phoneme_adapter.load_state_dict(asd["state_dict"], strict=False)
                            print(f"[resume] loaded phoneme adapter from {apath}")
                    except Exception as e:
                        print(f"[resume] failed to load phoneme adapter from {apath}: {e}")
            step = latest_step
            print(f"[resume] resuming from step={step}")
    it = iter(dl)
    t0 = time.time()
    print("[train] starting...")

    loss_csv_path = out_dir / "loss.csv"
    loss_png_path = out_dir / "loss.png"

    while step < args.steps:
        try:
            mel, conds, _ = next(it)
        except StopIteration:
            it = iter(dl)
            mel, conds, _ = next(it)

        mel = mel.to(device, dtype=torch.float32)
        vae_img = mel_to_vae_image(mel, in_channels=in_ch, mel_min=args.mel_min, mel_max=args.mel_max).to(device, dtype=pipe_dtype)

        if args.cond_type == "phoneme":
            # conds: Tensor (B, in_dim)
            if not isinstance(conds, torch.Tensor):
                raise RuntimeError("Phoneme mode expects tensor conditions from dataset")
            conds = conds.to(device=device, dtype=torch.float32)
            encoder_hidden_states = phoneme_adapter(conds)  # (B, seq_len, hidden)
            with torch.no_grad():
                latents = vae.encode(vae_img).latent_dist.sample() * scaling_factor
        else:
            # Text mode: conds is list[str]
            texts = conds
            tok = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tok.input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(vae_img).latent_dist.sample() * scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        # print(noise_scheduler.config.num_train_timesteps, timesteps)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.fp16 and use_cuda)):
            pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        step += 1

        # append loss to CSV
        try:
            append_loss_csv(loss_csv_path, step, float(loss.item()))
        except Exception:
            pass

        if step == 1 or step % 25 == 0:
            dt = time.time() - t0
            print(f"[train] step {step:6d}/{args.steps} loss={loss.item():.6f} sec/it={dt/max(step,1):.3f}")

        if step % args.save_every == 0 or step == args.steps:
            ckpt_dir = out_dir / "checkpoints" / f"step_{step:06d}"
            ensure_dir(ckpt_dir)

            state = extract_lora_state(unet)
            torch.save(state, ckpt_dir / "lora.pt")

            latest = out_dir / "checkpoints" / "latest"
            ensure_dir(latest)
            torch.save(state, latest / "lora.pt")

            # Save unfrozen base UNet weights (only params with requires_grad and not LoRA A/B)
            try:
                base_state = {}
                for name, p in unet.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name.endswith(".A.weight") or name.endswith(".B.weight"):
                        continue
                    base_state[name] = p.detach().cpu()
                if base_state:
                    torch.save(base_state, ckpt_dir / "base_unet.pt")
                    torch.save(base_state, latest / "base_unet.pt")
                    print(f"[save] wrote base_unet.pt with {len(base_state)} tensors")
                else:
                    print("[save] no unfrozen base UNet params to save")
            except Exception as e:
                print(f"[save] failed to save base UNet params: {e}")

            if phoneme_adapter is not None:
                adapter_state = {
                    "state_dict": phoneme_adapter.state_dict(),
                    "in_dim": getattr(phoneme_adapter, "in_dim", None),
                    "out_dim": getattr(phoneme_adapter, "out_dim", None),
                    "seq_len": getattr(phoneme_adapter, "seq_len", None),
                }
                torch.save(adapter_state, ckpt_dir / "phoneme_adapter.pt")
                torch.save(adapter_state, latest / "phoneme_adapter.pt")

            print(f"[save] wrote {ckpt_dir/'lora.pt'} and {latest/'lora.pt'}")

            # plot loss curve periodically
            try:
                if (step % max(1, args.plot_loss_every)) == 0 or step == args.steps:
                    plot_loss_curve(loss_csv_path, loss_png_path)
            except Exception:
                pass

    print("[train] done.")


if __name__ == "__main__":
    main()