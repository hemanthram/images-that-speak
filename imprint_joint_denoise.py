#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from diffusers import DDIMScheduler
from huggingface_hub import snapshot_download
from transformers import AlbertConfig, AlbertModel
import yaml
from phonemizer import phonemize
from phonemizer.separator import Separator
from tqdm import tqdm
# Local modules
from src.guidance.stable_diffusion import StableDiffusionGuidance
from src.guidance.auffusion import AuffusionGuidance
from src.transformation.identity import NaiveIdentity


class LoRALinear(nn.Module):
	def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
		super().__init__()
		self.base = base
		for p in self.base.parameters():
			p.requires_grad = False

		self.rank = int(rank)
		self.alpha = float(alpha)
		self.scale = self.alpha / max(1, self.rank)
		self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

		self.A = nn.Linear(base.in_features, self.rank, bias=False)
		self.B = nn.Linear(self.rank, base.out_features, bias=False)
		nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
		nn.init.zeros_(self.B.weight)

	def forward(self, x, *args, **kwargs):
		base_out = self.base(x)
		x32 = self.drop(x).float()
		lora_out = self.B(self.A(x32)) * self.scale
		return base_out + lora_out.to(dtype=base_out.dtype)


def _set_module(root: nn.Module, name: str, new_module: nn.Module):
	parts = name.split(".")
	parent = root
	for p in parts[:-1]:
		parent = getattr(parent, p)
	setattr(parent, parts[-1], new_module)


def inject_lora_wrappers(unet: nn.Module, device: torch.device, rank=8, alpha=16.0, dropout=0.0,
						 targets=("to_q", "to_k", "to_v", "to_out.0")):
	for p in unet.parameters():
		p.requires_grad = False

	to_wrap = []
	for name, mod in unet.named_modules():
		if isinstance(mod, nn.Linear):
			if any(name.endswith(t) for t in targets):
				to_wrap.append((name, mod))

	if not to_wrap:
		raise RuntimeError("No target Linear layers found to wrap with LoRA (to_q/to_k/to_v/to_out.0).")

	for name, base in to_wrap:
		wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
		wrapped.A = wrapped.A.to(device=device, dtype=torch.float32)
		wrapped.B = wrapped.B.to(device=device, dtype=torch.float32)
		_set_module(unet, name, wrapped)


def load_lora_state_into_unet(unet: nn.Module, lora_path: str):
	sd = torch.load(lora_path, map_location="cpu")
	loaded = 0
	missing = 0
	for k, v in sd.items():
		if not isinstance(v, torch.Tensor):
			continue
		if k.endswith(".A.weight"):
			modname = k[:-len(".A.weight")]
			mod = dict(unet.named_modules()).get(modname, None)
			if isinstance(mod, LoRALinear):
				mod.A.weight.data.copy_(v.to(mod.A.weight.dtype))
				loaded += 1
			else:
				missing += 1
		elif k.endswith(".B.weight"):
			modname = k[:-len(".B.weight")]
			mod = dict(unet.named_modules()).get(modname, None)
			if isinstance(mod, LoRALinear):
				mod.B.weight.data.copy_(v.to(mod.B.weight.dtype))
				loaded += 1
			else:
				missing += 1
	if loaded == 0:
		raise RuntimeError(f"Loaded 0 LoRA tensors from {lora_path}. Wrong file?")
	print(f"[lora] loaded tensors={loaded} missing={missing} from {lora_path}")


class PhonemeAdapter(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, seq_len: int):
		super().__init__()
		self.in_dim = int(in_dim)
		self.out_dim = int(out_dim)
		self.seq_len = int(seq_len)
		self.proj = nn.Linear(self.in_dim, self.out_dim)
		self.norm = nn.LayerNorm(self.out_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.proj(x.float())
		y = self.norm(y)
		y = y.unsqueeze(1).expand(-1, self.seq_len, -1)
		return y


def load_phoneme_adapter(adapter_path: str, device: torch.device) -> PhonemeAdapter:
	sd = torch.load(adapter_path, map_location="cpu")
	in_dim = int(sd.get("in_dim", 0))
	out_dim = int(sd.get("out_dim", 768))
	seq_len = int(sd.get("seq_len", 77))
	model = PhonemeAdapter(in_dim=in_dim, out_dim=out_dim, seq_len=seq_len)
	model.load_state_dict(sd["state_dict"], strict=True)
	model.eval().to(device=device, dtype=torch.float32)
	return model


def _load_yaml(path: Path):
	return yaml.safe_load(path.read_text())


def _pick_latest_step_t7(model_dir: Path) -> Path:
	t7s = sorted(model_dir.glob("step_*.t7"))
	if not t7s:
		raise FileNotFoundError(f"No step_*.t7 found in {model_dir}")
	def step_num(p: Path):
		try:
			return int(p.stem.split("_")[-1])
		except Exception:
			return -1
	return max(t7s, key=step_num)


def _load_token_maps(pkl_path: Path):
	import pickle
	obj = pickle.loads(pkl_path.read_bytes())
	if isinstance(obj, dict):
		if "token2id" in obj:
			token2id = obj["token2id"]
		elif "token_to_id" in obj:
			token2id = obj["token_to_id"]
		else:
			token2id = obj
	elif isinstance(obj, (tuple, list)) and len(obj) >= 1:
		token2id = obj[0]
	else:
		raise ValueError(f"Unrecognized token map format: {type(obj)}")
	def get_id(*cands, default=None):
		for c in cands:
			if c in token2id:
				return token2id[c]
		return default
	pad_id = get_id("<pad>", "[PAD]", "PAD", default=0)
	unk_id = get_id("<unk>", "[UNK]", "UNK", default=1)
	cls_id = get_id("[CLS]", "<s>", "CLS", default=None)
	sep_id = get_id("[SEP]", "</s>", "SEP", default=None)
	return token2id, pad_id, unk_id, cls_id, sep_id


def _phonemize_text(text: str, lang: str) -> list[list[str]]:
	WORD_SEP = " <w> "
	ph = phonemize(
		text,
		language=lang,
		backend="espeak",
		strip=True,
		preserve_punctuation=True,
		with_stress=True,
		separator=Separator(phone=" ", word=WORD_SEP, syllable=""),
	).strip()
	if not ph:
		return [[]]
	words = []
	for w in ph.split("<w>"):
		toks = [t for t in w.strip().split() if t]
		words.append(toks)
	return words


def _words_to_ids(words: list[list[str]], token2id: dict, unk_id: int, word_sep_id: int,
				  cls_id=None, sep_id=None, max_len: int = 512) -> list[int]:
	ids = []
	if cls_id is not None:
		ids.append(cls_id)
	for wi, w in enumerate(words):
		for p in w:
			ids.append(token2id.get(p, unk_id))
		if wi != len(words) - 1:
			ids.append(word_sep_id)
	if sep_id is not None:
		ids.append(sep_id)
	return ids[:max_len]


@torch.no_grad()
def encode_phoneme_sentence(
	prompt: str,
	model_id: str,
	lang: str,
	device: torch.device,
	max_len: int = 512,
) -> torch.Tensor:
	local_dir = Path(snapshot_download(repo_id=model_id))
	cfg = _load_yaml(local_dir / "config.yml")
	word_sep_id = int(cfg["dataset_params"]["word_separator"])
	token_maps_rel = cfg["dataset_params"]["token_maps"]
	token_maps_path = local_dir / token_maps_rel
	token2id, pad_id, unk_id, cls_id, sep_id = _load_token_maps(token_maps_path)

	albert_cfg = AlbertConfig(**cfg["model_params"])
	model = AlbertModel(albert_cfg)
	ckpt_path = _pick_latest_step_t7(local_dir)
	ckpt = torch.load(ckpt_path, map_location="cpu")
	if isinstance(ckpt, dict):
		if "net" in ckpt:
			state = ckpt["net"]
		elif "state_dict" in ckpt:
			state = ckpt["state_dict"]
		else:
			state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
	else:
		state = ckpt
	cleaned = {}
	for k, v in state.items():
		nk = k
		if nk.startswith("module."):
			nk = nk[len("module."):]
		if nk.startswith("encoder."):
			nk = nk[len("encoder."):]
		cleaned[nk] = v
	cleaned.pop("embeddings.position_ids", None)
	model.load_state_dict(cleaned, strict=False)
	model.eval().to(device)

	words = _phonemize_text(prompt, lang)
	ids = _words_to_ids(words, token2id, unk_id, word_sep_id, cls_id, sep_id, max_len=max_len)
	if not ids:
		ids = [pad_id]
	input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
	attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
	out = model(input_ids=input_ids, attention_mask=attn)
	token_emb = out.last_hidden_state  # [1,T,H]
	mask = attn.unsqueeze(-1).to(token_emb.dtype)
	sent = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # [1,H]
	return sent.cpu()


@torch.no_grad()
def encode_prompt(prompt: str, diffusion_guidance, device: torch.device, negative_prompt: str = '', time_repeat: int = 1):
	prompts = [prompt] * time_repeat
	negative_prompts = [negative_prompt] * time_repeat
	cond_embeds = diffusion_guidance.get_text_embeds(prompts, device)        # [B, 77, 768]
	uncond_embeds = diffusion_guidance.get_text_embeds(negative_prompts, device)  # [B, 77, 768]
	text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)             # [2B, 77, 768]
	return text_embeds


def estimate_noise(diffusion, latents, t, text_embeddings, guidance_scale):
	latent_model_input = torch.cat([latents] * 2)
	noise_pred = diffusion.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
	noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
	noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
	return noise_pred


@torch.no_grad()
def main():
	ap = argparse.ArgumentParser(description="Joint denoising to generate image + spectrogram using trained Auffusion LoRA.")
	ap.add_argument("--image_prompt", required=True)
	ap.add_argument("--audio_prompt", required=True)

	ap.add_argument("--sd_repo_id", default="runwayml/stable-diffusion-v1-5")
	ap.add_argument("--auffusion_repo_id", default="auffusion/auffusion-full-no-adapter")
	ap.add_argument("--lora_path", required=True, help="Path to checkpoints/latest/lora.pt from training")
	ap.add_argument("--base_unet_path", default=None, help="Optional path to checkpoints/latest/base_unet.pt for unfrozen UNet weights")
	ap.add_argument("--adapter_path", default=None, help="Path to checkpoints/latest/phoneme_adapter.pt")
	ap.add_argument("--cond_type", choices=["text","phoneme"], default="text")
	ap.add_argument("--phoneme_npz", default=None, help="Optional NPZ with 'emb' (N,D) for phoneme mode")
	ap.add_argument("--prompt_index", type=int, default=None, help="Row in NPZ to use (if provided)")
	ap.add_argument("--plbert_model_id", default="papercup-ai/multilingual-pl-bert")
	ap.add_argument("--ph_lang", default="en-us")
	# Match training defaults: include MLP projections in LoRA targets
	ap.add_argument("--lora_targets", type=str, default="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2",
					help="Comma-separated Linear names to wrap with LoRA (matched by module name suffix).")

	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--steps", type=int, default=100)

	ap.add_argument("--image_guidance", type=float, default=10.0)
	ap.add_argument("--audio_guidance", type=float, default=10.0)
	ap.add_argument("--image_start_step", type=int, default=10)
	ap.add_argument("--audio_start_step", type=int, default=0)
	ap.add_argument("--audio_weight", type=float, default=0.5)

	ap.add_argument("--img_height", type=int, default=256)
	ap.add_argument("--img_width", type=int, default=1024)

	ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
	ap.add_argument("--fp16", action="store_true")
	ap.add_argument("--seed", type=lambda x: None if x.lower() == "none" else int(x), default=None,
				help="Random seed (use 'none' for non-reproducible randomness)")

	ap.add_argument("--cutoff_latent", action="store_true")
	ap.add_argument("--crop_image", action="store_true")
	ap.add_argument("--use_colormap", action="store_true")

	args = ap.parse_args()

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	use_cuda = (args.device == "cuda" and torch.cuda.is_available())
	device = torch.device("cuda" if use_cuda else "cpu")
	dtype = torch.float16 if (args.fp16 and use_cuda) else torch.float32

	# Instantiate guidance models
	image_diff = StableDiffusionGuidance(repo_id=args.sd_repo_id, fp16=args.fp16).to(device)
	audio_diff = AuffusionGuidance(repo_id=args.auffusion_repo_id, fp16=args.fp16).to(device)

	# Inject/load LoRA into the audio (Auffusion) UNet
	targets = tuple(s.strip() for s in str(args.lora_targets).split(",") if s.strip())
	inject_lora_wrappers(audio_diff.unet, device=device, rank=8, alpha=16.0, dropout=0.0, targets=targets)
	# Load unfrozen base UNet weights if provided (after injection so keys with '.base.' resolve)
	if args.base_unet_path:
		try:
			bsd = torch.load(args.base_unet_path, map_location="cpu")
			named_params = dict(audio_diff.unet.named_parameters())
			loaded = 0
			for k, v in bsd.items():
				p = named_params.get(k, None)
				if p is not None and isinstance(v, torch.Tensor):
					with torch.no_grad():
						p.copy_(v.to(dtype=p.dtype))
					loaded += 1
			print(f"[base] loaded base UNet tensors={loaded} from {args.base_unet_path}")
		except Exception as e:
			print(f"[base] failed to load base UNet from {args.base_unet_path}: {e}")
	load_lora_state_into_unet(audio_diff.unet, args.lora_path)

	# Shared scheduler (match configs/main_denoise)
	scheduler = DDIMScheduler.from_pretrained(
		args.sd_repo_id,
		subfolder="scheduler",
		torch_dtype=dtype
	)

	latent_transform = NaiveIdentity().to(device)

	generator = torch.Generator(device=device)
	if args.seed is not None:
		generator.manual_seed(args.seed)

	# Encode prompts
	image_text = encode_prompt(args.image_prompt, image_diff, device, negative_prompt='', time_repeat=1)
	if args.cond_type == "phoneme":
		if not args.adapter_path:
			raise SystemExit("cond_type=phoneme requires --adapter_path")
		adapter = load_phoneme_adapter(args.adapter_path, device=device)
		if args.phoneme_npz is not None and args.prompt_index is not None:
			npz = np.load(args.phoneme_npz)
			if "emb" not in npz:
				raise SystemExit(f"NPZ missing 'emb': {args.phoneme_npz}")
			emb_arr = npz["emb"]
			if args.prompt_index < 0 or args.prompt_index >= emb_arr.shape[0]:
				raise SystemExit(f"prompt_index out of range 0..{emb_arr.shape[0]-1}")
			emb_pos_in = torch.from_numpy(emb_arr[args.prompt_index]).to(device=device, dtype=torch.float32).unsqueeze(0)
		else:
			emb_pos_in = encode_phoneme_sentence(
				prompt=args.audio_prompt or "",
				model_id=args.plbert_model_id,
				lang=args.ph_lang,
				device=device,
				max_len=512,
			).to(device=device, dtype=torch.float32)
		emb_uncond_in = torch.zeros_like(emb_pos_in)
		emb_pair = torch.cat([emb_uncond_in, emb_pos_in], dim=0)  # [2, in_dim]
		audio_text = adapter(emb_pair).to(device=device, dtype=audio_diff.unet.dtype)  # [2,77,hidden]
	else:
		audio_text = encode_prompt(args.audio_prompt, audio_diff, device, negative_prompt='', time_repeat=1)

	scheduler.set_timesteps(args.steps)

	# Init latents
	B = image_text.shape[0] // 2
	latents = torch.randn(
		(B, image_diff.unet.config.in_channels, args.img_height // 8, args.img_width // 8),
		generator=generator,
		dtype=image_diff.precision_t,
		device=device
	)

	for i, t in tqdm(enumerate(scheduler.timesteps)):
		image_noise = None
		audio_noise = None

		if i >= args.image_start_step:
			image_noise = estimate_noise(image_diff, latents, t, image_text, args.image_guidance)

		if i >= args.audio_start_step:
			tlat = latent_transform(latents, inverse=False)
			audio_noise = estimate_noise(audio_diff, tlat, t, audio_text, args.audio_guidance)
			audio_noise = latent_transform(audio_noise, inverse=True)

		if image_noise is not None and audio_noise is not None:
			noise_pred = (1.0 - args.audio_weight) * image_noise + args.audio_weight * audio_noise
		elif image_noise is not None:
			noise_pred = image_noise
		elif audio_noise is not None:
			noise_pred = audio_noise
		else:
			raise RuntimeError("Neither image nor audio guidance active at this step.")

		latents = scheduler.step(noise_pred, t, latents)['prev_sample']

	# Optional latent cut/crop
	if args.cutoff_latent and not args.crop_image:
		latents = latents[..., :-4]

	# Decode outputs
	img = image_diff.decode_latents(latents)                 # [1, 3, H, W] in 0..1
	audio_latents = latent_transform(latents, inverse=False)
	spec = audio_diff.decode_latents(audio_latents).squeeze(0)  # [3, 256, 1024] in 0..1

	if args.crop_image and not args.cutoff_latent:
		pixel = 32
		img = img[..., :-pixel]
		spec = spec[..., :-pixel]

	# Prepare mean-channel spectrogram [256, 1024] in 0..1
	spec_mean = spec.mean(dim=0)

	# Save image and spectrogram preview (normalized 0..1)
	save_image(img, str(out_dir / "img.png"))
	save_image(spec_mean.unsqueeze(0), str(out_dir / "spec.png"))

	if args.use_colormap:
		import matplotlib.pyplot as plt
		spec_np = spec_mean.cpu().numpy()
		plt.imsave(str(out_dir / "spec_colormap.png"), spec_np, cmap='gray')

	# Also save the spectrogram as log-mel .npy in [-11, 0] (mean channel),
	# matching infer_from_prompt.py and expected by decode_spec_to_audio.py
	mel = -11.0 + spec_mean * 11.0
	np.save(str(out_dir / "spec.npy"), mel.cpu().numpy())

	print(f"[done] wrote {out_dir/'img.png'}, {out_dir/'spec.png'} and {out_dir/'spec.npy'}")


if __name__ == "__main__":
	main()


