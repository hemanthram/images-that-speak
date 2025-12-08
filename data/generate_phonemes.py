#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from phonemizer import phonemize
from phonemizer.separator import Separator
from transformers import AlbertConfig, AlbertModel
import json
from tqdm import tqdm
import os
import time
import gc


MODEL_ID_DEFAULT = "papercup-ai/multilingual-pl-bert"


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def pick_latest_step_t7(model_dir: Path) -> Path:
    t7s = sorted(model_dir.glob("step_*.t7"))
    if not t7s:
        raise FileNotFoundError(f"No step_*.t7 found in {model_dir}")
    # highest step number
    def step_num(p: Path):
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return -1
    return max(t7s, key=step_num)


def load_token_maps(pkl_path: Path):
    obj = pickle.loads(pkl_path.read_bytes())

    # Common formats: dict w/ token2id, or token2id directly, or tuple(token2id, id2token)
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

    # attempt special tokens (fallbacks are safe)
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


def text_to_phoneme_words(text: str, lang: str) -> list[list[str]]:
    """
    Uses espeak phonemizer to produce phonemes.
    Returns list-of-words, each word is a list of phoneme tokens.
    """
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


def phonemize_texts(texts: list[str], lang: str, njobs: int = 1) -> list[list[list[str]]]:
    """
    Batch phonemize multiple texts using phonemizer's internal parallelization.
    Returns a list of list-of-words per input text.
    """
    if not texts:
        return []
    WORD_SEP = " <w> "
    ph_list = phonemize(
        texts,
        language=lang,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        separator=Separator(phone=" ", word=WORD_SEP, syllable=""),
        njobs=max(1, int(njobs)),
    )
    out: list[list[list[str]]] = []
    for ph in ph_list:
        ph = (ph or "").strip()
        if not ph:
            out.append([[]])
            continue
        words: list[list[str]] = []
        for w in ph.split("<w>"):
            toks = [t for t in w.strip().split() if t]
            words.append(toks)
        out.append(words)
    return out


def words_to_ids(words: list[list[str]], token2id: dict, unk_id: int, word_sep_id: int,
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
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # [B,T,H], [B,T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def encode_texts(
    texts: list[str],
    model: AlbertModel,
    token2id: dict,
    pad_id: int,
    unk_id: int,
    cls_id,
    sep_id,
    word_sep_id: int,
    lang: str,
    device: torch.device,
    max_len: int,
    njobs: int = 1,
):
    # Prefer batched phonemization; fallback per-text on error
    try:
        words_batch = phonemize_texts(texts, lang, njobs=njobs)
    except Exception:
        # Retry once with njobs=1
        time.sleep(0.2)
        words_batch = phonemize_texts(texts, lang, njobs=1)

    seqs: list[list[int]] = []
    phoneme_strs: list[str] = []
    for words in words_batch:
        phoneme_strs.append(" | ".join(" ".join(w) for w in words))
        ids = words_to_ids(words, token2id, unk_id, word_sep_id, cls_id, sep_id, max_len=max_len)
        if len(ids) == 0:
            ids = [pad_id]
        seqs.append(ids)

    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1

    input_ids = input_ids.to(device)
    attn = attn.to(device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn)
    token_emb = out.last_hidden_state              # [B,T,H]
    sent_emb = mean_pool(token_emb, attn)          # [B,H]
    return phoneme_strs, token_emb.cpu(), sent_emb.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=MODEL_ID_DEFAULT)
    ap.add_argument("--lang", default="en-us", help="phonemizer espeak language, e.g. en-us")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--text", default=None, help="encode a single string")
    ap.add_argument("--prompts", default=None, help="path to prompts.txt (one per line)")
    # Output NPZ only
    ap.add_argument("--out_npz", default="plbert_sentence_emb.npz", help="output .npz file (stores 'emb' array)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--njobs", type=int, default=max(1, os.cpu_count() or 1), help="CPU workers for phonemizer batch mode")
    args = ap.parse_args()

    if args.text is None and args.prompts is None:
        raise SystemExit("Provide --text or --prompts")

    # device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # download model snapshot
    local_dir = Path(snapshot_download(repo_id=args.model_id))
    cfg = load_yaml(local_dir / "config.yml")

    word_sep_id = int(cfg["dataset_params"]["word_separator"])
    token_maps_rel = cfg["dataset_params"]["token_maps"]
    token_maps_path = local_dir / token_maps_rel

    token2id, pad_id, unk_id, cls_id, sep_id = load_token_maps(token_maps_path)

    # build Albert model & load ckpt
    albert_cfg = AlbertConfig(**cfg["model_params"])
    model = AlbertModel(albert_cfg)

    ckpt_path = pick_latest_step_t7(local_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # handle different checkpoint structures
    state = None
    if isinstance(ckpt, dict):
        if "net" in ckpt:
            state = ckpt["net"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # maybe directly a state dict
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        state = ckpt

    # strip common prefixes
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

    # single text
    if args.text is not None:
        _, _, sent = encode_texts(
            [args.text],
            model, token2id,
            pad_id, unk_id, cls_id, sep_id,
            word_sep_id,
            args.lang, device, args.max_len
        )
        print("sentence_embedding:", tuple(sent.shape))
        return

    # prompts.txt -> sentence embedding memmap
    prompts_path = Path(args.prompts)
    # count lines (non-empty) for allocation
    with prompts_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = len(lines)
    hidden = albert_cfg.hidden_size

    emb = np.zeros((n, hidden), dtype=np.float32)

    for i in tqdm(range(0, n, args.batch_size)):
        batch = lines[i:i + args.batch_size]
        _, _, sent = encode_texts(
            batch, model, token2id, pad_id, unk_id, cls_id, sep_id,
            word_sep_id, args.lang, device, args.max_len, args.njobs
        )
        emb[i:i + len(batch)] = sent.numpy().astype(np.float32)
        # Free cached memory to avoid long-run stalls
        del sent
        gc.collect()

    np.savez(
        args.out_npz,
        emb=emb,
        model_id=args.model_id,
        ckpt=ckpt_path.name,
        lang=args.lang,
    )
    print("Wrote:", args.out_npz, "shape=", (n, hidden))


if __name__ == "__main__":
    main()