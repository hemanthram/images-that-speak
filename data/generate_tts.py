import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio from a prompts file without requiring a speaker sample."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("prompts.txt"),
        help="Path to a UTF-8 text file with one prompt per line.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audio"),
        help="Directory where generated WAV files and manifest.csv will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help=(
            "Coqui TTS model name. Examples:\n"
            "- tts_models/en/ljspeech/tacotron2-DDC (natural single-speaker)\n"
            "- tts_models/en/ljspeech/glow-tts (fast single-speaker)\n"
            "- tts_models/en/vctk/vits (multi-speaker model; speaker selection removed)\n"
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for multilingual models (ignored by mono-lingual models).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based start line index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="1-based end line index (inclusive). Defaults to the last line.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip synthesis if output WAV already exists.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_tts_kwargs(tts, language: Optional[str]) -> dict:
    kwargs: dict = {}
    is_multi_lingual = bool(getattr(tts, "is_multi_lingual", False))
    if is_multi_lingual and language:
        kwargs["language"] = language
    return kwargs


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    manifest_path = args.out_dir / "manifest.csv"

    from TTS.api import TTS  # type: ignore

    # Always use GPU as requested
    tts = TTS(args.model, gpu=True)

    if not args.prompts.exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts}")

    text_lines = args.prompts.read_text(encoding="utf-8").splitlines()
    total = len(text_lines)
    start_idx = max(1, args.start)
    end_idx = total if args.end is None else min(args.end, total)
    if start_idx > end_idx:
        print("Nothing to process (start > end).")
        return

    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow(["index", "wav_path", "text", "model", "language"])

        tts_kwargs_template = build_tts_kwargs(tts, args.language)

        for idx in range(start_idx, end_idx + 1):
            text = text_lines[idx - 1].strip()
            if not text:
                continue

            wav_path = args.out_dir / f"{idx:06d}.wav"
            if args.skip_existing and wav_path.exists():
                writer.writerow([idx, str(wav_path), text, args.model, args.language or ""])
                continue

            tts_kwargs = dict(tts_kwargs_template)
            tts.tts_to_file(
                text=text,
                file_path=str(wav_path),
                **tts_kwargs,
            )
            writer.writerow([idx, str(wav_path), text, args.model, args.language or ""])

    print(
        f"Done. Generated audio for lines {start_idx}..{end_idx} of {total}.\n"
        f"Output directory: {args.out_dir}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()


