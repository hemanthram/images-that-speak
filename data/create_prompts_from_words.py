import random

def generate_prompts(words_file_path, prompts_file_path, L, W):
    # Read words from file, stripping whitespace and skipping empty lines
    with open(words_file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    seen = set()
    prompts = []
    tries = 0
    max_tries = L * 30  # Prevent infinite loops for high L,W combos

    while len(prompts) < L and tries < max_tries:
        chosen = tuple(sorted(random.sample(words, W)))
        if chosen not in seen:
            seen.add(chosen)
            prompts.append(' '.join(chosen))
        tries += 1

    if len(prompts) < L:
        raise RuntimeError(f"Could only generate {len(prompts)} unique prompts; increase 'words.txt' or decrease L/W")

    with open(prompts_file_path, 'w', encoding='utf-8') as out:
        for line in prompts:
            out.write(line + '\n')

if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate L unique prompts from a word list, each with W words."
    )
    parser.add_argument("L", type=int, help="Number of prompts to generate.")
    parser.add_argument("W", type=int, help="Number of words per prompt.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prompts.txt",
        help="Output prompts file name (default: data/prompts.txt)"
    )
    parser.add_argument(
        "-w",
        "--words",
        type=str,
        default="words.txt",
        help="Input words file name (default: data/words.txt)"
    )

    args = parser.parse_args()

    generate_prompts(args.words, args.output, args.L, args.W)
