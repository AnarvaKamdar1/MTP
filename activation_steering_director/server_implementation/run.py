"""CLI runner for the DIRECTER PICO extraction pipeline."""

import argparse
import json
import os
import time

from config import DirecterConfig
from model_io import load_model_and_tokenizer
from pipeline import PICODirecterPipeline


def load_input_file(file_path: str):
    """Loads input abstracts from either .jsonl or .json format."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            # Standard JSON array
            rows = json.loads(content)
        else:
            # JSON Lines format
            for line in content.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run DIRECTER on PICO dataset")
    parser.add_argument("--input", required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--model", default=DirecterConfig.model_id, help="HuggingFace model ID")
    parser.add_argument("--device", default="cuda", help="Target device (cuda / cpu)")
    parser.add_argument("--alpha", type=float, default=100.0, help="Key scale factor (alpha)")
    parser.add_argument("--beta", type=float, default=0.5, help="Plausibility threshold (beta)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--steer_format", action="store_true", help="Also steer the output format sentence")
    parser.add_argument("--no_general_rules", action="store_true", help="Exclude shared general rules from steering")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of abstracts to process")
    parser.add_argument("--save_raw", action="store_true", help="Save raw generation strings alongside parsed spans")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose terminal logging")

    args = parser.parse_args()

    cfg = DirecterConfig(
        model_id=args.model,
        device=args.device,
        attn_implementation=args.attn,
        key_scale_factor=args.alpha,
        plausibility=args.beta,
        max_new_tokens=args.max_new_tokens,
        steer_general_rules=not args.no_general_rules,
        steer_output_format=args.steer_format,
        verbose=args.verbose,
    )

    print("=" * 60)
    print(f"[*] Initializing DIRECTER Pipeline")
    print(f"    - Model: {cfg.model_id}")
    print(f"    - Alpha (Key Scale): {cfg.key_scale_factor}")
    print(f"    - Beta (Plausibility): {cfg.plausibility}")
    print(f"    - Attention: {cfg.attn_implementation}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(cfg)
    pipe = PICODirecterPipeline(model, tokenizer, cfg)

    rows = load_input_file(args.input)
    if args.limit:
        rows = rows[:args.limit]

    print(f"[*] Processing {len(rows)} abstracts...")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for n, row in enumerate(rows, 1):
            abstract_text = row.get("abstract") or row.get("text", "")
            if not abstract_text:
                continue

            res = pipe.extract(abstract_text, return_raw=args.save_raw)
            payload = dict(row)

            if args.save_raw:
                payload["pico_directer"] = res["pico"]
                payload["raw_directer"] = res["raw"]
            else:
                payload["pico_directer"] = res

            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out.flush()

            elapsed = time.time() - t0
            avg_per_doc = elapsed / n
            print(f"[{n}/{len(rows)}] ID: {row.get('id', 'N/A')} (Avg: {avg_per_doc:.1f}s/doc)")

    print("=" * 60)
    print(f"[*] Finished processing in {time.time() - t0:.1f}s -> {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()