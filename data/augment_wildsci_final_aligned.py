#!/usr/bin/env python3
"""Create WildSci_final_aligned.parquet from WildSci_not_enhanced.parquet.

For each original question, sample four different subsets of incorrect options,
append the correct option, keep original option labels, and rebuild prompt and
raw_prompt for the sampled option set.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEM_PROMPT = "Please solve the problem step by step, and put your final answer inside \\boxed{}."
DEFAULT_INPUT = Path(__file__).resolve().parent / "WildSci_not_enhanced.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "WildSci_final_aligned.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input parquet path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet path")
    parser.add_argument("--num-variants", type=int, default=4, help="Augmented rows per source question")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def option_sort_key(key: str) -> tuple[int, str]:
    key = str(key)
    if len(key) == 1 and key.isalpha():
        return ord(key.upper()) - ord("A"), key
    return 10_000, key


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    raise TypeError(f"Expected options to be dict, got {type(options).__name__}")


def format_user_prompt(question: str, options: dict[str, str]) -> str:
    lines = [str(question).strip(), "", "Options:"]
    for key in sorted(options, key=option_sort_key):
        lines.append(f"{key}. {options[key].strip()}")
    return "\n".join(lines).strip()


def sample_wrong_subsets(wrong_keys: list[str], num_variants: int, rng: random.Random) -> list[tuple[str, ...]]:
    if not wrong_keys:
        raise ValueError("Cannot augment a question without incorrect options")

    max_unique = (2 ** len(wrong_keys)) - 1
    if num_variants > max_unique:
        raise ValueError(f"Requested {num_variants} variants, but only {max_unique} unique wrong-option subsets exist")

    size_pool = list(range(1, len(wrong_keys) + 1))
    rng.shuffle(size_pool)

    subsets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(subsets) < num_variants:
        if len(subsets) < len(size_pool):
            subset_size = size_pool[len(subsets)]
        else:
            subset_size = rng.randint(1, len(wrong_keys))

        subset = tuple(sorted(rng.sample(wrong_keys, subset_size), key=option_sort_key))
        attempts += 1
        if subset in seen:
            if attempts > 10_000:
                raise RuntimeError("Failed to sample unique option subsets")
            continue
        seen.add(subset)
        subsets.append(subset)
    return subsets


def build_augmented_records(df: pd.DataFrame, num_variants: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for source_index, row in df.iterrows():
        options = normalize_options(row["options"])
        answer = str(row["answer"]).strip()
        if answer not in options:
            raise ValueError(f"Row {source_index} answer {answer!r} is not present in options")

        wrong_keys = [key for key in sorted(options, key=option_sort_key) if key != answer]
        wrong_subsets = sample_wrong_subsets(wrong_keys, num_variants, rng)

        for variant_index, wrong_subset in enumerate(wrong_subsets):
            selected_keys = sorted((answer, *wrong_subset), key=option_sort_key)
            selected_options = {key: options[key] for key in selected_keys}
            user_prompt = format_user_prompt(row["question"], selected_options)
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            record = row.to_dict()
            base_uid = str(record.get("uid") or f"wildsci-{source_index}")
            record.update(
                {
                    "uid": f"{base_uid}-aug{variant_index}",
                    "options": selected_options,
                    "prompt": chat,
                    "raw_prompt": chat,
                    "ground_truth": answer,
                    "reward_model": {"ground_truth": answer},
                    "extra_info": {
                        **(record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}),
                        "source_index": int(source_index),
                        "source_uid": base_uid,
                        "augment_index": variant_index,
                        "selected_option_keys": selected_keys,
                        "sampled_wrong_option_keys": list(wrong_subset),
                    },
                }
            )
            records.append(record)

    return records


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input)
    records = build_augmented_records(df, args.num_variants, args.seed)

    out = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    print(f"Read {len(df)} rows from {args.input}")
    print(f"Wrote {len(out)} rows to {args.output}")
    print("Rows by discipline:")
    print(out["discipline"].value_counts().sort_index().to_string())
    print("Option-count distribution:")
    print(out["options"].map(len).value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
