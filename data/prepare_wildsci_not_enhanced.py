#!/usr/bin/env python3
"""Sample and format JustinTX/WildSci for verl training.

This script downloads WildSci from HuggingFace, samples 500 examples from each
`discipline`, formats multiple-choice questions as chat prompts, and writes the
result to `data/WildSci_not_enhanced.parquet` by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEM_PROMPT = "Please solve the problem step by step, and put your final answer inside \\boxed{}."
DATASET_URL = "https://huggingface.co/datasets/JustinTX/WildSci/resolve/main/wildsci.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "WildSci_not_enhanced.parquet"
DEFAULT_CACHE = Path(__file__).resolve().parent / ".wildsci_cache" / "wildsci.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATASET_URL, help="WildSci JSONL URL on HuggingFace")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Local JSONL cache path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet path")
    parser.add_argument("--per-discipline", type=int, default=500, help="Rows sampled per discipline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force-download", action="store_true", help="Download even if cache exists")
    return parser.parse_args()


def download(url: str, cache_path: Path, force: bool = False) -> None:
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
        print(f"Using cached dataset: {cache_path}")
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    print(f"Downloading {url}")
    print(f"Saving cache to {cache_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "Project1-WildSci-prep/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp_path.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total and downloaded >= next_report:
                pct = downloaded * 100 / total
                print(f"  {downloaded / (1024**2):.1f} MiB / {total / (1024**2):.1f} MiB ({pct:.1f}%)")
                next_report += max(total // 20, 1)
            elif not total and downloaded // (50 * 1024**2) > (downloaded - len(chunk)) // (50 * 1024**2):
                print(f"  {downloaded / (1024**2):.1f} MiB")
    tmp_path.replace(cache_path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def option_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    key = str(item[0])
    if len(key) == 1 and key.isalpha():
        return ord(key.upper()) - ord("A"), key
    return 10_000, key


def format_user_prompt(question: str, options: dict[str, Any]) -> str:
    lines = [question.strip(), "", "Options:"]
    for key, value in sorted(options.items(), key=option_sort_key):
        lines.append(f"{key}. {str(value).strip()}")
    return "\n".join(lines).strip()


def build_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    discipline = str(row["discipline"])
    answer = str(row["answer"]).strip()
    user_prompt = format_user_prompt(str(row["question"]), row["options"])
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    record = dict(row)
    record.update(
        {
            "uid": f"wildsci-{discipline.replace(' ', '_').lower()}-{index}",
            "data_source": "JustinTX/WildSci",
            "ability": discipline,
            "prompt": chat,
            "raw_prompt": chat,
            "ground_truth": answer,
            "reward_model": {"ground_truth": answer},
            "extra_info": {
                "index": index,
                "paper_id": row.get("paper_id"),
                "discipline": discipline,
                "nc_domain": row.get("nc_domain"),
                "nc_subdomain": row.get("nc_subdomain"),
            },
        }
    )
    return record


def sample_by_discipline(rows: list[dict[str, Any]], per_discipline: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        missing = [name for name in ("discipline", "question", "options", "answer") if name not in row]
        if missing:
            raise KeyError(f"Row is missing required fields: {missing}")
        if not isinstance(row["options"], dict):
            raise TypeError(f"Expected options to be a dict, got {type(row['options']).__name__}")
        groups.setdefault(str(row["discipline"]), []).append(row)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    print("Discipline counts:")
    for discipline in sorted(groups):
        count = len(groups[discipline])
        print(f"  {discipline}: {count}")
        if count < per_discipline:
            raise ValueError(f"Discipline {discipline!r} has only {count} rows, need {per_discipline}")
        sampled.extend(rng.sample(groups[discipline], per_discipline))

    expected = len(groups) * per_discipline
    if len(groups) != 9:
        raise ValueError(f"Expected 9 disciplines, found {len(groups)}: {sorted(groups)}")
    if len(sampled) != expected:
        raise AssertionError(f"Expected {expected} sampled rows, got {len(sampled)}")
    return sampled


def main() -> int:
    args = parse_args()
    download(args.url, args.cache, args.force_download)
    rows = load_jsonl(args.cache)
    print(f"Loaded {len(rows)} rows")

    sampled_rows = sample_by_discipline(rows, args.per_discipline, args.seed)
    records = [build_record(row, i) for i, row in enumerate(sampled_rows)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print("Sampled rows by discipline:")
    print(df["discipline"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
