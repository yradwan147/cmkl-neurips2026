"""Merge per-seed result JSON files into a single combined file.

After running 1 seed per SLURM job, each seed produces a separate file like:
    naive_sequential_TransE_seed42.json
    naive_sequential_TransE_seed123.json

This script merges them into a single file:
    naive_sequential_TransE.json

Usage:
    python scripts/merge_seed_results.py --input-dir results
    python scripts/merge_seed_results.py --input-dir results --method naive_sequential
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def find_seed_files(input_dir: Path, method: str | None = None) -> dict[str, list[Path]]:
    """Group per-seed files by their base name (without _seedXXX suffix).

    Returns:
        Dict mapping base_name -> list of seed files.
    """
    seed_pattern = re.compile(r"^(.+)_seed(\d+)\.json$")
    groups: dict[str, list[Path]] = {}

    for f in sorted(input_dir.glob("*_seed*.json")):
        m = seed_pattern.match(f.name)
        if not m:
            continue
        base = m.group(1)
        if method and method not in base:
            continue
        groups.setdefault(base, []).append(f)

    return groups


def merge_group(files: list[Path]) -> dict:
    """Merge per-seed JSON files into a combined result.

    Each file has a 'results' list with 1 entry. We combine them into
    a single 'results' list with all seeds.
    """
    all_results = []
    base_data = None

    for f in sorted(files):
        with open(f) as fh:
            data = json.load(fh)
        if base_data is None:
            base_data = {k: v for k, v in data.items() if k not in ("results", "seeds")}
        results_list = data.get("results", [])
        all_results.extend(results_list)

    if base_data is None:
        return {}

    seeds = sorted(set(
        r.get("seed", r.get("cl_metrics", {}).get("seed"))
        for r in all_results
        if r.get("seed") is not None or r.get("cl_metrics", {}).get("seed") is not None
    ))

    base_data["seeds"] = seeds
    base_data["results"] = all_results
    return base_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-seed results")
    parser.add_argument("--input-dir", default="results")
    parser.add_argument("--method", default=None,
                        help="Only merge files matching this method name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be merged without writing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    groups = find_seed_files(input_dir, args.method)

    if not groups:
        print(f"No per-seed files found in {input_dir}")
        return

    for base, files in sorted(groups.items()):
        output_path = input_dir / f"{base}.json"
        seeds = [f.stem.split("_seed")[-1] for f in files]
        print(f"  {base}: {len(files)} seed files ({', '.join(seeds)}) -> {output_path.name}")

        if args.dry_run:
            continue

        merged = merge_group(files)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"    Written: {output_path}")

    print(f"\nMerged {len(groups)} result groups")


if __name__ == "__main__":
    main()
