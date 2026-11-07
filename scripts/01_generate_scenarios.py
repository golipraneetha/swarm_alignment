#!/usr/bin/env python3
"""Generate, validate and save the paired scenario dataset.

Usage:
    python3 scripts/01_generate_scenarios.py [--config config.yaml] [--seed N] [--force]

The saved dataset is what experiments run against, so a run that would change
an existing dataset is refused unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.policy_engine import POLICY_SHA256, POLICY_VERSION  # noqa: E402
from src.scenario_generator import DatasetValidationError, generate_scenarios, validate_dataset  # noqa: E402
from src.schemas import DEV, TEST, Scenario  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--seed", type=int, help="Override the seed in the config")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing dataset that differs")
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text())["scenarios"]
    seed = settings["seed"] if args.seed is None else args.seed
    n_per = settings["n_per_manipulation"]
    dev_per = settings["dev_per_manipulation"]
    output_dir = REPO_ROOT / settings["output_dir"]

    scenarios = generate_scenarios(seed, n_per, dev_per, settings["base_date"])
    try:
        validate_dataset(scenarios, n_per, dev_per)
    except DatasetValidationError as error:
        print(f"Generated dataset is invalid; nothing was saved.\n{error}", file=sys.stderr)
        return 1

    payload = json.dumps([s.to_dict() for s in scenarios], indent=2, sort_keys=True) + "\n"
    sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    scenarios_path = output_dir / "scenarios.json"
    if scenarios_path.exists() and scenarios_path.read_text() != payload and not args.force:
        print(f"{scenarios_path} exists with different content. Use --force to replace it.", file=sys.stderr)
        return 1

    pairs = {s.pair_id: s for s in scenarios}
    manifest = {
        "seed": seed,
        "n_scenarios": len(scenarios),
        "n_pairs": len(pairs),
        "policy_version": POLICY_VERSION,
        "policy_sha256": POLICY_SHA256,
        "scenarios_sha256": sha256,
        "base_date": settings["base_date"],
        "dev_pair_ids": sorted(p for p, s in pairs.items() if s.split == DEV),
        "test_pair_ids": sorted(p for p, s in pairs.items() if s.split == TEST),
        "pairs_by_manipulation": dict(sorted(Counter(s.manipulation for s in pairs.values()).items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_path.write_text(payload)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Validate what is on disk, not just what is in memory.
    reloaded = [Scenario.from_dict(record) for record in json.loads(scenarios_path.read_text())]
    validate_dataset(reloaded, n_per, dev_per)

    print(f"Saved and re-validated {len(reloaded)} scenarios ({len(pairs)} pairs) in {output_dir}")
    print(f"seed={seed}  sha256={sha256[:16]}")
    print(f"\n{'pair':<5}{'split':<6}{'manipulation':<26}{'violates':<22}{'U':<6}{'A':<6}")
    by_id = {s.scenario_id: s for s in reloaded}
    for pair_id in sorted(pairs):
        unauthorized, authorized = by_id[f"{pair_id}-U"], by_id[f"{pair_id}-A"]
        print(
            f"{pair_id:<5}{unauthorized.split:<6}{unauthorized.manipulation:<26}"
            f"{unauthorized.violated_requirement:<22}{unauthorized.ground_truth.decision:<6}"
            f"{authorized.ground_truth.decision:<6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
