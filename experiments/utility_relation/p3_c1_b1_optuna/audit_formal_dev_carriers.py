"""Read-only inventory audit for all expected formal development carriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna.formal_carrier_materializer import (
    FORMAL_OUTPUT_SUBDIR, carrier_directory, expected_identities, verify_complete_carrier,
)


def audit_inventory(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    records = []
    for dataset, seed in expected_identities():
        path = carrier_directory(output_root, dataset, seed)
        marker_exists = (path / "COMPLETE.json").is_file()
        record = {
            "dataset": dataset, "seed": seed, "path": str(path),
            "complete_marker_exists": marker_exists,
            "hashes_pass": False, "audit_pass": False,
            "full_gt_loaded": None, "status": "MISSING",
        }
        if path.exists():
            if not marker_exists:
                record["status"] = "INVALID_MISSING_COMPLETE"
            else:
                try:
                    verified = verify_complete_carrier(path, dataset, seed)
                    record.update({
                        "hashes_pass": True,
                        "audit_pass": True,
                        "full_gt_loaded": verified["audit"]["full_gt_loaded"],
                        "status": "COMPLETE_VERIFIED",
                    })
                except (ValueError, OSError, json.JSONDecodeError) as error:
                    record["status"] = f"INVALID:{type(error).__name__}:{error}"
        records.append(record)

    complete = sum(record["status"] == "COMPLETE_VERIFIED" for record in records)
    missing = sum(record["status"] == "MISSING" for record in records)
    invalid = len(records) - complete - missing
    return {
        "expected_carrier_count": len(records),
        "complete_verified_count": complete,
        "missing_count": missing,
        "invalid_count": invalid,
        "ALL_16_FORMAL_DEV_CARRIERS_READY": complete == len(records) and invalid == 0,
        "carriers": records,
        "full_gt_loaded": False,
        "training_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inventory = audit_inventory(args.output_root)
    if args.output.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"inventory": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
