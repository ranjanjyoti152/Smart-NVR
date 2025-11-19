#!/usr/bin/env python3
"""Utility script to run face auto-assimilation outside the web API."""

import argparse
import json
from typing import Any

from app import app
from app.models.face_profile import FaceProfile


def _serialize_result(result: Any) -> Any:
    if isinstance(result, (list, dict)):
        return result
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge unlabeled faces into known identities")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold required to merge profiles (0.0 - 1.0)",
    )
    args = parser.parse_args()

    threshold = max(0.0, min(1.0, args.threshold))

    with app.app_context():
        merges = FaceProfile.auto_assimilate_unlabeled(threshold=threshold)

    payload = {
        "threshold": threshold,
        "count": len(merges),
        "merged": [_serialize_result(item) for item in merges],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
