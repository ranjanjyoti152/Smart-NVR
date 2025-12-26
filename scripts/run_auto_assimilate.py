#!/usr/bin/env python3
"""Utility script to run face auto-assimilation with enhanced accuracy options."""

import argparse
import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
from app.models.face_profile import FaceProfile


def _serialize_result(result: Any) -> Any:
    if isinstance(result, (list, dict)):
        return result
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge unlabeled faces into known identities with enhanced accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings (high accuracy)
  python run_auto_assimilate.py
  
  # Lower threshold for more matches, but require more samples
  python run_auto_assimilate.py --threshold 0.85 --min-samples 3
  
  # Aggressive mode (lower threshold, no multi-sample validation)
  python run_auto_assimilate.py --threshold 0.8 --no-multi-sample
  
  # Conservative mode (high threshold, high samples)
  python run_auto_assimilate.py --threshold 0.95 --min-samples 5
"""
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold required to merge profiles (0.0-1.0, default: 0.9)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples required in unlabeled profile (default: 1)",
    )
    parser.add_argument(
        "--no-multi-sample",
        action="store_true",
        help="Disable multi-sample agreement validation (faster but less accurate)",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.6,
        help="Minimum agreement ratio for multi-sample validation (0.0-1.0, default: 0.6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually merging",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including skipped profiles",
    )
    args = parser.parse_args()

    threshold = max(0.0, min(1.0, args.threshold))
    min_samples = max(1, args.min_samples)
    min_agreement = max(0.0, min(1.0, args.min_agreement))
    require_multi_sample = not args.no_multi_sample

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    with app.app_context():
        if args.dry_run:
            # For dry run, we need to peek at what would match without merging
            # This is a simplified version - just show the count of candidates
            from app import db
            known_count = db.face_profiles.count_documents({"status": "known", "name": {"$ne": None}})
            unlabeled_count = db.face_profiles.count_documents({"status": "unlabeled"})
            
            print(json.dumps({
                "dry_run": True,
                "settings": {
                    "threshold": threshold,
                    "min_samples": min_samples,
                    "require_multi_sample_agreement": require_multi_sample,
                    "min_agreement_ratio": min_agreement,
                },
                "known_profiles": known_count,
                "unlabeled_candidates": unlabeled_count,
                "message": "Run without --dry-run to execute merges"
            }, indent=2))
            return

        merges = FaceProfile.auto_assimilate_unlabeled(
            threshold=threshold,
            min_samples=min_samples,
            require_multi_sample_agreement=require_multi_sample,
            min_agreement_ratio=min_agreement,
        )

    payload = {
        "settings": {
            "threshold": threshold,
            "min_samples": min_samples,
            "require_multi_sample_agreement": require_multi_sample,
            "min_agreement_ratio": min_agreement,
        },
        "count": len(merges),
        "merged": [_serialize_result(item) for item in merges],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
