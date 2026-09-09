"""Command-line interface for the two-branch preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .api import Sim2RealPreprocessingPipeline


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _pipeline(args: argparse.Namespace) -> Sim2RealPreprocessingPipeline:
    return Sim2RealPreprocessingPipeline(
        args.config,
        dataset_root=args.dataset,
        output_root=args.output,
        dataset_glob=args.dataset_glob,
    )


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    return {"episodes": args.episodes, "limit": args.limit}


def _inspect(args: argparse.Namespace) -> int:
    _print_json(_pipeline(args).inspect(**_selection(args), show=args.show))
    return 0


def _run(args: argparse.Namespace) -> int:
    report = _pipeline(args).run(
        **_selection(args), force=args.force, audit=not args.no_audit
    )
    _print_json(report)
    return 0 if report["status"] == "complete" else 1


def _audit(args: argparse.Namespace) -> int:
    report = _pipeline(args).audit(**_selection(args), show=args.show)
    _print_json(report)
    return 0 if report["status"] == "complete" else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML configuration path")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Dataset root override (a dataset or directory of datasets)",
    )
    parser.add_argument("--output", type=Path, help="Checkpoint/report root override")
    parser.add_argument(
        "--dataset-glob",
        help="Child dataset glob override; ignored when --dataset is a dataset root",
    )
    parser.add_argument("--episodes", help="Episode selection, e.g. 0,2,5-9")
    parser.add_argument("--limit", type=int, help="Maximum number of episodes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim2real-prompt",
        description=(
            "Create Real-video prompts and YOLOE-S Seg Multi-References for Transfer"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate metadata and list work without decoding video"
    )
    _add_common(inspect_parser)
    inspect_parser.add_argument("--show", type=int, default=3)
    inspect_parser.set_defaults(handler=_inspect)

    run_parser = subparsers.add_parser("run", help="Run both branches and publish")
    _add_common(run_parser)
    run_parser.add_argument(
        "--force", action="store_true", help="Ignore valid branch checkpoints"
    )
    run_parser.add_argument(
        "--no-audit", action="store_true", help="Skip the final product audit"
    )
    run_parser.set_defaults(handler=_run)

    audit_parser = subparsers.add_parser(
        "audit", help="Check published products without API or model access"
    )
    _add_common(audit_parser)
    audit_parser.add_argument("--show", type=int, default=20)
    audit_parser.set_defaults(handler=_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
