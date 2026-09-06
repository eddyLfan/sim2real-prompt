"""Thin command-line wrapper around the public pipeline interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .api import PromptAnnotationPipeline


def _pipeline(args: argparse.Namespace) -> PromptAnnotationPipeline:
    return PromptAnnotationPipeline(
        args.config,
        dataset_root=getattr(args, "dataset_root", None),
        output_root=getattr(args, "output_root", None),
    )


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_glob": args.dataset_glob,
        "episodes": args.episodes,
        "limit": args.limit,
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def command_inspect(args: argparse.Namespace) -> int:
    _print_json(_pipeline(args).inspect(**_selection(args), show=args.show))
    return 0


def command_run(args: argparse.Namespace) -> int:
    try:
        result = _pipeline(args).run(
            **_selection(args),
            force=args.force,
            dry_run=args.dry_run,
            prepare_media=args.prepare_media,
            show=args.show,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    _print_json(result)
    return 1 if result.get("failed", 0) else 0


def command_references(args: argparse.Namespace) -> int:
    result = _pipeline(args).export_references(
        **_selection(args),
        directory_name=args.directory_name,
        overwrite=args.overwrite,
        full_resolution=not args.resized,
        jpeg_quality=args.jpeg_quality,
    )
    _print_json(result)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    result = _pipeline(args).audit(**_selection(args), show=args.show)
    _print_json(result)
    return 1 if result["incomplete"] else 0


def command_render(args: argparse.Namespace) -> int:
    text = _pipeline(args).render(args.annotation)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def command_schema(args: argparse.Namespace) -> int:
    del args
    _print_json(PromptAnnotationPipeline.schemas())
    return 0


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Optional path to YAML configuration")
    parser.add_argument("--dataset-root", help="Override the configured dataset root")
    parser.add_argument("--output-root", help="Override the configured output root")


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-glob",
        default=os.environ.get("SIM2REAL_PROMPT_DATASET_GLOBS", "*"),
        help="Dataset directory glob or comma-separated globs",
    )
    parser.add_argument("--episodes", help="Episode ids/ranges, e.g. 0,2,5-9")
    parser.add_argument("--limit", type=int, help="Global sample limit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim2real-prompt",
        description="Project-specific compact prompts for paired LeRobot datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect bounded paired LeRobot metadata discovery"
    )
    _add_config(inspect_parser)
    _add_selection(inspect_parser)
    inspect_parser.add_argument("--show", type=int, default=3)
    inspect_parser.set_defaults(handler=command_inspect)

    run_parser = subparsers.add_parser("run", help="Run the linear batch pipeline")
    _add_config(run_parser)
    _add_selection(run_parser)
    run_parser.add_argument(
        "--force", action="store_true", help="Regenerate completed samples"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Do not create a VLM client"
    )
    run_parser.add_argument(
        "--prepare-media",
        action="store_true",
        help="In dry-run mode, decode selected media for every matched sample",
    )
    run_parser.add_argument("--show", type=int, default=3)
    run_parser.set_defaults(handler=command_run)

    reference_parser = subparsers.add_parser(
        "references", help="Export deterministic same-episode Reference JPEGs"
    )
    _add_config(reference_parser)
    _add_selection(reference_parser)
    reference_parser.add_argument("--directory-name", default="Reference")
    reference_parser.add_argument("--jpeg-quality", type=int, default=95)
    reference_parser.add_argument(
        "--overwrite", action="store_true", help="Replace conflicting images"
    )
    reference_parser.add_argument(
        "--resized",
        action="store_true",
        help="Export the resized prompt representation instead of full resolution",
    )
    reference_parser.set_defaults(handler=command_references)

    audit_parser = subparsers.add_parser(
        "audit", help="Audit current outputs without API access"
    )
    _add_config(audit_parser)
    _add_selection(audit_parser)
    audit_parser.add_argument("--show", type=int, default=10)
    audit_parser.set_defaults(handler=command_audit)

    render_parser = subparsers.add_parser(
        "render", help="Deterministically render an existing canonical annotation"
    )
    _add_config(render_parser)
    render_parser.add_argument("--annotation", required=True)
    render_parser.add_argument("--output")
    render_parser.set_defaults(handler=command_render)

    schema_parser = subparsers.add_parser(
        "schema", help="Print the annotation JSON Schema"
    )
    schema_parser.set_defaults(handler=command_schema)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
