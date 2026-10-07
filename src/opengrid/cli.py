"""Public deterministic compiler boundary; no research, model lookup or vision."""

import argparse
import json
import os
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .drawings import write_pdf, write_svg
from .export import export_layout, prepare_output
from .spec import SpecError, ensure_ready, load_spec


def _positive_jobs(value: str) -> int:
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("jobs must be a positive integer") from exc
    if jobs < 1:
        raise argparse.ArgumentTypeError("jobs must be a positive integer")
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile precise geometry into native openGrid Lite or Full panels"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preview", "Preview ready layouts or explicitly unresolved installation envelopes"),
        ("generate", "Export ready installation panels and assembly files"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument(
            "--spec", required=True, type=Path, help="InstallationSpec YAML or JSON"
        )
        command.add_argument(
            "--output", required=True, type=Path, help="New or empty output directory"
        )
        if name == "generate":
            command.add_argument(
                "--jobs",
                type=_positive_jobs,
                default=min(4, os.cpu_count() or 1),
                help="Parallel fresh-process panel builds and print exports (default: up to 4 CPUs; 1: in-process)",
            )
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.command == "preview":
            provisional = None
            layout = None
            try:
                ensure_ready(spec)
            except SpecError as exc:
                provisional = str(exc)
            if not provisional:
                from .layout import plan_installation

                layout = plan_installation(spec)
            prepare_output(args.output)
            write_svg(spec, args.output / "preview.svg", layout, provisional=provisional)
            write_pdf(spec, args.output / "preview.pdf", layout, provisional=provisional)
            print(
                json.dumps(
                    {
                        "installation": spec.id,
                        "provisional": bool(provisional),
                        "reason": provisional,
                        "preview": str(args.output / "preview.svg"),
                        "warnings": layout.warnings if layout else [],
                    }
                )
            )
        else:
            # Block unresolved desk inputs before planning, CAD import or filesystem writes.
            ensure_ready(spec)
            from .layout import plan_installation

            manifest = export_layout(plan_installation(spec), args.output, jobs=args.jobs)
            print(
                json.dumps(
                    {
                        "installation": spec.id,
                        "panels": len(manifest["panels"]),
                        "joints": len(manifest["joints"]),
                        "functional_cells": manifest["coverage"]["functional_cells"],
                        "output": str(args.output),
                        "warnings": manifest["warnings"],
                    }
                )
            )
        return 0
    except (SpecError, OSError, ValueError, RuntimeError, BrokenProcessPool) as exc:
        print(f"opengrid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
