"""Public deterministic compiler boundary; no research, model lookup or vision."""

import argparse
import json
import sys
from pathlib import Path

from .drawings import write_pdf, write_svg
from .export import export_layout, prepare_output
from .spec import SpecError, ensure_ready, load_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile precise geometry into openGrid Lite panels"
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

            manifest = export_layout(plan_installation(spec), args.output)
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
    except (SpecError, OSError, ValueError, RuntimeError) as exc:
        print(f"opengrid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
