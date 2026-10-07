import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from opengrid.cli import main

ROOT = Path(__file__).resolve().parents[1]
DESK = ROOT / "examples/draft-installation.yaml"


def test_draft_preview_public_boundary(tmp_path):
    output = tmp_path / "preview"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opengrid.cli",
            "preview",
            "--spec",
            str(DESK),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["provisional"] is True
    tree = ET.parse(output / "preview.svg")
    text = " ".join(tree.getroot().itertext())
    assert "PROVISIONAL" in text and "NO PRINTABLE LAYOUT" in text
    assert (output / "preview.pdf").read_bytes().startswith(b"%PDF")
    assert not list(output.glob("*.stl")) and not list(output.glob("*.step"))


def test_unresolved_draft_rejects_print_export_before_writing(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opengrid.cli",
            "generate",
            "--spec",
            str(DESK),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Printable export blocked" in result.stderr
    assert "Installed frame" in result.stderr
    assert not output.exists()


def test_output_is_not_silently_overwritten(tmp_path, capsys):
    output = tmp_path / "existing"
    output.mkdir()
    existing = output / "important.txt"
    existing.write_text("keep this")
    assert main(["preview", "--spec", str(DESK), "--output", str(output)]) == 2
    assert existing.read_text() == "keep this"
    assert "empty directory" in capsys.readouterr().err


def test_xml_content_is_escaped(tmp_path):
    from shapely.geometry import box

    from opengrid.drawings import write_svg
    from opengrid.models import InstallationSpec

    spec = InstallationSpec("test<script>alert(1)</script>", box(0, 0, 56, 56), status="draft")
    f = tmp_path / "escaped.svg"
    write_svg(spec, f, provisional="unknown & unmeasured")
    root = ET.parse(f).getroot()
    assert not root.findall("{http://www.w3.org/2000/svg}script")


def test_ready_approval_preview_discloses_layout_warnings(tmp_path, capsys):
    output = tmp_path / "warning-preview"
    assert (
        main(["preview", "--spec", str(ROOT / "examples/irregular.yaml"), "--output", str(output)])
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["warnings"] and summary["provisional"] is False
    text = " ".join(ET.parse(output / "preview.svg").getroot().itertext())
    assert "WARNING:" in text and "conservative" in text and "0.8 mm" in text
    assert "assembly.step uses installation coordinates" in text
