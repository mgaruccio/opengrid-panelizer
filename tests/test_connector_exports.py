"""New connector coupons through the public CLI and their actual exported solids."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh
from build123d import Axis, import_step

from opengrid.cad import SOCKET_CUTTER_ASSET

ROOT = Path(__file__).resolve().parents[1]


def overlap(a, b):
    return a.volume - a.cut(b).volume


@pytest.mark.parametrize("style", ["wall", "under-desk"])
def test_public_connector_coupon_geometry(style, tmp_path):
    output = tmp_path / style
    result = subprocess.run(
        [sys.executable, "-m", "opengrid.cli", "generate", "--spec",
         str(ROOT / f"examples/{style}-fit-coupon.yaml"), "--output", str(output)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["panels"]) == 2 and len(manifest["joints"]) == 1
    assert manifest["coverage"]["functional_cells"] == 8
    assert manifest["joint_strategy"]["physical_fit"] == "requires_coupon"
    assert manifest["joint_strategy"]["clearance_definition"] == "millimetres per mating side"
    shapes = {}
    cutter = import_step(SOCKET_CUTTER_ASSET)
    for record in manifest["panels"]:
        cad = import_step(output / record["step"])
        mesh = trimesh.load_mesh(output / record["stl"])
        assert cad.is_valid and len(cad.solids()) == 1 and cad.volume > 0
        assert mesh.is_watertight and mesh.is_volume and len(mesh.split()) == 1
        assert cad.bounding_box().min.Z == pytest.approx(0, abs=1e-6)
        assert cad.bounding_box().max.Z == pytest.approx(4, abs=1e-6)
        assert mesh.bounds[0, 2] == pytest.approx(0, abs=1e-6)
        assert mesh.bounds[1, 2] == pytest.approx(4, abs=1e-6)
        placement = record["print_placement"]
        seated = cad.translate((-placement["x"], -placement["y"], 0)).rotate(Axis.Z, -placement["rotation"])
        for cell in record["cells"]:
            assert abs(overlap(seated, cutter.translate((cell["x"], cell["y"], 0)))) < 1e-5
        shapes[record["id"]] = seated
    joint = manifest["joints"][0]
    a, b = shapes[joint["male_panel"]], shapes[joint["female_panel"]]
    assert abs(overlap(a, b)) < 1e-5
    assert len(manifest["edge_interfaces"]) >= 2
    if style == "under-desk":
        assert joint["assembly_axis"] == [1, 0, 0]
        # Genuine capture in both thickness directions, not an XY-only puzzle extrusion.
        for dz in (-1.0, 1.0):
            assert overlap(a.translate((0, 0, dz)), b) > 0.1
    else:
        assert joint["assembly_axis"] == [0, 0, 1]
        assert overlap(a.translate((-0.6, 0, 0)), b) > 0.1
        assert overlap(a.translate((0, 0, 1)), b) > 0.1
        for dz in (-4, -2, -1, -0.2):
            assert abs(overlap(a.translate((0, 0, dz)), b)) < 1e-5
