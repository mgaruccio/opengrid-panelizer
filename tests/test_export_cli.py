"""Exercise the actual public compiler and reload its files (not mocked CAD)."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import trimesh
from build123d import Axis, Plane, import_step
from shapely import union_all
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, shape

from opengrid.cad import MOUNTING_CUTTER_ASSET

ROOT = Path(__file__).resolve().parents[1]


def projected_mesh(mesh):
    triangles = mesh.triangles[:, :, :2]
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    doubled_area = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return union_all([Polygon(t) for t, area in zip(triangles, doubled_area) if area > 1e-9])


def test_public_generate_exports_irregular_native_panels(tmp_path):
    output = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opengrid.cli",
            "generate",
            "--spec",
            str(ROOT / "examples/irregular.yaml"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["panels"] >= 2 and summary["joints"] >= 1
    assert manifest["native_interface"]["thickness_mm"] == 4
    assert manifest["native_interface"]["mounting_hole"]["head_face_z_mm"] == 4
    assert manifest["native_interface"]["license"] == "CC-BY-4.0"
    assert "David D" in (output / "ATTRIBUTION.txt").read_text()
    ET.parse(output / "assembly.svg")
    assert "head face Z=4 (away from print bed)" in (output / "assembly.svg").read_text()
    assert (output / "assembly.pdf").read_bytes().startswith(b"%PDF")
    surface = shape(manifest["installation"]["surface"])
    keepouts = union_all(
        [
            shape(k["geometry"]).buffer(k["clearance"] + k["uncertainty"])
            for k in manifest["installation"]["keepouts"]
        ]
    )
    usable = surface.difference(keepouts)
    bed = shape(manifest["printer"]["usable_bed"])
    seen_cells = set()
    for panel in manifest["panels"]:
        mesh = trimesh.load_mesh(output / panel["stl"])
        assert mesh.is_watertight and mesh.is_volume
        assert mesh.body_count == 1
        assert mesh.bounds[0, 2] == pytest.approx(0, abs=1e-5)
        assert mesh.bounds[1, 2] == pytest.approx(4, abs=1e-5)
        assert mesh.volume == pytest.approx(panel["volume_mm3"], rel=0.005)
        cad = import_step(output / panel["step"])
        assert cad.is_valid and len(cad.solids()) == 1
        assert cad.volume == pytest.approx(panel["volume_mm3"], rel=1e-7)
        mount = import_step(MOUNTING_CUTTER_ASSET).mirror(Plane.XY).translate((0, 0, 4))
        p = panel["print_placement"]
        assert panel["mounting_holes"], "fixture must exercise real mounting holes"
        for hole in panel["mounting_holes"]:
            cut = mount.translate((hole["x"], hole["y"], 0))
            cut = cut.rotate(Axis.Z, p["rotation"]).translate((p["x"], p["y"], 0))
            assert cad.cut(cut).volume == pytest.approx(cad.volume, abs=1e-5)
        expected_volume = (
            shape(panel["footprint"]).area * 4 - len(panel["cells"]) * 2424.898419520364
        )
        expected_volume -= len(panel["mounting_holes"]) * mount.volume
        # Allow CAD/planar kernel rounding, far below the ~100 mm³ per native hole.
        assert cad.volume == pytest.approx(expected_volume, rel=1e-6, abs=1e-5)
        actual = projected_mesh(mesh)
        assert actual.difference(bed.buffer(0.0002)).area < 0.001
        p = panel["print_placement"]
        original = rotate(
            translate(actual, xoff=-p["x"], yoff=-p["y"]), -p["rotation"], origin=(0, 0)
        )
        assert original.difference(usable.buffer(0.0002)).area < 0.001
        for cell in panel["cells"]:
            key = (cell["ix"], cell["iy"])
            assert key not in seen_cells
            seen_cells.add(key)
            assert cell["x"] == pytest.approx((cell["ix"] + 0.5) * 28)
            assert cell["y"] == pytest.approx((cell["iy"] + 0.5) * 28)
    assembly = import_step(output / "assembly.step")
    assert assembly.is_valid and len(assembly.solids()) == len(manifest["panels"])
    assert len(seen_cells) == manifest["coverage"]["functional_cells"]
    circles = (
        ET.parse(output / "assembly.svg").getroot().findall(".//{http://www.w3.org/2000/svg}circle")
    )
    assert len([c for c in circles if c.attrib.get("class") == "mounting-hole"]) == sum(
        len(p["mounting_holes"]) for p in manifest["panels"]
    )
