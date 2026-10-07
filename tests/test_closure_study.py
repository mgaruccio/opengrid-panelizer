"""Bounded experimental geometry: real native CAD, exported meshes, and 2-D assembly."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh
from build123d import import_step
from shapely.affinity import translate
from shapely.geometry import box

from opengrid.joints import puzzle_profiles
from opengrid.native import socket_projection

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "closure_study", ROOT / "experiments" / "closure_study.py"
)
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)

def test_native_land_and_unadorned_puzzle_primitives():
    male, female = puzzle_profiles(study.PUZZLE)
    assert study.PUZZLE.depth == 2.2
    assert study.PUZZLE.neck_width == 3.4
    assert study.PUZZLE.head_width == 4.6
    assert study.PUZZLE.clearance == 0.05
    assert study.PUZZLE.fillet_radius == 0.4
    assert male.difference(female).area == pytest.approx(0)
    land = study.collar_outline()
    bay = box(-2.25, -2.65, 2.25, 0.15)
    channel = study.bowtie(travel=study.KEY_TRAVEL, offset=study.POCKET_ALLOWANCE)
    assert land.covers(bay) and land.covers(channel)
    for x in (-14, 14):
        for y in (-14, 14):
            socket = socket_projection(x, y)
            for footprint in (male, female, land, bay, channel):
                assert socket.intersection(footprint).area < 1e-9


@pytest.mark.parametrize("clearance", study.KEY_CLEARANCES)
def test_rigid_key_continuous_sweep(clearance):
    key = study.key_solid(clearance)
    sweep = study.key_solid(clearance, travel=study.KEY_TRAVEL)
    cutter = study.key_cutter()
    for shape in (key, sweep, cutter):
        assert shape.is_valid and len(shape.solids()) == 1
    # The complete translational swept solid fits the pocket, not just endpoints.
    assert sweep.volume - study.overlap(sweep, cutter) < study.TOL
    for offset in (0, -0.31, -1.25, -1.89, -2.5):
        posed = key.translate((0, offset, 0))
        assert posed.volume - study.overlap(posed, sweep) < study.TOL
    # A top-insertion certificate includes every section of the rigid key.
    bounds = key.bounding_box()
    drop = study.prism(box(bounds.min.X, bounds.min.Y - 2.5,
                           bounds.max.X, bounds.max.Y - 2.5), study.KEY_BOTTOM, 7.5)
    assert drop.volume - study.overlap(drop, cutter) < study.TOL
    assert key.bounding_box().min.Z == pytest.approx(1.25, abs=1e-6)
    assert key.bounding_box().max.Z == pytest.approx(7, abs=1e-6)


def test_invalid_key_clearance_is_rejected():
    with pytest.raises(ValueError):
        study.key_solid(0)
    with pytest.raises(ValueError):
        study.key_solid(travel=-1)


@pytest.mark.parametrize(("concept", "clearance"),
                         [("A", 0.05), ("B", 0.05), ("C", 0.05), ("C", 0.12), ("C", 0.20)])
def test_actual_exported_two_by_two_assembly(concept, clearance, tmp_path):
    samples, sites = study.build_grid(concept, 2, 2)
    assert len(samples) == 4 and len(sites) == 4
    assert {s.angle for s in sites} == {0, 90}
    assert all((s.x, s.y) != (0, 0) for s in sites)
    shapes = {}
    for i, sample in enumerate(samples):
        _, seated, _, mesh = study.export_part(tmp_path, sample.panel.id, sample.shape,
                                               4 + i * 62, 4)
        assert mesh["watertight"] and mesh["components"] == 1
        assert seated.bounding_box().max.Z == pytest.approx(study.HEIGHTS[concept], abs=1e-6)
        native = study.verify_native(sample, seated)
        assert native["socket_count"] == 4
        assert native["mount_count"] == 1
        shapes[sample.panel.id] = seated
    keys = {}
    if concept == "C":
        for i, site in enumerate(sites):
            _, key, _, _ = study.export_part(
                tmp_path, f"key-{i}", study.pose(study.key_solid(clearance), site),
                4 + i * 14, 100)
            keys[site.id] = key
    result = study.verify_assembly(samples, sites, shapes, keys, key_clearance=clearance)
    assert len(result["panel_insertion"]) == 4
    assert result["seated_max_interference_mm3"] < study.TOL
    assert result["positive_z_capture"] is (concept == "C")
    if concept == "C":
        assert len(result["key_insertion"]) == 4
        assert all(min(k["capture"].values()) > 0.01 for k in result["key_insertion"])
    else:
        # This study deliberately removes the production wall preset's edge lips.
        # The full-depth puzzle can lift out and does not pretend to capture Z.
        for a in shapes.values():
            for b in shapes.values():
                if a is not b:
                    assert study.overlap(a.translate((0, 0, 1)), b) < study.TOL


def test_public_cli_plate_native_exports_and_fea(tmp_path):
    result = subprocess.run(
        [sys.executable, "experiments/closure_study.py", "--output", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=110, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    checks = json.loads((tmp_path / "geometry-verification.json").read_text())
    panels = [p for p in manifest["parts"] if p["kind"] == "panel"]
    assert len(panels) == 8
    assert sum(len(p["cells"]) for p in panels) == 48
    assert len({p["stl"] for p in manifest["parts"]}) == 28
    assert len({p["step"] for p in manifest["parts"]}) == 28
    for part in manifest["parts"]:
        assert (tmp_path / part["stl"]).is_file()
        assert (tmp_path / part["step"]).is_file()
    for concept, expected_solids in (("A", 2), ("B", 2), ("C", 8)):
        assembly = import_step(tmp_path / f"assembly-{concept}.step")
        assert assembly.is_valid and len(assembly.solids()) == expected_solids
        assert checks["two_by_two"][concept]["panel_count"] == 4
        assert checks["two_by_two"][concept]["joint_count"] == 4
        assert checks["assemblies"][concept]["positive_z_capture"] is (concept == "C")
        left, right = [import_step(tmp_path / "fea" / concept / f"{role}.step")
                       for role in ("left", "right")]
        assert len(left.solids()) == len(right.solids()) == 1
        assert study.overlap(left, right) < study.TOL
        assert left.bounding_box().min.X == pytest.approx(-5, abs=1e-6)
        assert right.bounding_box().max.X == pytest.approx(5, abs=1e-6)
    fea_key = import_step(tmp_path / "fea/C/key.step")
    for role in ("left", "right"):
        panel = import_step(tmp_path / f"fea/C/{role}.step")
        assert study.overlap(fea_key, panel) < study.TOL
        for dz in (-0.5, 0.5):
            assert study.overlap(fea_key.translate((0, 0, dz)), panel) > 0.01
    for name in (f"key-spare-{c:.2f}-{i}" for c in (0.12, 0.20) for i in range(1, 5)):
        assert (tmp_path / f"parts/{name}.step").is_file()
        assert checks["spare_keys"][name]["slide_interference_mm3"] < study.TOL
    assert manifest["parameters"]["key_clearance_variants_mm"] == [0.05, 0.12, 0.20]
    for clearance in (0.05, 0.12, 0.20):
        assert sum(p.get("xy_clearance_per_side_mm") == clearance for p in manifest["parts"]) == 4
    # Measured against actual STEP surfaces: the inner sloping roof slightly
    # reduces X clearance (and nominal-key downward play), unlike a box estimate.
    for clearance, z_play, x_play in (("0.05", 0.1392, 0.044),
                                    ("0.12", 0.2200, 0.184),
                                    ("0.20", 0.3000, 0.344)):
        play = checks["key_play"][clearance]
        assert play["pure_z_total_mm"] == pytest.approx(z_play, abs=2e-4)
        assert play["across_x_separation_mm"] == pytest.approx(x_play, abs=2e-4)
        assert play["z_minus_mm"] > 0 and play["z_plus_mm"] > 0
    assert "squared" in manifest["concepts"]["C"]["name"]
    assert any("mounting back stays flat" in note for note in manifest["limitations"])
    mesh = trimesh.load_mesh(tmp_path / "plate.stl")
    assert mesh.is_watertight and mesh.is_volume and len(mesh.split()) == 28
    assert min(mesh.bounds[0, :2]) >= 0 and max(mesh.bounds[1, :2]) <= 256
    assert mesh.bounds[1, 2] == pytest.approx(8, abs=1e-6)
    assert all(checks["parts"][p["id"]]["watertight"] for p in panels)


def test_translated_swept_profile_is_not_just_endpoint_union():
    flange = study.bowtie()
    endpoints = translate(flange, yoff=-1.25).union(translate(flange, yoff=1.25))
    assert study.bowtie(travel=2.5).area > endpoints.area


def test_squared_shoulders_have_x_normals_and_printable_land():
    from shapely.geometry import LineString

    pocket = study.bowtie(travel=2.5, offset=study.POCKET_ALLOWANCE)
    # Below the roof, actual retained seam stock is 0.95 mm, not the old
    # 0.15 mm that simply squaring a 0.35 mm waist would have produced.
    land = box(0, -4, 4, 4).difference(pocket)
    assert land.intersection(LineString([(0, 2.3), (2, 2.3)])).length == pytest.approx(0.95)
    for clearance in study.KEY_CLEARANCES:
        offset = study.POCKET_ALLOWANCE - clearance
        stem = study.bowtie(1.2, offset=offset)
        neck = stem.intersection(LineString([(0, -2), (0, 2)])).length
        assert neck >= 3 * 0.45
        key = study.key_solid(clearance)
        shoulder_x = study.KEY_WAIST_X - offset
        faces = [f for f in key.faces()
                 if abs(abs(f.center().X) - shoulder_x) < 1e-6
                 and f.center().Y > 2.15 and f.center().Z < study.KEY_FLANGE_TOP]
        assert len(faces) == 2
        for face in faces:
            normal = face.normal_at()
            assert abs(normal.X) == pytest.approx(1)
            assert normal.Y == pytest.approx(0, abs=1e-7)
            assert normal.Z == pytest.approx(0, abs=1e-7)
            assert face.area >= 1.0  # Broad flat bearing, no rounding-induced Y cam.
