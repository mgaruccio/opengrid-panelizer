"""Fresh-worker transport, failure cleanup, exact collision checks and public CLI parity."""

import json
import multiprocessing
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
import trimesh
from build123d import Axis, Box, Compound, Location, Part, ShapeList, import_step
from build123d.persistence import deserialize_shape
from shapely.geometry import shape

import opengrid.export as exporter
from opengrid.cad import _mounting_cutter, _socket_cutter
from opengrid.cli import main
from opengrid.layout import plan_installation
from opengrid.native import mounting_holes
from opengrid.spec import SpecError
from opengrid.spec import from_dict as parse_spec

ROOT = Path(__file__).resolve().parents[1]


def document(board="lite", width=168):
    # Three panels > two workers exercises process recycling. Fractional origins
    # also expose the precision loss from accidentally pickling CAD Locations.
    return {"schema_version": 1, "installation": {
        "id": f"parallel-{board}", "status": "ready", "units": "mm", "board": board,
        "surface": {"type": "rectangle", "x": 0.123456789, "y": 0.987654321,
                    "width": width, "depth": 56},
        "grid_origin": [0.123456789, 0.987654321],
        "printer": {"width": 128, "depth": 128, "max_panel_span": 56, "exclusions": []},
        "joints": {"style": "wall"},
    }}


def _record_build(panel):
    # Spawn imports the real worker anew; only this test wrapper records its PID.
    Path(panel._pid_path).write_text(str(os.getpid()))
    return exporter._build_panel_brep(panel)


def _crash_build(panel):
    os._exit(7)


def no_pool(*args, **kwargs):
    pytest.fail("Preflight/serial execution must not start a worker pool")


@pytest.mark.parametrize("jobs", [0, -1, 1.5, "2", True, False, None])
def test_api_rejects_invalid_jobs_before_any_output_or_pool(tmp_path, monkeypatch, jobs):
    monkeypatch.setattr(exporter, "ProcessPoolExecutor", no_pool)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="jobs must be a positive integer"):
        exporter.export_layout(None, output, jobs=jobs)
    assert not output.exists()


@pytest.mark.parametrize("jobs", ["0", "-1", "1.5", "True"])
def test_public_cli_rejects_invalid_jobs_before_loading_spec(tmp_path, jobs):
    output = tmp_path / "invalid"
    result = subprocess.run([
        sys.executable, "-m", "opengrid.cli", "generate", "--spec", "missing-spec.json",
        "--output", str(output), "--jobs", jobs,
    ], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 2
    assert "jobs must be a positive integer" in result.stderr
    assert not output.exists()


def test_preview_has_no_jobs_option(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["preview", "--spec", "missing.json", "--output", str(tmp_path), "--jobs", "2"])
    assert exc.value.code == 2


@pytest.mark.parametrize(("cpus", "expected"), [(32, 4), (2, 2), (None, 1)])
def test_cli_jobs_default_is_bounded(tmp_path, monkeypatch, cpus, expected):
    from opengrid import cli

    seen = []

    def capture(layout, output, *, jobs):
        seen.append(jobs)
        return {"panels": [], "joints": [], "coverage": {"functional_cells": 0}, "warnings": []}

    monkeypatch.setattr(cli.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(cli, "export_layout", capture)
    assert main(["generate", "--spec", str(ROOT / "examples/fit-coupon.yaml"),
                 "--output", str(tmp_path / "unused")]) == 0
    assert seen == [expected]


@pytest.mark.parametrize("reason", ["draft", "unresolved", "nonempty"])
def test_parallel_preflight_precedes_pool(tmp_path, monkeypatch, reason):
    layout = plan_installation(parse_spec(document()))
    output = tmp_path / "output"
    if reason == "draft":
        layout.spec = replace(layout.spec, status="draft")
    elif reason == "unresolved":
        layout.spec = replace(layout.spec, unresolved=("Unmeasured obstacle",))
    else:
        output.mkdir()
        (output / "keep.txt").write_text("do not touch")
    monkeypatch.setattr(exporter, "ProcessPoolExecutor", no_pool)
    with pytest.raises(SpecError):
        exporter.export_layout(layout, output, jobs=2)
    if reason == "nonempty":
        assert list(output.iterdir()) == [output / "keep.txt"]
        assert (output / "keep.txt").read_text() == "do not touch"
    else:
        assert not output.exists()


@pytest.mark.parametrize("kwargs", [{}, {"jobs": 1}])
def test_serial_api_never_starts_pool(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(exporter, "ProcessPoolExecutor", no_pool)
    layout = plan_installation(parse_spec(document(width=56)))
    result = exporter.export_layout(layout, tmp_path / "serial", **kwargs)
    assert [p["id"] for p in result["panels"]] == [p.id for p in layout.panels]


def test_native_binary_transport_preserves_nonidentity_location(tmp_path, monkeypatch):
    from opengrid import cad

    original = Part([Box(2, 3, 4)]).moved(Location(
        (12345.123456789, -12.987654321, 3.14159265359), (12.3456789, 23.4567891, 34.5678912)
    ))
    # Part ancestry and Python-only attributes must not cross the transport boundary.
    original.not_pickleable = lambda: None
    monkeypatch.setattr(cad, "build_panel", lambda panel: original)
    data = exporter._build_panel_brep(None)
    assert isinstance(data, bytes)
    restored = Part(deserialize_shape(data))
    assert not hasattr(restored, "not_pickleable")
    assert restored.is_valid and len(restored.solids()) == 1
    assert restored.volume == pytest.approx(original.volume, abs=1e-10)
    assert tuple(restored.bounding_box().min) == pytest.approx(
        tuple(original.bounding_box().min), abs=1e-10, rel=0
    )
    assert tuple(restored.bounding_box().max) == pytest.approx(
        tuple(original.bounding_box().max), abs=1e-10, rel=0
    )
    assert original.cut(restored).volume < 1e-5
    assert restored.cut(original).volume < 1e-5


def test_every_panel_gets_a_fresh_spawned_process(tmp_path, monkeypatch):
    layout = plan_installation(parse_spec(document(width=280)))
    assert len(layout.panels) == 5
    for panel in layout.panels:
        panel._pid_path = str(tmp_path / f"{panel.id}.pid")
    before = {child.pid for child in multiprocessing.active_children()}
    monkeypatch.setattr(exporter, "_build_panel_brep", _record_build)
    with exporter._panel_shapes(layout.panels, jobs=2) as shapes:
        volumes = [s.volume for s in shapes]
    assert all(v > 0 for v in volumes)
    pids = {int(Path(panel._pid_path).read_text()) for panel in layout.panels}
    assert len(pids) == 5 and os.getpid() not in pids
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_worker_error_has_panel_context_no_manifest_and_joins_workers(tmp_path):
    layout = plan_installation(parse_spec(document()))
    bad = layout.panels[1]
    bad.cells = (replace(bad.cells[0], x=10000), *bad.cells[1:])
    before = {child.pid for child in multiprocessing.active_children()}
    output = tmp_path / "failed"
    with pytest.raises(SpecError, match=f"Panel {bad.id} build failed:.*native cavity"):
        exporter.export_layout(layout, output, jobs=2)
    assert (output / f"{layout.panels[0].id}.step").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / "assembly.step").exists()
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_parent_export_error_joins_workers(tmp_path, monkeypatch):
    import build123d

    layout = plan_installation(parse_spec(document()))
    before = {child.pid for child in multiprocessing.active_children()}
    monkeypatch.setattr(build123d, "export_step", lambda *args: False)
    output = tmp_path / "failed"
    with pytest.raises(SpecError, match=f"STEP export failed for {layout.panels[0].id}"):
        exporter.export_layout(layout, output, jobs=2)
    assert not (output / "manifest.json").exists()
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_broken_pool_is_reported_at_cli_boundary_and_joined(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(document()))
    before = {child.pid for child in multiprocessing.active_children()}
    monkeypatch.setattr(exporter, "_build_panel_brep", _crash_build)
    output = tmp_path / "failed"
    assert main(["generate", "--spec", str(spec), "--output", str(output), "--jobs", "2"]) == 2
    error = capsys.readouterr().err
    assert "Panel worker pool failed while waiting for P001" in error
    assert "Traceback" not in error
    assert not (output / "manifest.json").exists()
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.parametrize(("offset", "expected"), [
    ((0.5, 0, 0), 0.5), ((1, 0, 0), 0), ((1, 1, 0), 0), ((2, 0, 0), 0), ((0, 0, 1), 0),
])
def test_exact_common_overlap_not_boundary_contact(offset, expected):
    a = Box(1, 1, 1)
    b = a.translate(offset)
    assert exporter._overlap_volume(a, b) == pytest.approx(expected, abs=1e-10)
    assert exporter._overlap_volume(a, b) == pytest.approx(a.volume - a.cut(b).volume)


def test_multiple_tiny_common_solids_are_summed():
    a = Compound([Box(0.02, 0.02, 0.02), Box(0.02, 0.02, 0.02).translate((1, 0, 0))])
    b = Box(4, 4, 4)
    common = a.intersect(b)
    assert len(common) == 2 and all(s.volume < 1e-5 for s in common)
    assert exporter._overlap_volume(a, b) == pytest.approx(1.6e-5)
    assert exporter._overlap_volume(a, b) > 1e-5
    assert exporter._overlap_volume(a, b) == pytest.approx(a.volume - a.cut(b).volume)


def test_common_return_variants_only_count_solids():
    solid = Box(1, 1, 1).solids()[0]
    face = solid.faces()[0]
    for common, volume in [
        (None, 0), (ShapeList(), 0), (Part(), 0), (face, 0), (solid, 1),
        (Compound([solid]), 1), (ShapeList([face, solid]), 1),
    ]:
        assert exporter._solid_volume(common) == pytest.approx(volume)


def test_common_preserves_assembly_child_locations():
    original = Compound(children=[Box(1, 1, 1)])
    moved = original.translate((10.123456789, 0, 0))
    assert exporter._overlap_volume(original, moved) == 0
    assert exporter._overlap_volume(moved, moved) == pytest.approx(1)


def test_colliding_panels_keep_guard_and_failure_context(tmp_path):
    layout = plan_installation(parse_spec(document(width=112)))
    layout.panels[1] = replace(layout.panels[0], id=layout.panels[1].id)
    output = tmp_path / "collision"
    with pytest.raises(SpecError, match="Connectors collide: P001/P002"):
        exporter.export_layout(layout, output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("board", ["lite", "full"])
def test_public_cli_serial_parallel_geometry_and_manifest_parity(tmp_path, board):
    raw = document(board)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(raw))
    layout = plan_installation(parse_spec(raw))
    assert len(layout.panels) == 3
    outputs = []
    for jobs in (1, 2):
        output = tmp_path / f"jobs-{jobs}"
        command = [sys.executable, "-m", "opengrid.cli", "generate", "--spec", str(spec_path),
                   "--output", str(output), "--jobs", str(jobs)]
        started = time.monotonic()
        process = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
        elapsed = time.monotonic() - started
        (tmp_path / f"jobs-{jobs}.log").write_text(
            " ".join(command) + f"\nElapsed: {elapsed:.3f}s\n" + process.stdout + process.stderr
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads(process.stdout.splitlines()[-1])["panels"] == 3
        outputs.append(output)
    serial, parallel = outputs
    manifests = [json.loads((output / "manifest.json").read_text()) for output in outputs]
    a, b = manifests
    assert sorted(p.name for p in serial.iterdir()) == sorted(p.name for p in parallel.iterdir())
    for ra, rb in zip(a["panels"], b["panels"]):
        assert ra["volume_mm3"] == pytest.approx(rb["volume_mm3"], abs=1e-7, rel=0)
        rb["volume_mm3"] = ra["volume_mm3"]
    assert a == b
    assert [r["id"] for r in b["panels"]] == [p.id for p in layout.panels]
    thickness = 4 if board == "lite" else 6.8
    assert b["native_interface"]["thickness_mm"] == thickness
    if board == "full":
        assert "mounting_hole" not in b["native_interface"]
        assert "native Full mounting tiles/snaps" in b["mounting"]
    assembly = import_step(parallel / "assembly.step")
    serial_assembly = import_step(serial / "assembly.step")
    assert assembly.is_valid and len(assembly.solids()) == 3
    assert assembly.cut(serial_assembly).volume < 1e-5
    assert serial_assembly.cut(assembly).volume < 1e-5
    ET.parse(parallel / "assembly.svg")
    assert (parallel / "assembly.pdf").read_bytes().startswith(b"%PDF")
    for panel, record in zip(layout.panels, b["panels"]):
        baseline = import_step(serial / record["step"])
        restored = import_step(parallel / record["step"])
        assert restored.is_valid and len(restored.solids()) == 1
        assert baseline.cut(restored).volume < 1e-5
        assert restored.cut(baseline).volume < 1e-5
        placement = record["print_placement"]
        local = restored.translate((-placement["x"], -placement["y"], 0)).rotate(
            Axis.Z, -placement["rotation"]
        )
        assert local.bounding_box().min.Z == pytest.approx(0, abs=1e-6)
        assert local.bounding_box().max.Z == pytest.approx(thickness, abs=1e-6)
        assert any(local.cut(s).volume < 1e-5 for s in assembly.solids())
        cutters = Compound([_socket_cutter(board).translate((c.x, c.y, 0)) for c in panel.cells])
        assert exporter._overlap_volume(local, cutters) < 1e-5
        holes = mounting_holes(replace(panel, board="lite"))
        assert holes  # Each fixture panel has a native Lite mounting location.
        for x, y in holes:
            plug = _mounting_cutter().translate((x, y, 0))
            expected = 0 if board == "lite" else plug.volume
            assert exporter._overlap_volume(local, plug) == pytest.approx(expected, abs=1e-5)
        if board == "full":
            assert record["mounting_holes"] == []
        for output in outputs:
            mesh = trimesh.load_mesh(output / record["stl"])
            assert mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume
            assert mesh.body_count == 1
            assert mesh.volume == pytest.approx(record["volume_mm3"], rel=0.005)
            assert tuple(mesh.bounds[0]) == pytest.approx(tuple(restored.bounding_box().min), abs=1e-5)
            assert tuple(mesh.bounds[1]) == pytest.approx(tuple(restored.bounding_box().max), abs=1e-5)
        assert shape(record["footprint"]).equals(panel.footprint)
    print(f"{board}: serial/parallel STEP symmetric differences <1e-5; native cavities, "
          f"mounting, print placements, assembly and single-body watertight STLs match: {tmp_path}")
