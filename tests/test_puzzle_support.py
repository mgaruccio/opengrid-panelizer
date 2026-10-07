"""Opt-in puzzle support rows, including the public CLI and full rigid approach."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest
import yaml
from shapely.geometry import box

from opengrid.edges import lip_sections
from opengrid.export import _joint_assembly_axis, _puzzle_support
from opengrid.joints import puzzle_profiles
from opengrid.layout import plan_installation
from opengrid.models import InstallationSpec, JointSpec, Keepout, PrinterSpec
from opengrid.native import mounting_holes
from opengrid.spec import SpecError, load_spec, validate_spec


def rows(**kwargs):
    spec = InstallationSpec(
        "support", box(0, 0, 336, 336), printer=PrinterSpec(max_panel_span=112),
        joints=JointSpec(style="under_desk_puzzle"),
    )
    return plan_installation(replace(spec, **kwargs))


def candidates(record):
    return {p["panel"] for p in record["panels"] if p["fastening"] == "indirect_support_candidate"}


def test_wall_dimensions_and_mounting_side_are_reused_without_z_mirroring():
    wall, desk = JointSpec(style="wall"), JointSpec(style="under_desk_puzzle")
    assert replace(desk, style="wall") == wall
    assert all(a.equals(b) for a, b in zip(puzzle_profiles(wall), puzzle_profiles(desk)))
    for clearance in (0, 0.05, 0.1):
        a, b = lip_sections("wall", clearance), lip_sections("under_desk_puzzle", clearance)
        assert all(x.equals(y) for x, y in zip(a, b))
        assert b[0].bounds[1::2] == (0.1, 1.9)  # X/Z section: mounting-side tongue.
    with pytest.raises(SpecError, match="0.1"):
        validate_spec(InstallationSpec("bad", box(0, 0, 112, 56),
                      joints=replace(desk, clearance=0.11)))


def test_joint_male_axis_follows_lip_ownership_not_puzzle_polarity():
    layout = rows()
    observed = set()
    for joint in layout.joints:
        pair = {joint.male_panel, joint.female_panel}
        lips = [e for p in layout.panels for e in p.edges
                if {e.male_panel, e.female_panel} == pair]
        if lips:
            expected = 1 if lips[0].male_panel == joint.male_panel else -1
            assert _joint_assembly_axis(joint, layout) == [0, 0, expected]
            observed.add(expected)
    assert observed == {-1, 1}


@pytest.mark.parametrize("clearance", [0, 0.05, 0.1])
def test_actual_lips_define_dag_and_only_two_interior_candidates(clearance):
    layout = rows(joints=JointSpec(style="under_desk_puzzle", clearance=clearance))
    record = _puzzle_support(layout)
    assert len(layout.panels) == 10
    assert sum(len(p.cells) for p in layout.panels) == 144
    assert candidates(record) == {"P005", "P006"}
    assert not record["mounting_blockers"]
    panels = {p.id: p for p in layout.panels}
    edges = {e.id: e for p in layout.panels for e in p.edges}
    for edge in edges.values():
        male, female = panels[edge.male_panel], panels[edge.female_panel]
        if male.grid_row == female.grid_row:
            assert edge.angle == 0
        else:
            assert male.grid_row % 2 == 1 and female.grid_row % 2 == 0
            assert edge.angle == (90 if female.grid_row > male.grid_row else 270)
    assert any(e.angle == 270 for e in edges.values())
    for panel in record["panels"]:
        if panel["panel"] in candidates(record):
            for side in ("north", "south"):
                assert panel["supporting_lips"][side]
                for support in panel["supporting_lips"][side]:
                    assert mounting_holes(panels[support["panel"]])
                    assert edges[support["lip"]].male_panel == panel["panel"]
    steps = record["assembly_steps"]
    assert [s["grid_row"] for s in steps if s["action"] == "build_row"] == [1, 0, 2]
    assert steps[-1]["action"] == "mount"
    assert set(steps[-1]["panels"]) == set(panels) - candidates(record)


@pytest.mark.parametrize("side,angle", [("north", 90), ("south", 270)])
def test_missing_either_opposed_lip_side_fails_closed(side, angle):
    layout = rows()
    omitted = {e.id for p in layout.panels for e in p.edges
               if e.male_panel == "P005" and e.angle == angle}
    assert omitted
    for panel in layout.panels:
        panel.edges = tuple(e for e in panel.edges if e.id not in omitted)
    record = _puzzle_support(layout)
    assert "P005" not in candidates(record)
    assert next(p for p in record["panels"] if p["panel"] == "P005")["supporting_lips"][side] == []
    assert "P006" in candidates(record)


def test_anchor_without_native_holes_cannot_support_candidate(monkeypatch):
    layout = rows()
    north = {e.female_panel for p in layout.panels for e in p.edges
             if e.male_panel == "P005" and e.angle == 90}
    monkeypatch.setattr("opengrid.export.mounting_holes", lambda p: () if p.id in north else mounting_holes(p))
    record = _puzzle_support(layout)
    assert "P005" not in candidates(record)
    assert all(any(p in b for b in record["mounting_blockers"]) for p in north)


def test_negative_global_phase_and_irregular_fragmentation():
    origin = (-0.1, 0.1)
    layout = rows(surface=box(origin[0] - 336, origin[1] - 224, origin[0], origin[1] + 112),
                  grid_origin=origin)
    record = _puzzle_support(layout)
    assert {p.grid_row for p in layout.panels} == {-2, -1, 0}
    assert len(candidates(record)) == 2
    assert all(p["grid_row"] == -1 for p in record["panels"] if p["panel"] in candidates(record))
    irregular = rows(keepouts=(Keepout("notch", box(112, 112, 140, 140)),))
    for p in _puzzle_support(irregular)["panels"]:
        if p["fastening"] == "indirect_support_candidate":
            actual = next(panel for panel in irregular.panels if panel.id == p["panel"])
            assert actual.base_footprint.distance(irregular.usable.boundary) > 1e-7
            assert len(actual.cells) == 16
    fragmented = rows(printer=PrinterSpec(width=80, depth=140, exclusions=(), max_panel_span=112))
    assert not candidates(_puzzle_support(fragmented))


def test_real_single_cell_fragments_report_native_mounting_blockers():
    layout = rows(surface=box(0, 0, 84, 84), printer=PrinterSpec(max_panel_span=28))
    record = _puzzle_support(layout)
    assert len(layout.panels) == 9
    assert not candidates(record)
    assert all(not mounting_holes(p) for p in layout.panels)
    assert len(record["mounting_blockers"]) == len(layout.panels)


def test_actual_edge_contradiction_is_rejected():
    layout = rows()
    edge = layout.panels[0].edges[0]
    wrong = replace(edge, male_panel=edge.female_panel, female_panel=edge.male_panel,
                    angle=(edge.angle + 180) % 360)
    for panel in layout.panels:
        panel.edges = tuple(wrong if e.id == edge.id else e for e in panel.edges)
    with pytest.raises(SpecError, match="contradicts"):
        _puzzle_support(layout)


def common_volume(a, b):
    # Use OCCT directly: build123d Compound.intersect can lose imported STEP locations.
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    aa, bb = a.bounding_box(), b.bounding_box()
    if (aa.max.X < bb.min.X - 1e-7 or bb.max.X < aa.min.X - 1e-7
            or aa.max.Y < bb.min.Y - 1e-7 or bb.max.Y < aa.min.Y - 1e-7
            or aa.max.Z < bb.min.Z - 1e-7 or bb.max.Z < aa.min.Z - 1e-7):
        return 0
    operation = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
    assert operation.IsDone()
    result = operation.Shape()
    if result.IsNull():
        return 0
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(result, properties)
    return properties.Mass()


def assert_lowering_clear(moving, stationary):
    """Exact continuous +4.1 -> 0 Z sweep, not sampled poses or clipped solids.

    Sweeping every boundary face plus the initial solid covers the full rigid
    swept body. Bounding boxes only reject disjoint faces; nothing is clipped.
    Planar faces parallel to travel sweep zero volume and may be skipped.
    """
    from build123d import Compound, GeomType
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    assert abs(common_volume(moving, stationary)) < 1e-5
    target = stationary.bounding_box()
    for face in moving.faces():
        bounds = face.bounding_box()
        if (bounds.max.X < target.min.X - 1e-6 or bounds.min.X > target.max.X + 1e-6
                or bounds.max.Y < target.min.Y - 1e-6 or bounds.min.Y > target.max.Y + 1e-6
                or bounds.min.Z >= target.max.Z - 1e-7):
            continue
        if face.geom_type == GeomType.PLANE and abs(face.normal_at().Z) < 1e-8:
            continue
        prism = BRepPrimAPI_MakePrism(face.wrapped, gp_Vec(0, 0, 4.1))
        assert prism.IsDone() and not prism.Shape().IsNull()
        swept = Compound(prism.Shape())
        assert swept.is_valid
        assert abs(common_volume(swept, stationary)) < 1e-5


@pytest.fixture(scope="module", params=["rows", "t-coupon", "irregular"])
def public_output(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"puzzle-{request.param}")
    document = yaml.safe_load(Path("examples/under-desk-puzzle.yaml").read_text())
    if request.param == "t-coupon":
        document["installation"]["surface"].update(width=112, depth=168)
    if request.param == "irregular":
        document["installation"]["surface"].update(width=168, depth=336)
        document["installation"]["keepouts"] = [{"id": "notch", "geometry": {
            "type": "rectangle", "x": 84, "y": 112, "width": 28, "depth": 28,
        }}]
    spec = root / "input.yaml"
    spec.write_text(yaml.safe_dump(document))
    output = root / "output"
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    process = subprocess.run([sys.executable, "-m", "opengrid.cli", "generate", "--spec", str(spec),
                              "--output", str(output)], capture_output=True, text=True, env=env,
                              timeout=90, check=False)
    (root / "cli.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    return request.param, output, load_spec(spec)


@pytest.mark.parametrize("phase", ["native", "within_rows", "anchor_0", "anchor_2"])
def test_public_cli_native_solids_and_complete_assembly_sweep(public_output, phase):
    import trimesh
    from build123d import Axis, Compound, Location, import_step

    from opengrid.cad import _mounting_cutter, _socket_cutter

    case, output, spec = public_output
    manifest = json.loads((output / "manifest.json").read_text())
    support = manifest["puzzle_support"]
    for joint in manifest["joints"]:
        pair = {joint["male_panel"], joint["female_panel"]}
        lips = [e for e in manifest["edge_interfaces"]
                if {e["male_panel"], e["female_panel"]} == pair]
        if lips:
            expected = 1 if lips[0]["male_panel"] == joint["male_panel"] else -1
            assert joint["assembly_axis"] == [0, 0, expected]
    shapes = {}
    for panel in manifest["panels"]:
        printed = import_step(output / panel["step"])
        assert printed.is_valid and len(printed.solids()) == 1
        bounds = printed.bounding_box()
        assert bounds.min.Z == pytest.approx(0, abs=1e-6)
        assert bounds.max.Z == pytest.approx(4, abs=1e-6)
        placement = panel["print_placement"]
        shape = printed.translate((-placement["x"], -placement["y"], 0)).rotate(Axis.Z, -placement["rotation"])
        shapes[panel["id"]] = shape
        if phase == "native":
            mesh = trimesh.load_mesh(output / panel["stl"])
            assert mesh.is_watertight and mesh.is_volume and len(mesh.split()) == 1
            cutters = [_socket_cutter().moved(Location((c["x"], c["y"], 0))) for c in panel["cells"]]
            cutters += [_mounting_cutter().moved(Location((h["x"], h["y"], 0)))
                        for h in panel["mounting_holes"]]
            assert abs(common_volume(shape, Compound(children=cutters))) < 1e-5
            print(f"{case}: {panel['id']} native cutters / connected STEP / watertight STL clear", flush=True)
    if case == "rows":
        assert len(shapes) == 10 and candidates(support) == {"P005", "P006"}
    if case == "t-coupon":
        assert len(shapes) == 3
    if phase != "native":
        placed = []
        checked = False
        for step in support["assembly_steps"]:
            if step["action"] == "build_row" and phase == "within_rows":
                for index, panel in enumerate(step["panels"]):
                    for previous in step["panels"][:index]:
                        assert_lowering_clear(shapes[panel], shapes[previous])
                checked = True
            if step["action"] == "position_infill":
                placed.extend(step["panels"])
            if step["action"] == "lower_anchor_row":
                if phase == f"anchor_{step['grid_row']}":
                    # The whole row translates together. Check every moving/fixed pair,
                    # not just lip neighbours: T-junctions and third-party keys stay present.
                    for panel in step["panels"]:
                        for previous in placed:
                            assert_lowering_clear(shapes[panel], shapes[previous])
                    checked = True
                placed.extend(step["panels"])
        assert set(placed) == set(shapes)
        if not checked:
            pytest.skip("This coupon has no such anchor row")
        if case == "rows" and phase == "within_rows":
            # Negative control: trying to lower an infill tongue over a seated
            # anchor MUST fail. This also checks imported placements in the sweep.
            with pytest.raises(AssertionError):
                assert_lowering_clear(shapes["P005"], shapes["P001"])
    else:
        for a, b in combinations(shapes.values(), 2):
            assert abs(common_volume(a, b)) < 1e-5
        # Each actual tongue bears in +Z against the unmodified receiver.
        layout = plan_installation(spec)
        for panel in layout.panels:
            for edge in panel.edges:
                if edge.male_panel == panel.id and edge.angle in (90, 270):
                    from opengrid.edges import lip_solids
                    tongue, _ = lip_solids(edge)
                    assert common_volume(tongue.translate((0, 0, 0.3)), shapes[edge.female_panel]) > 0.001
    text = (output / "ASSEMBLY.md").read_text()
    assert all(f"{s['number']}. {s['instruction']}" in text for s in support["assembly_steps"])
    print(f"{case}: {len(shapes)} panels; {phase} checks clear: {output}")


def test_public_cli_draft_never_produces_printable_output(tmp_path):
    document = yaml.safe_load(Path("examples/under-desk-puzzle.yaml").read_text())
    document["installation"]["status"] = "draft"
    spec = tmp_path / "draft.yaml"
    spec.write_text(yaml.safe_dump(document))
    output = tmp_path / "blocked"
    result = subprocess.run([sys.executable, "-m", "opengrid.cli", "generate", "--spec", str(spec),
                             "--output", str(output)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
                            timeout=20, check=False)
    assert result.returncode != 0
    assert not output.exists() or not list(output.iterdir())
