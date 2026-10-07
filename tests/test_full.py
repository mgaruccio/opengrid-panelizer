"""Native Full family through specs, Python API, and the real CLI's exported solids."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from itertools import combinations

import pytest
import trimesh
import yaml
from build123d import Align, Axis, Box, Compound, Location, extrude, import_step
from shapely.geometry import MultiPoint, box, shape
from test_puzzle_support import assert_lowering_clear, common_volume

from opengrid import generate_installation
from opengrid.cad import (
    FULL_SOCKET_CUTTER_ASSET,
    _footprint_face,
    _mounting_cutter,
    _socket_cutter,
    build_panel,
    printable_panel,
)
from opengrid.edges import lip_solids
from opengrid.export import _puzzle_support
from opengrid.layout import plan_installation, transformed_print_footprint
from opengrid.models import THICKNESS, JointSpec, PrintPlacement, board_thickness
from opengrid.native import mounting_holes, socket_projection
from opengrid.spec import SpecError, from_dict, load_spec

ASSETS = FULL_SOCKET_CUTTER_ASSET.parent


def document(style="none", width=56, depth=56, span=112):
    return {"schema_version": 1, "installation": {
        "id": "synthetic-full", "status": "ready", "units": "mm", "board": "full",
        "surface": {"type": "rectangle", "width": width, "depth": depth},
        "printer": {"width": 128, "depth": 128, "max_panel_span": span, "exclusions": []},
        "joints": {"style": style},
    }}


@pytest.mark.parametrize("board", ["unknown", "Full", "", None, True, [], {}])
def test_unknown_family_is_rejected_at_spec_and_python_boundaries(board):
    raw = document()
    raw["installation"]["board"] = board
    with pytest.raises(SpecError, match="board must be lite or full"):
        from_dict(raw)
    with pytest.raises(SpecError, match="board must be lite or full"):
        generate_installation(box(0, 0, 56, 56), board=board)


def test_lite_default_and_full_python_selection():
    raw = document()
    del raw["installation"]["board"]
    implicit = from_dict(raw)
    raw["installation"]["board"] = "lite"
    assert implicit == from_dict(raw)
    assert THICKNESS == board_thickness("lite") == 4
    assert board_thickness("full") == 6.8
    default = generate_installation(box(0, 0, 56, 56))
    full = generate_installation(box(0, 0, 56, 56), board="full")
    assert default.spec.board == "lite" and all(p.board == "lite" for p in default.panels)
    assert full.spec.board == "full" and all(p.board == "full" for p in full.panels)
    assert sum(len(p.cells) for p in default.panels) == sum(len(p.cells) for p in full.panels)
    assert any(mounting_holes(p) for p in default.panels)
    assert all(not mounting_holes(p) for p in full.panels)
    assert _socket_cutter().volume == pytest.approx(_socket_cutter("lite").volume)


def test_full_spring_clip_rejected_before_any_cli_output(tmp_path):
    raw = document("under_desk")
    with pytest.raises(SpecError, match="Lite-only under_desk"):
        from_dict(raw)
    with pytest.raises(SpecError, match="Lite-only under_desk"):
        generate_installation(box(0, 0, 112, 56), board="full",
                              joint_strategy=JointSpec(style="under_desk"))
    spec = tmp_path / "unsupported.json"
    spec.write_text(json.dumps(raw))
    output = tmp_path / "blocked"
    process = subprocess.run([sys.executable, "-m", "opengrid.cli", "generate", "--spec", str(spec),
                              "--output", str(output)], capture_output=True, text=True,
                             timeout=20, check=False)
    assert process.returncode == 2 and "Lite-only under_desk" in process.stderr
    assert not output.exists()


@pytest.mark.parametrize("size,centre", [(2, (1400, -1596)), (4, (1176, -1372))])
def test_full_native_cutter_matches_official_sources_and_shared_projection(size, centre):
    reference = import_step(ASSETS / f"openGrid_Full_{size}x{size}.step")
    tile = Box(28, 28, 6.8, align=(Align.MIN, Align.MIN, Align.MIN)).translate(
        (centre[0] - 14, centre[1] - 14, 0))
    source = max(tile.cut(reference).solids(), key=lambda s: s.volume).translate(
        (-centre[0], -centre[1], 0))
    cutter = _socket_cutter("full")
    assert cutter.is_valid and len(cutter.solids()) == 1
    assert cutter.volume == pytest.approx(4108.003995374757, abs=1e-6)
    assert source.cut(cutter).volume < 1e-5
    assert cutter.cut(source).volume < 1e-5
    bounds = cutter.bounding_box()
    assert tuple(bounds.min) == pytest.approx((-13.2, -13.2, 0), abs=1e-6)
    assert tuple(bounds.max) == pytest.approx((13.2, 13.2, 6.8), abs=1e-6)
    hull = MultiPoint([(v.center().X, v.center().Y) for v in cutter.vertices()]).convex_hull
    lite_hull = MultiPoint([(v.center().X, v.center().Y)
                           for v in _socket_cutter().vertices()]).convex_hull
    assert len(hull.exterior.coords) == 17
    assert hull.hausdorff_distance(socket_projection(0, 0)) < 1e-7
    assert hull.hausdorff_distance(lite_hull) < 1e-7


@pytest.mark.parametrize("clockwise", [False, True])
def test_full_slab_has_exact_cavities_no_direct_holes_and_retains_print_pose(clockwise):
    panel = plan_installation(from_dict(document())).panels[0]
    if clockwise:
        panel.footprint = panel.footprint.reverse()
    cad = build_panel(panel)
    assert cad.is_valid and len(cad.solids()) == 1
    assert cad.volume == pytest.approx(56 * 56 * 6.8 - 4 * _socket_cutter("full").volume, abs=1e-5)
    assert mounting_holes(panel) == ()
    assert mounting_holes(replace(panel, board="lite")) == ((28, 28),)
    assert cad.is_inside((28, 28, 3.4))  # Native Full node is solid, not a Lite screw hole.
    printed = printable_panel(replace(panel, placement=PrintPlacement(90, 100, 200)))
    assert tuple(printed.bounding_box().min) == pytest.approx((44, 200, 0), abs=1e-6)
    assert tuple(printed.bounding_box().max) == pytest.approx((100, 256, 6.8), abs=1e-6)


def test_full_support_never_approves_indirect_only_mounting():
    layout = plan_installation(from_dict(document("under_desk_puzzle", 336, 336)))
    support = _puzzle_support(layout)
    assert all(p["fastening"] == "native_tiles_required" for p in support["panels"])
    assert len(support["mounting_blockers"]) == len(layout.panels)
    assert set(support["assembly_steps"][-1]["panels"]) == {p.id for p in layout.panels}
    assert "indirect-only support is not approved" in support["limitation"]
    assert support["accessory_face_z_mm"] == 6.8
    # The same interior bricks remain Lite candidates; Full must not inherit their approval.
    lite = plan_installation(replace(layout.spec, board="lite"))
    assert any(p["fastening"] == "indirect_support_candidate"
               for p in _puzzle_support(lite)["panels"])


CASES = {
    "single": ("none", 56, 56, 112, 1, 4),
    "puzzle": ("puzzle", 112, 56, 56, 2, 8),
    "wall": ("wall", 112, 56, 56, 2, 8),
    "support": ("under_desk_puzzle", 112, 336, 112, 4, 48),
}


@pytest.fixture(scope="module", params=CASES)
def full_output(request, tmp_path_factory):
    case = request.param
    style, width, depth, span, _, _ = CASES[case]
    root = tmp_path_factory.mktemp(f"full-{case}")
    raw = document(style, width, depth, span)
    spec = root / ("input.json" if case == "support" else "input.yaml")
    spec.write_text(json.dumps(raw) if spec.suffix == ".json" else yaml.safe_dump(raw))
    output = root / "output"
    command = [sys.executable, "-m", "opengrid.cli", "generate", "--spec", str(spec), "--output", str(output)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    (root / "cli.log").write_text(" ".join(command) + "\n" + process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    return case, output, load_spec(spec)


def test_public_full_cli_geometry_metadata_and_mounting(full_output):
    case, output, spec = full_output
    manifest = json.loads((output / "manifest.json").read_text())
    _, _, _, _, panel_count, cell_count = CASES[case]
    assert len(manifest["panels"]) == panel_count
    assert manifest["coverage"]["functional_cells"] == cell_count
    assert manifest["installation"]["board"] == manifest["native_interface"]["board"] == "full"
    assert manifest["native_interface"]["thickness_mm"] == 6.8
    assert "mounting_hole" not in manifest["native_interface"]
    assert "native Full mounting tiles/snaps required on every panel" in manifest["mounting"]
    attribution = (output / "ATTRIBUTION.txt").read_text()
    assert "Native openGrid Full accessory geometry" in attribution
    assert "David D" in attribution and "CC BY 4.0" in attribution
    assert "Z=4" not in attribution and "Lite" not in attribution
    svg = ET.parse(output / "assembly.svg")
    assert not svg.findall(".//{http://www.w3.org/2000/svg}circle[@class='mounting-hole']")
    assert (output / "assembly.pdf").read_bytes().startswith(b"%PDF")
    assembly = import_step(output / "assembly.step")
    assert assembly.is_valid and len(assembly.solids()) == panel_count
    layout = plan_installation(spec)
    planned = {p.id: p for p in layout.panels}
    shapes, cells = {}, set()
    for record in manifest["panels"]:
        panel = planned[record["id"]]
        assert panel.board == "full" and record["mounting_holes"] == []
        printed = import_step(output / record["step"])
        assert printed.is_valid and len(printed.solids()) == 1
        assert printed.bounding_box().min.Z == pytest.approx(0, abs=1e-6)
        assert printed.bounding_box().max.Z == pytest.approx(6.8, abs=1e-6)
        assert printed.volume == pytest.approx(record["volume_mm3"], abs=1e-5)
        mesh = trimesh.load_mesh(output / record["stl"])
        assert mesh.is_watertight and mesh.is_volume and len(mesh.split()) == 1
        assert mesh.bounds[:, 2] == pytest.approx([0, 6.8], abs=1e-5)
        assert mesh.volume == pytest.approx(printed.volume, rel=0.005)
        footprint = transformed_print_footprint(panel.footprint, panel.placement)
        assert shape(manifest["printer"]["usable_bed"]).covers(footprint)
        assert tuple(printed.bounding_box().min)[:2] == pytest.approx(footprint.bounds[:2], abs=1e-6)
        assert tuple(printed.bounding_box().max)[:2] == pytest.approx(footprint.bounds[2:], abs=1e-6)
        place = record["print_placement"]
        local = printed.translate((-place["x"], -place["y"], 0)).rotate(Axis.Z, -place["rotation"])
        shapes[panel.id] = local
        assert any(abs(common_volume(local, s) - local.volume) < 1e-5 for s in assembly.solids())
        cutters = []
        for cell in record["cells"]:
            key = (cell["ix"], cell["iy"])
            assert key not in cells
            cells.add(key)
            assert cell["x"] == pytest.approx((cell["ix"] + 0.5) * 28)
            assert cell["y"] == pytest.approx((cell["iy"] + 0.5) * 28)
            assert panel.footprint.contains_properly(socket_projection(cell["x"], cell["y"]))
            cutters.append(_socket_cutter("full").moved(Location((cell["x"], cell["y"], 0))))
        assert abs(common_volume(local, Compound(children=cutters))) < 1e-5
        for x, y in mounting_holes(replace(panel, board="lite")):
            plug = _mounting_cutter().translate((x, y, 0))
            assert common_volume(local, plug) == pytest.approx(plug.volume, abs=1e-5)
    assert len(cells) == cell_count
    for a, b in combinations(shapes.values(), 2):
        assert abs(common_volume(a, b)) < 1e-5
    if case != "single":
        assert manifest["joints"]
        for joint in layout.joints:
            key = extrude(_footprint_face(joint.male), amount=6.8, dir=(0, 0, 1))
            pocket = extrude(_footprint_face(joint.female), amount=6.8, dir=(0, 0, 1))
            assert common_volume(key, shapes[joint.male_panel]) == pytest.approx(key.volume, abs=1e-5)
            assert abs(common_volume(pocket, shapes[joint.female_panel])) < 1e-5
    if case in {"wall", "support"}:
        assert manifest["edge_interfaces"]
        for edge in {e.id: e for p in layout.panels for e in p.edges}.values():
            tongue, _ = lip_solids(edge)
            assert common_volume(tongue.translate((0, 0, 0.3)), shapes[edge.female_panel]) > 0.001
    print(f"{case}: {panel_count} single-solid Full panels, {cell_count} intact sockets, "
          f"watertight STLs, assembly/placement/mounting/drawings verified: {output}")


def test_public_full_rigid_insertion_and_support_instructions(full_output):
    case, output, _ = full_output
    if case == "single":
        return
    manifest = json.loads((output / "manifest.json").read_text())
    shapes = {}
    for record in manifest["panels"]:
        p = record["print_placement"]
        shapes[record["id"]] = import_step(output / record["step"]).translate(
            (-p["x"], -p["y"], 0)).rotate(Axis.Z, -p["rotation"])
    if case != "support":
        assert_lowering_clear(shapes["P002"], shapes["P001"], travel=6.9)
        if case == "wall":
            with pytest.raises(AssertionError):
                assert_lowering_clear(shapes["P001"], shapes["P002"], travel=6.9)
        return
    support = manifest["puzzle_support"]
    assert support["mounting_face_z_mm"] == 0 and support["accessory_face_z_mm"] == 6.8
    assert support["tongue_z_mm"] == [0.1, 1.9]
    assert all(p["fastening"] == "native_tiles_required" for p in support["panels"])
    assert len(support["mounting_blockers"]) == len(shapes)
    text = (output / "ASSEMBLY.md").read_text()
    assert "Z=6.8" in text and "Z=4" not in text
    assert "Use their existing native mounting holes" not in text
    assert "indirect-only support is not approved" in text
    placed = []
    for step in support["assembly_steps"]:
        assert f"{step['number']}. {step['instruction']}" in text
        if step["action"] == "build_row":
            for index, panel in enumerate(step["panels"]):
                for previous in step["panels"][:index]:
                    assert_lowering_clear(shapes[panel], shapes[previous], travel=6.9)
        if step["action"] == "position_infill":
            placed.extend(step["panels"])
        if step["action"] == "lower_anchor_row":
            for panel in step["panels"]:
                for previous in placed:
                    assert_lowering_clear(shapes[panel], shapes[previous], travel=6.9)
            placed.extend(step["panels"])
    assert set(placed) == set(shapes)
    assert set(support["assembly_steps"][-1]["panels"]) == set(shapes)
    # Full-thickness negative control: a tongue cannot be lowered over a seated receiver.
    edge = manifest["edge_interfaces"][0]
    with pytest.raises(AssertionError):
        assert_lowering_clear(shapes[edge["male_panel"]], shapes[edge["female_panel"]], travel=6.9)
    print(f"{case}: complete rigid row/anchor insertion from Z=6.9 to 0 verified: {output}")
