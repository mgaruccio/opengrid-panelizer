"""Production presets, deliberate solid seam stock, and native-cavity preservation."""
from dataclasses import replace
from itertools import combinations

import pytest
from shapely import union_all
from shapely.geometry import LineString, box

from opengrid.edges import edge_reservation
from opengrid.joints import puzzle_profiles
from opengrid.layout import plan_installation
from opengrid.models import InstallationSpec, JointSpec, PrinterSpec
from opengrid.native import socket_projection
from opengrid.spec import SpecError, load_spec, validate_spec


def test_existing_puzzle_defaults_are_unchanged():
    j = JointSpec()
    assert (j.depth, j.neck_width, j.head_width, j.clearance, j.fillet_radius) == (1.5, 1.2, 2.4, 0.15, 0)
    assert JointSpec(style="none").clearance == 0.15
    assert JointSpec(style="wall").clearance == 0.05
    assert JointSpec(style="under_desk").clearance == 0.0


def test_wall_has_broad_neck_and_rounded_roots():
    j = JointSpec(style="wall")
    male, female = puzzle_profiles(j)
    assert j.neck_width >= 2 * JointSpec().neck_width
    assert male.intersection(LineString([(0, -20), (0, 20)])).length >= j.neck_width - 1e-9
    assert len(male.exterior.coords) > 20
    assert female.contains_properly(male)
    assert male.intersection(LineString([(j.depth * 0.8, -20), (j.depth * 0.8, 20)])).length > j.neck_width + 2 * j.clearance


@pytest.mark.parametrize("style", ["wall", "under_desk"])
def test_coupon_reserves_stock_without_clipping_claimed_sockets(style):
    spec = load_spec(f"examples/{'under-desk' if style == 'under_desk' else style}-fit-coupon.yaml")
    layout = plan_installation(spec)
    assert len(layout.panels) == 2
    assert len(layout.joints) == 1
    assert len({(c.ix, c.iy) for p in layout.panels for c in p.cells}) == 8
    assert not any("omitted" in w for w in layout.warnings)
    slots = union_all([socket_projection(c.x, c.y) for p in layout.panels for c in p.cells])
    assert all(not j.reservation.intersects(slots) for j in layout.joints)
    for panel in layout.panels:
        assert len(panel.cells) == 4
        for cell in panel.cells:
            assert panel.footprint.contains_properly(socket_projection(cell.x, cell.y))
    assert all(p.edges for p in layout.panels)
    allowed = union_all([*[edge_reservation(e) for e in layout.panels[0].edges],
                         *[j.reservation for j in layout.joints if style == "under_desk"]])
    assert layout.panels[0].footprint.intersection(layout.panels[1].footprint).difference(allowed).area < 1e-7


@pytest.mark.parametrize("origin", [(0, 0), (-112, -56), (0.1, -0.1)])
def test_wall_partition_has_brick_stagger_on_global_lattice(origin):
    x, y = origin
    spec = InstallationSpec(
        "brick", box(x, y, x + 336, y + 224), grid_origin=origin,
        printer=PrinterSpec(max_panel_span=112), joints=JointSpec(style="wall"),
    )
    layout = plan_installation(spec)
    # First row's full panel seam is at origin+112; next row offset is two cells (56mm).
    first = [p for p in layout.panels if p.footprint.bounds[1] < y + 1]
    second = [p for p in layout.panels if abs(p.footprint.bounds[1] - (y + 112)) < 1]
    assert any(abs(p.footprint.bounds[0] - (x + 112)) < 12 for p in first)
    assert any(abs(p.footprint.bounds[0] - (x + 56)) < 12 for p in second)
    for a, b in combinations(layout.panels, 2):
        allowed = union_all([edge_reservation(e) for e in a.edges if e in b.edges])
        assert a.footprint.intersection(b.footprint).difference(allowed).area < 1e-7
    assert union_all([p.footprint for p in layout.panels]).difference(spec.surface).area < 1e-7
    cells = [(c.ix, c.iy) for p in layout.panels for c in p.cells]
    assert len(cells) == len(set(cells))


@pytest.mark.parametrize("clearance", [-0.1, 0.11, 0.36, float("nan")])
def test_underdesk_rejects_unsupported_clearance(clearance):
    with pytest.raises(SpecError):
        validate_spec(InstallationSpec("bad", box(0, 0, 112, 56), joints=JointSpec(style="under_desk", clearance=clearance)))


def test_underdesk_rejects_ignored_dimension_overrides():
    j = JointSpec(style="under_desk")
    with pytest.raises(SpecError, match="fixed"):
        validate_spec(InstallationSpec("bad", box(0, 0, 112, 56), joints=replace(j, depth=5)))


@pytest.mark.parametrize("surface,angle", [(box(0, 0, 112, 56), 0), (box(0, 0, 56, 112), 90)])
@pytest.mark.parametrize("clearance", [0, 0.05, 0.1])
def test_underdesk_integration_for_both_seam_axes(surface, angle, clearance):
    from opengrid.cad import build_panel

    layout = plan_installation(InstallationSpec(
        "axes", surface, printer=PrinterSpec(max_panel_span=56),
        joints=JointSpec(style="under_desk", clearance=clearance),
    ))
    assert len(layout.joints) == 1 and layout.joints[0].angle == angle
    a, b = [build_panel(p) for p in layout.panels]
    assert abs(a.volume - a.cut(b).volume) < 1e-5
    for dz in (-1, 1):
        moved = a.translate((0, 0, dz))
        assert moved.volume - moved.cut(b).volume > 0.1


@pytest.mark.parametrize("style", ["wall", "under_desk"])
@pytest.mark.parametrize("clearance", [0, 0.05, 0.1])
def test_long_seams_use_every_safe_node_without_removing_cells(style, clearance):
    spec = InstallationSpec(
        "many", box(0, 0, 224, 224), printer=PrinterSpec(max_panel_span=112),
        joints=JointSpec(style=style, clearance=clearance),
    )
    layout = plan_installation(spec)
    assert sum(len(p.cells) for p in layout.panels) == 64
    assert len(layout.joints) > 8  # No two-joints-per-seam cap.
    assert not any("no safe" in w for w in layout.warnings)
    slots = union_all([socket_projection(c.x, c.y) for p in layout.panels for c in p.cells])
    assert all(not j.reservation.intersects(slots) for j in layout.joints)


def test_oversized_wall_joint_is_rejected_instead_of_removing_sockets():
    layout = plan_installation(InstallationSpec(
        "unsafe", box(0, 0, 112, 56), printer=PrinterSpec(max_panel_span=56),
        joints=JointSpec(style="wall", depth=8, neck_width=8, head_width=12, fillet_radius=1),
    ))
    assert not layout.joints
    assert sum(len(p.cells) for p in layout.panels) == 8
    assert any("no safe" in w for w in layout.warnings)


def test_final_wide_coupon_has_three_catches_and_a_four_cell_seam():
    spec = load_spec("examples/under-desk-wide-fit-coupon.yaml")
    layout = plan_installation(spec)
    assert (spec.printer.width, spec.printer.depth) == (256, 256)
    assert spec.joints.clearance == 0
    assert [p.base_footprint.bounds for p in layout.panels] == [
        (0, 0, 112, 56), (0, 56, 112, 112),
    ]
    assert all(len(p.cells) == 8 for p in layout.panels)
    seam = layout.panels[0].base_footprint.boundary.intersection(
        layout.panels[1].base_footprint.boundary
    )
    assert seam.length == pytest.approx(112)
    assert len(layout.joints) == 3
    assert all(j.angle == 90 for j in layout.joints)
    assert len(layout.panels[0].edges) == 4
    assert sum(e.length for e in layout.panels[0].edges) > 69
    assert not any("no safe" in w or "omitted" in w for w in layout.warnings)
    for panel in layout.panels:
        for cell in panel.cells:
            assert panel.footprint.contains_properly(socket_projection(cell.x, cell.y))
