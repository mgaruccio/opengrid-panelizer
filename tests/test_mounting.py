"""Native Lite mounting holes, not a screw/wood attachment design."""

from pathlib import Path

import pytest
from build123d import Align, Box, Plane, import_step
from shapely.geometry import Point, box

from opengrid.cad import MOUNTING_CUTTER_ASSET, SOCKET_CUTTER_ASSET, _mounting_cutter, build_panel
from opengrid.layout import plan_installation
from opengrid.models import Cell, Panel, PrintPlacement
from opengrid.native import mounting_holes, socket_projection
from opengrid.spec import load_spec

ASSETS = Path(__file__).parents[1] / "src/opengrid/assets/native"
ROOT = Path(__file__).parents[1]


def test_generated_mounting_recess_faces_away_from_print_bed():
    raw = import_step(MOUNTING_CUTTER_ASSET)
    expected = raw.mirror(Plane.XY).translate((0, 0, 4))
    cutter = _mounting_cutter()
    assert cutter.cut(expected).volume < 1e-5
    assert expected.cut(cutter).volume < 1e-5
    for z, radius in ((0, 2.05), (4, 3.6)):
        assert any(
            abs(e.arc_center.Z - z) < 1e-6 and abs(e.radius - radius) < 1e-6
            for e in cutter.edges()
            if e.geom_type.name == "CIRCLE"
        )


def grid_panel(size=4, origin=(0, 0)):
    ox, oy = origin
    return Panel(
        "test",
        box(ox, oy, ox + size * 28, oy + size * 28),
        tuple(
            Cell(ix, iy, ox + (ix + 0.5) * 28, oy + (iy + 0.5) * 28)
            for ix in range(size)
            for iy in range(size)
        ),
        PrintPlacement(0, 0, 0),
    )


def test_cutter_preserves_native_through_hole_and_recess():
    source = import_step(ASSETS / "openGrid_Lite_4x4.step")
    tile = Box(8, 8, 4, align=(Align.CENTER, Align.CENTER, Align.MIN)).translate((1162, -1722, 0))
    actual = max(tile.cut(source).solids(), key=lambda s: s.volume).translate((-1162, 1722, 0))
    cutter = import_step(MOUNTING_CUTTER_ASSET)
    assert actual.cut(cutter).volume < 1e-5
    assert cutter.cut(actual).volume < 1e-5
    assert cutter.volume == pytest.approx(99.6950394695547, rel=1e-8)
    # Native head recess is on Z=0, not an invented reversed countersink.
    for z, radius in ((0, 3.6), (4, 2.05)):
        assert any(
            abs(e.arc_center.Z - z) < 1e-6 and abs(e.radius - radius) < 1e-6
            for e in cutter.edges()
            if e.geom_type.name == "CIRCLE"
        )


@pytest.mark.parametrize("origin", [(0, 0), (-20.25, 100.5)])
def test_native_56mm_pattern_tracks_socket_phase(origin):
    p = grid_panel(origin=origin)
    expected = {(origin[0] + x, origin[1] + y) for x in (28, 84) for y in (28, 84)}
    assert set(mounting_holes(p)) == expected
    for x, y in mounting_holes(p):
        head = Point(x, y).buffer(3.6)
        assert p.footprint.covers(head)
        assert all(head.disjoint(socket_projection(c.x, c.y)) for c in p.cells)


def test_no_partial_edge_or_missing_cell_mounting_holes():
    p = grid_panel(2)
    assert mounting_holes(p) == ((28, 28),)
    p.footprint = p.footprint.difference(box(26, 26, 30, 30))
    assert mounting_holes(p) == ()
    p = grid_panel(2)
    p.cells = p.cells[:-1]
    assert mounting_holes(p) == ()
    assert mounting_holes(grid_panel(1)) == ()


def test_cad_removes_complete_native_holes_without_affecting_sockets():
    p = grid_panel(2)
    cad = build_panel(p)
    socket = import_step(SOCKET_CUTTER_ASSET)
    mount = import_step(MOUNTING_CUTTER_ASSET).mirror(Plane.XY).translate((0, 0, 4))
    assert cad.is_valid and len(cad.solids()) == 1
    assert cad.volume == pytest.approx(56 * 56 * 4 - 4 * socket.volume - mount.volume, abs=1e-5)
    assert cad.cut(mount.translate((28, 28, 0))).volume == pytest.approx(cad.volume, abs=1e-5)
    # Mirroring the mounting cutter must not mirror the accessory sockets.
    for cell in p.cells:
        cavity = socket.translate((cell.x, cell.y, 0))
        assert cad.cut(cavity).volume == pytest.approx(cad.volume, abs=1e-5)
    bounds = cad.bounding_box()
    assert bounds.min.Z == pytest.approx(0, abs=1e-6)
    assert bounds.max.Z == pytest.approx(4, abs=1e-6)


def test_coupon_keeps_a_complete_mount_in_each_panel_clear_of_joint():
    layout = plan_installation(load_spec(ROOT / "examples/fit-coupon.yaml"))
    assert len(layout.panels) == 2 and layout.joints
    for panel in layout.panels:
        holes = mounting_holes(panel)
        assert len(holes) == 1
        for x, y in holes:
            head = Point(x, y).buffer(3.6)
            assert all(head.disjoint(j.male) and head.disjoint(j.female) for j in layout.joints)
