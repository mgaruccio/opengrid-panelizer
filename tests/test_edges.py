"""Real native stock, assembly and layer geometry for supporting edge lips."""
from itertools import combinations

import pytest
from build123d import Compound, extrude
from shapely.geometry import LineString, box

from opengrid.cad import _footprint_face, _socket_cutter, build_panel
from opengrid.edges import edge_reservation, lip_sections, lip_solids
from opengrid.layout import plan_installation
from opengrid.models import InstallationSpec, JointSpec, PrinterSpec


def overlap(a, b):
    return a.volume - a.cut(b).volume


@pytest.mark.parametrize("style", ["wall", "under_desk", "under_desk_puzzle"])
@pytest.mark.parametrize("clearance", [0, 0.05, 0.1])
def test_lip_preserves_native_edge_web_and_supported_layer_growth(style, clearance):
    tongue, cutter = lip_sections(style, clearance)
    native = _socket_cutter()
    last_tongue = None
    last_stock = None
    for i in range(80):
        z = i * 0.05 + 0.001
        slab = extrude(_footprint_face(box(-20, -20, 20, 20)), amount=0.001, dir=(0, 0, 1)).translate((0, 0, z))
        cut = native.intersect(slab)
        if isinstance(cut, list):
            cut = Compound(cut)
        web = 14 + cut.bounding_box().min.X
        band = box(-1, z, 2, z + 0.001)
        lip_cut = cutter.intersection(band)
        reach = 0 if lip_cut.is_empty else lip_cut.bounds[2]
        assert web - reach >= 0.799, (style, clearance, z, web - reach)
        # No lip/receiver growth faster than one horizontal mm per vertical mm.
        tongue_line = tongue.intersection(LineString([(-1, z), (1, z)]))
        tongue_reach = 0 if tongue_line.is_empty else max(0, tongue_line.bounds[2])
        stock_reach = reach
        if last_tongue is not None:
            assert tongue_reach - last_tongue <= 0.0501
            assert last_stock - stock_reach <= 0.0501
        last_tongue, last_stock = tongue_reach, stock_reach


@pytest.mark.parametrize("style", ["wall", "under_desk"])
@pytest.mark.parametrize("horizontal", [False, True])
def test_actual_production_lips_and_native_cavities(style, horizontal):
    x, y = -112.1, -56.1
    w, h = (56, 112) if horizontal else (112, 56)
    span, cells = 56, 8
    if style == "wall" and horizontal:
        # Brick staggering makes a56mm-wide second row two half-panels.
        # Use a full-width production brick layout to test a genuine Y seam.
        w, h, span, cells = 112, 224, 112, 32
    layout = plan_installation(InstallationSpec(
        "edges", box(x, y, x + w, y + h), grid_origin=(x, y),
        printer=PrinterSpec(max_panel_span=span), joints=JointSpec(style=style),
    ))
    assert sum(len(p.cells) for p in layout.panels) == cells
    node = next(j for j in layout.joints if j.angle == (90 if horizontal else 0))
    panels = {p.id: p for p in layout.panels}
    edges = tuple(e for e in panels[node.male_panel].edges if e.female_panel == node.female_panel)
    assert len(edges) >= 2
    assert sum(e.length for e in edges) > 35
    shapes = {p.id: build_panel(p) for p in layout.panels}
    a, b = shapes[node.male_panel], shapes[node.female_panel]
    assert abs(overlap(a, b)) < 1e-5
    for edge in edges:
        assert edge in panels[node.female_panel].edges
        assert all(not edge_reservation(edge).intersects(j.reservation) for j in layout.joints)
        add, cut = lip_solids(edge)
        # Test the actual production lip away from the spring, not a replacement shape.
        if style == "under_desk":
            for dz in (-0.3, 0.3):
                assert overlap(add.translate((0, 0, dz)), b) > 0.01
        else:
            assert overlap(add.translate((0, 0, 0.3)), b) > 0.01
            assert abs(overlap(add.translate((0, 0, -0.3)), b)) < 1e-5
        assert abs(overlap(a, cut)) > 0  # Pocket corresponds to the real tongue.
    if style == "wall":
        for dz in (-4, -2, -1, -0.2):
            assert abs(overlap(a.translate((0, 0, dz)), b)) < 1e-5


def test_brick_wall_and_aligned_desk_lips_do_not_collide_at_corners():
    for style in ("wall", "under_desk"):
        layout = plan_installation(InstallationSpec(
            "multi", box(0, 0, 224, 224), printer=PrinterSpec(max_panel_span=112),
            joints=JointSpec(style=style),
        ))
        assert sum(len(p.cells) for p in layout.panels) == 64
        assert all(p.edges for p in layout.panels)
        shapes = [build_panel(p) for p in layout.panels]
        for a, b in combinations(shapes, 2):
            assert abs(overlap(a, b)) < 1e-5
