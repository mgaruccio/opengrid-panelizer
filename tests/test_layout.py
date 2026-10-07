"""Planar contract tests; the rectangular socket below is a synthetic test double.

It conservatively treats a fixture socket's entire 24 mm envelope as negative,
including its notional chamfers. It is NOT evidence of native accessory fit. The
separate native-provider test runs when the CAD implementation is integrated.
"""

import importlib.util
import math
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import ModuleType

import pytest
from shapely import union_all
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Point, Polygon, box

from opengrid.layout import (
    LayoutError,
    find_print_placement,
    plan_installation,
    transformed_print_footprint,
)
from opengrid.models import PITCH, InstallationSpec, JointSpec, Keepout, PrinterSpec, PrintPlacement
from opengrid.spec import SpecError, load_spec


def fixture_socket(x, y):
    return box(x - 12, y - 12, x + 12, y + 12)


def compile_fixture(surface, **kwargs):
    return plan_installation(
        InstallationSpec("fixture", surface, **kwargs), socket_projection=fixture_socket
    )


def assert_layout_contract(layout, projection=fixture_socket):
    seen = set()
    all_slots = []
    for panel in layout.panels:
        assert isinstance(panel.footprint, Polygon)
        assert panel.footprint.is_valid and not panel.footprint.is_empty
        assert panel.cells
        assert layout.usable.covers(panel.footprint)
        assert panel.footprint.intersection(layout.keepout_union).area == 0
        assert layout.spec.printer.usable_bed.covers(
            transformed_print_footprint(panel.footprint, panel.placement)
        )
        assert panel.placement.rotation in (0, 90, 180, 270)
        negatives = []
        for cell in panel.cells:
            assert (cell.ix, cell.iy) not in seen
            seen.add((cell.ix, cell.iy))
            ox, oy = layout.spec.grid_origin
            assert cell.x == ox + (cell.ix + 0.5) * PITCH
            assert cell.y == oy + (cell.iy + 0.5) * PITCH
            square = box(
                ox + cell.ix * 28,
                oy + cell.iy * 28,
                ox + (cell.ix + 1) * 28,
                oy + (cell.iy + 1) * 28,
            )
            assert layout.usable.covers(square)
            negative = projection(cell.x, cell.y)
            assert panel.footprint.contains_properly(negative)
            negatives.append(negative)
        material = panel.footprint.difference(union_all(negatives))
        assert isinstance(material, Polygon) and material.is_valid
        all_slots.extend(negatives)
    for a, b in combinations(layout.panels, 2):
        assert a.footprint.intersection(b.footprint).area == 0
    protected = union_all(all_slots)
    panels = {panel.id: panel for panel in layout.panels}
    for joint in layout.joints:
        assert joint.female.covers(joint.male)
        assert joint.male.disjoint(protected)
        assert joint.female.disjoint(protected)
        assert layout.usable.covers(joint.female)
        assert panels[joint.male_panel].footprint.covers(joint.male)
        assert panels[joint.female_panel].footprint.intersection(joint.female).area == 0
    covered = union_all([panel.footprint for panel in layout.panels])
    assert layout.unused.intersection(covered).area == 0
    assert layout.unused.union(covered).symmetric_difference(layout.usable).area < 1e-7


def test_rectangle_reserves_bed_space_and_preserves_all_cells():
    layout = compile_fixture(box(0, 0, 448, 112))
    assert len(layout.panels) == 2
    assert len(layout.joints) == 2
    assert sum(len(panel.cells) for panel in layout.panels) == 64
    # A tab actually increases the nominal 224 mm blank's finished bounds.
    assert any(
        panel.footprint.bounds[2] - panel.footprint.bounds[0] > 224 for panel in layout.panels
    )
    assert layout.unused.area > 0  # The clearance kerfs are reported too.
    assert_layout_contract(layout)


def test_overlapping_keepouts_are_buffered_then_unioned():
    keepouts = (
        Keepout("one", box(60, 60, 95, 95), clearance=3, uncertainty=2),
        Keepout("two", box(85, 65, 120, 100), clearance=4, uncertainty=1),
    )
    layout = compile_fixture(box(0, 0, 224, 168), keepouts=keepouts)
    expected = union_all([k.geometry.buffer(5) for k in keepouts])
    assert layout.keepout_union.equals(expected)
    assert layout.usable.equals(layout.spec.surface.difference(expected))
    assert expected.area < sum(k.expanded.area for k in keepouts)
    assert_layout_contract(layout)


def test_shifted_origin_and_negative_indices_keep_one_phase():
    origin = (3.25, -5.5)
    layout = compile_fixture(box(-80.75, -61.5, 59.25, 50.5), grid_origin=origin)
    cells = [cell for panel in layout.panels for cell in panel.cells]
    assert {(c.ix, c.iy) for c in cells} == {(x, y) for x in range(-3, 2) for y in range(-2, 2)}
    assert_layout_contract(layout)


def test_partial_edge_cells_are_solid_filler_not_sockets():
    layout = compile_fixture(box(0, 0, 60, 30))
    assert [(cell.ix, cell.iy) for panel in layout.panels for cell in panel.cells] == [
        (0, 0),
        (1, 0),
    ]
    assert union_all([p.footprint for p in layout.panels]).equals(layout.usable)
    assert_layout_contract(layout)


def test_round_boundary_hole_and_disconnected_scrap_are_explicit():
    useful = Point(84, 84).buffer(78).difference(Point(84, 84).buffer(17))
    scrap = box(250, 0, 257, 3)
    layout = compile_fixture(MultiPolygon([useful, scrap]), joints=JointSpec(style="none"))
    expected = {
        (ix, iy)
        for iy in range(6)
        for ix in range(6)
        if useful.covers(box(ix * 28, iy * 28, (ix + 1) * 28, (iy + 1) * 28))
    }
    assert {(c.ix, c.iy) for p in layout.panels for c in p.cells} == expected
    assert layout.unused.covers(scrap)
    assert any("cell-less border/scrap" in warning for warning in layout.warnings)
    assert_layout_contract(layout)


def test_disconnected_regions_split_deterministically_even_inside_one_chunk():
    surface = MultiPolygon([box(0, 0, 56, 56), box(112, 0, 168, 56)])
    first = compile_fixture(surface)
    second = compile_fixture(MultiPolygon(list(reversed(surface.geoms))))
    assert len(first.panels) == 2
    assert not first.joints
    assert [(p.id, p.footprint.normalize().wkb, p.cells, p.placement) for p in first.panels] == [
        (p.id, p.footprint.normalize().wkb, p.cells, p.placement) for p in second.panels
    ]
    assert_layout_contract(first)


def test_irregular_fixture_with_injected_projection():
    spec = load_spec(Path(__file__).parents[1] / "examples/irregular.yaml")
    layout = plan_installation(spec, socket_projection=fixture_socket)
    again = plan_installation(spec, socket_projection=fixture_socket)
    assert len(layout.panels) > 1 and layout.joints
    assert [(p.id, p.footprint.wkb, p.cells, p.placement) for p in layout.panels] == [
        (p.id, p.footprint.wkb, p.cells, p.placement) for p in again.panels
    ]
    assert [(j.id, j.male.wkb, j.female.wkb) for j in layout.joints] == [
        (j.id, j.male.wkb, j.female.wkb) for j in again.joints
    ]
    assert layout.warnings == again.warnings
    assert_layout_contract(layout)


def test_irregular_fixture_with_native_projection():
    if importlib.util.find_spec("opengrid.native") is None:
        pytest.skip("CAD slice not integrated yet")
    # Only module absence is skippable: broken native imports/dependencies must fail.
    from opengrid import native

    spec = load_spec(Path(__file__).parents[1] / "examples/irregular.yaml")
    layout = plan_installation(spec)  # Real lazy native-provider path, no injection.
    assert len(layout.panels) > 1 and layout.joints
    assert_layout_contract(layout, native.socket_projection)


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "draft"},
        {"unresolved": ("frame dimensions unknown",)},
        {"keepouts": (Keepout("uncertain", box(0, 0, 1, 1), confidence=0.5),)},
    ],
)
def test_readiness_is_enforced_before_projection_or_native_import(patch, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "opengrid.native", None)
    spec = replace(InstallationSpec("blocked", box(0, 0, 56, 56)), **patch)

    def forbidden_projection(x, y):
        pytest.fail("Projection must not run for an unready spec")

    with pytest.raises(SpecError, match="Printable export blocked"):
        plan_installation(spec, socket_projection=forbidden_projection)
    with pytest.raises(SpecError, match="Printable export blocked"):
        plan_installation(spec)


def test_default_provider_is_loaded_lazily_and_receives_global_centres(monkeypatch):
    import sys

    calls = []
    native = ModuleType("opengrid.native")

    def projection(x, y):
        calls.append((x, y))
        return fixture_socket(x, y)

    native.socket_projection = projection
    monkeypatch.setitem(sys.modules, "opengrid.native", native)
    layout = plan_installation(InstallationSpec("lazy", box(28, 56, 56, 84)))
    assert calls == [(42, 70)]
    assert_layout_contract(layout)


@pytest.mark.parametrize(
    "surface, keepouts, message",
    [
        (box(0, 0, 27.99, 200), (), "no complete native"),
        (box(0, 0, 56, 56), (Keepout("all", box(-1, -1, 57, 57)),), "no usable surface"),
        (box(0, 0, 28_000, 28_000), (), "lattice candidates"),
    ],
)
def test_no_useful_layout_fails_clearly(surface, keepouts, message):
    with pytest.raises(LayoutError, match=message):
        compile_fixture(surface, keepouts=keepouts)


@pytest.mark.parametrize(
    "projection",
    [
        lambda x, y: box(x - 14, y - 14, x + 14, y + 14),
        lambda x, y: Polygon(),
        lambda x, y: Point(x, y),
        lambda x, y: Polygon([(x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)]),
    ],
)
def test_invalid_or_cell_clipping_projection_is_rejected(projection):
    with pytest.raises(LayoutError, match="Socket projection"):
        plan_installation(
            InstallationSpec("bad-negative", box(0, 0, 56, 56)), socket_projection=projection
        )


def test_disconnected_manufacturing_islands_are_rejected():
    def island_projection(x, y):
        return fixture_socket(x, y).difference(box(x - 1, y - 1, x + 1, y + 1))

    with pytest.raises(LayoutError, match="disconnect material"):
        plan_installation(
            InstallationSpec("island", box(0, 0, 56, 56)), socket_projection=island_projection
        )


def test_narrow_connected_border_is_reported_without_inventing_cells():
    surface = union_all([box(0, 0, 28, 28), box(56, 0, 84, 28), box(28, 13.8, 56, 14.2)])
    layout = compile_fixture(surface)
    assert sum(len(p.cells) for p in layout.panels) == 2
    assert any("narrow land/ligament" in warning for warning in layout.warnings)
    assert_layout_contract(layout)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_printer_search_uses_all_four_rotations_and_global_origin(angle):
    shape = union_all([box(0, 0, 70, 10), box(0, 0, 10, 50)])
    turned = rotate(shape, angle, origin=(0, 0))
    x0, y0, x1, y1 = turned.bounds
    target = translate(turned, xoff=-x0, yoff=-y0)
    width, depth = x1 - x0, y1 - y0
    printer = PrinterSpec(
        width=width, depth=depth, margin=0, exclusions=(box(0, 0, width, depth).difference(target),)
    )
    installation_shape = translate(shape, xoff=112, yoff=-224)
    placement = find_print_placement(installation_shape, printer)
    assert placement is not None
    assert placement.rotation == angle
    assert transformed_print_footprint(installation_shape, placement).equals(target)


def test_bed_exclusion_inside_shape_hole_is_legal_not_a_bbox_false_negative():
    shape = box(100, 200, 200, 300).difference(box(140, 240, 160, 260))
    printer = PrinterSpec(width=100, depth=100, margin=0, exclusions=(box(40, 40, 60, 60),))
    placement = find_print_placement(shape, printer)
    assert placement == PrintPlacement(0, -100, -200)
    assert printer.usable_bed.covers(transformed_print_footprint(shape, placement))


def test_feature_aligned_translation_finds_a_non_corner_non_centre_pocket():
    # The bed's four isolated corner islands keep its bounds at 0..100. The only
    # pocket for this part is off-centre; bed-vertex alignment must be considered.
    pocket = box(17, 33, 47, 73)
    usable = union_all(
        [pocket, box(0, 0, 1, 1), box(99, 0, 100, 1), box(0, 99, 1, 100), box(99, 99, 100, 100)]
    )
    printer = PrinterSpec(
        width=100, depth=100, margin=0, exclusions=(box(0, 0, 100, 100).difference(usable),)
    )
    footprint = box(0, 0, 30, 40)
    placement = find_print_placement(footprint, printer)
    assert placement is not None
    assert transformed_print_footprint(footprint, placement).equals(pocket)


def test_printer_margin_and_exclusions_are_not_just_nominal_dimensions():
    printer = PrinterSpec(width=60, depth=60, margin=2, exclusions=())
    assert find_print_placement(box(0, 0, 57, 57), printer) is None
    blocked = replace(printer, exclusions=(box(29, 0, 31, 60),))
    assert find_print_placement(box(0, 0, 28, 28), blocked) is None


def test_oversized_chunks_reduce_on_lattice_lines_until_the_actual_bed_fits():
    printer = PrinterSpec(width=122, depth=122, margin=0.5, exclusions=(box(56, 0, 66, 122),))
    layout = compile_fixture(box(0, 0, 112, 112), printer=printer)
    assert len(layout.panels) > 1
    assert sum(len(p.cells) for p in layout.panels) == 16
    assert_layout_contract(layout)


@pytest.mark.parametrize(
    "printer, message",
    [
        (PrinterSpec(width=28, depth=28, margin=0.5, exclusions=()), "No usable-bed placement"),
        (PrinterSpec(max_panel_span=27), "max_panel_span"),
    ],
)
def test_unprintable_single_cell_fails_instead_of_exporting_fragments(printer, message):
    with pytest.raises(LayoutError, match=message):
        compile_fixture(box(0, 0, 56, 56), printer=printer)


def test_transformed_print_footprint_rotates_then_translates_not_about_centroid():
    footprint = box(28, 56, 84, 112)
    transformed = transformed_print_footprint(footprint, PrintPlacement(90, 120, -20))
    assert transformed.bounds == (8, 8, 64, 64)
    assert math.isclose(transformed.area, footprint.area)


@pytest.mark.parametrize("origin", [(0.1, 0.1), (2.4, 4.2), (17.7, -3.1), (0.3, 0.7), (-0.1, -0.1)])
def test_decimal_grid_origins_do_not_lose_edge_cells_to_float_roundoff(origin):
    ox, oy = origin
    layout = compile_fixture(box(ox, oy, ox + 448, oy + 112), grid_origin=origin)
    assert {(c.ix, c.iy) for p in layout.panels for c in p.cells} == {
        (ix, iy) for ix in range(16) for iy in range(4)
    }
    assert layout.joints
    assert_layout_contract(layout)
