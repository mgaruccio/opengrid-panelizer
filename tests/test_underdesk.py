"""Production connector tools on native-node land, not oversized blank slabs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh
from build123d import (
    Align,
    Box,
    Compound,
    Face,
    Location,
    ShapeList,
    Solid,
    Wire,
    export_step,
    export_stl,
    import_step,
)
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from opengrid.cad import _socket_cutter
from opengrid.native import socket_projection
from opengrid.underdesk import (
    ASSEMBLY_TRAVEL,
    FLEX_ENVELOPE,
    LATCH_EFFECTIVE_LENGTH,
    LATCH_HALF_WIDTH,
    LATCH_RAMP_START_X,
    LATCH_RELEASE_TRAVEL,
    LATCH_ROOT_RADIUS,
    LATCH_ROOT_X,
    LATCH_SHOULDER_X,
    LATCH_SHOULDER_Y,
    LATCH_TIP_X,
    LATCH_TOOTH_RADIUS,
    RESERVATION_RADIUS,
    THICKNESS,
    TONGUE_BOTTOM_HALF_WIDTH,
    TONGUE_CENTRES,
    TONGUE_DEPTH,
    TONGUE_TAPER_BOTTOM_Z,
    TONGUE_TAPER_TOP_Z,
    TONGUE_TOP_HALF_WIDTH,
    underdesk_plan,
    underdesk_solids,
)

TAPER = (TONGUE_BOTTOM_HALF_WIDTH - TONGUE_TOP_HALF_WIDTH) / (
    TONGUE_TAPER_TOP_Z - TONGUE_TAPER_BOTTOM_Z
)
FIT_LADDER = (0, 0.05, 0.1)


def test_retaining_face_stays_flat_above_the_receiver_ledge(joint):
    clearance, _, _, male, female = joint
    # Measure the actual cut node, not just nominal tooth dimensions. Both
    # contact faces must be normal to withdrawal, not rounded cam surfaces.
    receiver_x = LATCH_SHOULDER_X - clearance
    receiver_face = _projection(female).boundary.intersection(
        LineString([(receiver_x, LATCH_HALF_WIDTH), (receiver_x, LATCH_SHOULDER_Y + 0.5)])
    )
    ledge_y = receiver_face.bounds[1]
    assert ledge_y == pytest.approx(LATCH_HALF_WIDTH + clearance)
    face = _projection(male).boundary.intersection(
        LineString([(LATCH_SHOULDER_X, ledge_y), (LATCH_SHOULDER_X, LATCH_SHOULDER_Y)])
    )
    assert face.length >= 0.45 - clearance - 1e-7
    # The measured overlap is backed by stock for the full four-mm height.
    top = face.bounds[3]
    for panel, x0, x1 in (
        (male, LATCH_SHOULDER_X, LATCH_SHOULDER_X + 0.1),
        (female, receiver_x - 0.1, receiver_x),
    ):
        stock = _box(x0, ledge_y, 0, x1, top, THICKNESS)
        assert _volume(stock.cut(panel)) < 1e-7


@pytest.mark.parametrize("shrink,rounding", [(0, 0.1), (0.1, 0), (0.1, 0.1)])
def test_printed_hook_envelope_keeps_a_flat_catch_after_small_loss(joint, shrink, rounding):
    clearance, plan, _, male, female = joint
    latch = _projection(male).intersection(_latch_projection(plan))
    # Erode the actual contour, then round its corners without adding material.
    # These are bounded geometric print deviations, NOT a retention/load rating.
    shrunken = latch.buffer(-shrink, quad_segs=16)
    worn = (
        shrunken.buffer(-rounding, quad_segs=16)
        .buffer(rounding, quad_segs=16)
        .intersection(shrunken)
        if rounding
        else shrunken
    )
    assert isinstance(worn, Polygon) and worn.is_valid
    assert latch.buffer(1e-7).covers(worn)
    hook_x = LATCH_SHOULDER_X + shrink
    receiver_x = LATCH_SHOULDER_X - clearance
    receiver_face = _projection(female).boundary.intersection(
        LineString([(receiver_x, LATCH_HALF_WIDTH), (receiver_x, LATCH_SHOULDER_Y + 0.5)])
    )
    face = worn.boundary.intersection(
        LineString([(hook_x, receiver_face.bounds[1]), (hook_x, LATCH_SHOULDER_Y)])
    )
    # Even simultaneous 0.1 mm shrink and corner rounding leave >=0.2 mm
    # of flat engagement at the loosest fit, versus the old nominal 0.1 mm.
    assert face.length >= 0.3 - clearance - 1e-7
    printed_latch = _prism(worn)
    assert _volume(printed_latch.intersect(female)) < 1e-7
    free = printed_latch.moved(Location((-clearance - shrink, 0, 0)))
    assert _volume(free.intersect(female)) < 1e-7
    caught = free.moved(Location((-0.05, 0, 0)))
    assert _volume(caught.intersect(female)) >= face.length * 0.05 * THICKNESS - 1e-7


def _box(x0, y0, z0, x1, y1, z1):
    return Box(x1 - x0, y1 - y0, z1 - z0, align=(Align.MIN,) * 3).moved(Location((x0, y0, z0)))


def _shape(result):
    # OCC intersections can include zero-volume touching faces at zero clearance.
    if result is None:
        return Compound(children=[])
    return Compound(children=result) if isinstance(result, ShapeList) else result


def _volume(shape):
    return _shape(shape).volume


def _prism(polygon, z=0, height=THICKNESS):
    return Solid.extrude(
        Face(Wire.make_polygon([(x, y, z) for x, y in polygon.exterior.coords])),
        (0, 0, height),
    )


def _projection(shape):
    vertices, triangles = shape.tessellate(0.01)
    return unary_union(
        [
            Polygon([(vertices[i].X, vertices[i].Y) for i in triangle]).convex_hull
            for triangle in triangles
        ]
    )


def _latch_projection(plan):
    """Select the WHOLE spring; clipping it to the flex envelope can hide collisions."""
    assert plan.male_add.geom_type == "MultiPolygon"
    assert len(plan.male_add.geoms) == 2
    return min(plan.male_add.geoms, key=lambda component: component.bounds[0])


@pytest.fixture(scope="module", params=FIT_LADDER)
def joint(request):
    clearance = request.param
    plan, tools = underdesk_plan(clearance), underdesk_solids(clearance)
    # Deliberately use ONLY the guaranteed native diamond stock. Neither an
    # oversized slab nor a filled-in accessory socket may rescue this connector.
    left = plan.reservation.intersection(box(-6, -6, 0, 6))
    right = plan.reservation.intersection(box(0, -6, 6, 6))
    male = _prism(left).fuse(tools.male_add).cut(tools.male_cut)
    female = _prism(right).fuse(tools.female_add).cut(tools.female_cut)
    return clearance, plan, tools, male, female


def test_planning_and_invalid_cad_requests_do_not_import_cad():
    code = """
import builtins
original = builtins.__import__
def without_cad(name, *args, **kwargs):
    assert not name.startswith(('build123d', 'OCP')), name
    return original(name, *args, **kwargs)
builtins.__import__ = without_cad
from opengrid.underdesk import underdesk_plan, underdesk_solids
assert underdesk_plan().reservation.is_valid
assert underdesk_plan(0).reservation.is_valid
try:
    underdesk_solids(-0.01)
except ValueError:
    pass
else:
    raise AssertionError('invalid clearance accepted')
"""
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize(
    "clearance",
    [-0.001, 0.1001, 0.35, float("nan"), float("inf"), float("-inf"), True, None, "0.05"],
)
@pytest.mark.parametrize("factory", [underdesk_plan, underdesk_solids])
def test_clearance_is_finite_and_bounded(factory, clearance):
    with pytest.raises(ValueError, match="clearance"):
        factory(clearance)


def test_default_is_the_middle_tight_fit():
    default, middle = underdesk_plan(), underdesk_plan(0.05)
    for name in ("male_add", "male_cut", "female_add", "female_cut", "reservation"):
        assert getattr(default, name).equals(getattr(middle, name))
    default_tools, middle_tools = underdesk_solids(), underdesk_solids(0.05)
    for name in ("male_add", "male_cut", "female_add", "female_cut"):
        assert _volume(getattr(default_tools, name).cut(getattr(middle_tools, name))) < 1e-7
        assert _volume(getattr(middle_tools, name).cut(getattr(default_tools, name))) < 1e-7


def test_strengthened_spring_has_a_longer_lower_strain_proxy_and_stockworthy_tooth(joint):
    _, plan, _, male, _ = joint
    latch = _projection(male).intersection(_latch_projection(plan))
    width = latch.intersection(LineString([(0, -2), (0, 2)])).length
    beam = latch.boundary.intersection(
        LineString([(-6, LATCH_HALF_WIDTH), (LATCH_SHOULDER_X, LATCH_HALF_WIDTH)])
    )
    # Geometry proxy only, not FEA or a load rating. Compare with the LAST
    # 1.0/4.5/0.30 spring, not the weaker, older 0.8/3.9/0.30 design.
    previous_proxy = 1.0 * 0.30 / 4.5**2
    current_proxy = width * LATCH_RELEASE_TRAVEL / beam.length**2
    assert width == pytest.approx(1.0)
    assert LATCH_ROOT_RADIUS == pytest.approx(0.5)
    assert beam.length == pytest.approx(LATCH_EFFECTIVE_LENGTH)
    assert beam.length > 4.5
    assert current_proxy <= previous_proxy
    tooth = latch.intersection(box(LATCH_SHOULDER_X, 0, LATCH_TIP_X, 2))
    assert tooth.bounds[3] - width / 2 == pytest.approx(0.5)
    plateau = tooth.boundary.intersection(
        LineString([(LATCH_SHOULDER_X, tooth.bounds[3]), (LATCH_TIP_X, tooth.bounds[3])])
    )
    assert plateau.length == pytest.approx(0.6)
    assert LATCH_TIP_X > TONGUE_DEPTH
    assert LATCH_TIP_X - LATCH_RAMP_START_X >= 0.5


def test_reservation_preserves_all_four_native_socket_projections(joint):
    _, plan, _, _, _ = joint
    assert plan.reservation.area == pytest.approx(2 * RESERVATION_RADIUS**2)
    assert plan.reservation.area < 70
    assert not plan.reservation.equals(box(*plan.reservation.bounds))
    for x in (-14, 14):
        for y in (-14, 14):
            cavity = socket_projection(x, y)
            assert not plan.reservation.intersects(cavity)
            assert plan.reservation.distance(cavity) > 0.03


def test_tools_and_node_halves_are_valid_connected_and_native_thickness(joint):
    _, _, tools, male, female = joint
    for tool in (tools.male_add, tools.male_cut, tools.female_add, tools.female_cut):
        assert tool.is_valid
    for panel in (male, female):
        assert panel.is_valid
        assert len(panel.solids()) == 1
        assert panel.volume > 100
        bounds = panel.bounding_box()
        assert bounds.min.Z == pytest.approx(0, abs=1e-7)
        assert bounds.max.Z == pytest.approx(4, abs=1e-7)
    assert _volume(male.intersect(female)) < 1e-7


def test_plan_matches_additive_projection_and_reserves_tools_stock_and_flexion(joint):
    clearance, plan, tools, _, _ = joint
    assert plan.reservation.covers(FLEX_ENVELOPE)
    for name in ("male_add", "male_cut", "female_add", "female_cut"):
        footprint = getattr(plan, name)
        projected = _projection(getattr(tools, name))
        assert footprint.is_valid
        assert plan.reservation.covers(projected)
        assert plan.reservation.covers(footprint)
        if name.endswith("add"):
            assert footprint.buffer(1e-7).covers(projected)
            assert footprint.symmetric_difference(projected).area < 1e-6
        else:
            # Plan cuts remain ONLY through-slots, never flattened capture pockets.
            polygons = [footprint] if isinstance(footprint, Polygon) else footprint.geoms
            for polygon in polygons:
                assert _volume(_prism(polygon).cut(getattr(tools, name))) < 1e-6
            assert projected.difference(footprint).area > 1
            assert footprint.bounds[1] >= -1.7
            assert footprint.bounds[3] <= 1.7

    # The vertical base of each receiver's outer wall has >= 0.8 mm of stock
    # even within the conservative reserved diamond (not an oversized slab).
    for sign in (-1, 1):
        corner = Point(
            -sign * (TONGUE_DEPTH + clearance),
            sign * (abs(TONGUE_CENTRES[0]) + TONGUE_BOTTOM_HALF_WIDTH + clearance),
        )
        assert plan.reservation.boundary.distance(corner) >= 0.8
        # A continuous relief starts just outside the enlarged, filleted root;
        # the fixed root guard keeps native diagonal stock from becoming thin.
        root_outer = LATCH_HALF_WIDTH + LATCH_ROOT_RADIUS
        root_relief = box(
            LATCH_ROOT_X,
            root_outer * sign,
            LATCH_ROOT_X + LATCH_ROOT_RADIUS,
            (root_outer + 0.2) * sign,
        )
        assert plan.male_cut.covers(root_relief)
    assert 2 * TONGUE_TOP_HALF_WIDTH >= 0.8
    assert 2 * LATCH_HALF_WIDTH >= 0.8
    assert LATCH_ROOT_RADIUS >= LATCH_HALF_WIDTH
    assert clearance / TAPER <= 0.2 + 1e-9


def test_opposite_rails_capture_both_z_directions_away_from_the_snap(joint):
    clearance, _, _, male, female = joint
    for sign in (-1, 1):
        free = male.moved(Location((0, 0, sign * clearance / TAPER * 0.9)))
        assert _volume(free.intersect(female)) < 1e-7
        # Capture begins within 0.02 mm of the stated play, not only after a
        # visibly loose fraction of the 4 mm board thickness.
        first_contact = male.moved(Location((0, 0, sign * (clearance / TAPER + 0.02))))
        assert _volume(first_contact.intersect(female)) > 0.001
        caught = male.moved(Location((0, 0, sign * (clearance / TAPER + 0.3))))
        collision = _shape(caught.intersect(female))
        assert _volume(collision) > 0.1
        # +Z is stopped by the lower-Y male rail, -Z by the female-owned upper rail.
        lane = _box(-2, -3.2 if sign > 0 else 1, -1, 2, -1 if sign > 0 else 3.2, 5)
        assert _volume(collision.cut(lane)) < 1e-6


def test_straight_assembly_interferes_only_at_the_snap(joint):
    clearance, _, _, male, female = joint
    hook = _box(
        LATCH_SHOULDER_X - LATCH_TOOTH_RADIUS,
        LATCH_HALF_WIDTH,
        0,
        LATCH_TIP_X,
        LATCH_SHOULDER_Y,
        4,
    )
    saw_click_interference = False
    for step in range(21):
        shift = -ASSEMBLY_TRAVEL * step / 20
        location = Location((shift, 0, 0))
        collision = _shape(male.moved(location).intersect(female))
        if _volume(collision) > 1e-7:
            saw_click_interference = True
            assert _volume(collision.cut(hook.moved(location))) < 1e-6
        if step in (0, 20):
            assert _volume(collision) < 1e-7
    assert saw_click_interference
    # The raised tooth really blocks withdrawal, including the loosest ladder
    # member; this is not merely a thickness-captured slide fit.
    assert _volume(male.moved(Location((-clearance - 0.1, 0, 0))).intersect(female)) > 0.01


def test_relaxed_snap_has_running_clearance_and_an_accessible_release_path(joint):
    clearance, plan, tools, male, female = joint
    latch = _latch_projection(plan)
    latch_solid = _prism(latch)
    assert _volume(latch_solid.cut(male)) < 1e-7
    assert latch_solid.distance_to(female) == pytest.approx(clearance, abs=1e-7)

    # Kinematic space check, NOT FEA: the actual production contour follows a
    # continuous cantilever displacement field, fixed at its filleted root.
    root_end = LATCH_ROOT_X + LATCH_ROOT_RADIUS
    length = LATCH_EFFECTIVE_LENGTH

    def displacement(x):
        t = max(0, (x - root_end) / length)
        if t <= 1:
            return LATCH_RELEASE_TRAVEL * t * t * (3 - t) / 2
        return LATCH_RELEASE_TRAVEL * (1 + 1.5 * (t - 1))

    rigid_male = male.cut(latch_solid)
    lower_rail = _shape(tools.male_add.cut(latch_solid))
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        released_latch = Polygon(
            [(x, y - fraction * displacement(x)) for x, y in latch.segmentize(0.15).exterior.coords]
        )
        assert released_latch.is_valid
        assert FLEX_ENVELOPE.covers(released_latch)
        bent_latch_solid = _prism(released_latch)
        # Full solids check EVERY Z, especially the bottom-wide male rail; a
        # thin slice or a trimmed tooth-only proxy would miss this collision.
        assert _volume(rigid_male.intersect(bent_latch_solid)) < 1e-6
        assert bent_latch_solid.distance_to(lower_rail) >= 0.04
        assert _volume(bent_latch_solid.intersect(female)) < 1e-6
    released_male = rigid_male.fuse(bent_latch_solid)
    assert released_male.is_valid
    assert len(released_male.solids()) == 1
    for step in range(21):
        shift = -ASSEMBLY_TRAVEL * step / 20
        assert _volume(released_male.moved(Location((shift, 0, 0))).intersect(female)) < 1e-6

    # A 0.4 mm-wide release approach remains open through BOTH faces, even at
    # zero fit clearance; access must not depend on the nominal running gap.
    access_xy = (LATCH_SHOULDER_X + 0.3, LATCH_SHOULDER_Y + 0.25)
    access = _box(
        access_xy[0] - 0.2, access_xy[1] - 0.2, -1, access_xy[0] + 0.2, access_xy[1] + 0.2, 5
    )
    assert plan.female_cut.covers(box(*_projection(access).bounds))
    assert _volume(access.intersect(female)) < 1e-7
    assert _volume(access.intersect(male)) < 1e-7


def test_flat_print_has_connected_layers_and_no_roof_or_steep_underside(joint):
    _, _, _, male, female = joint
    for panel in (male, female):
        # Every 0.2 mm print layer, not just the two endpoint sections.
        for index in range(20):
            z = index * 0.2
            layer = _shape(panel.intersect(_box(-6, -6, z, 6, 6, z + 0.1)))
            assert layer.is_valid
            assert len(layer.solids()) == 1
        for face in panel.faces():
            if face.center().Z > 1e-7:
                assert face.normal_at().Z >= -TAPER / (1 + TAPER**2) ** 0.5 - 1e-7


def test_production_tools_preserve_native_cavities_and_round_trip(joint, tmp_path):
    _, _, tools, _, _ = joint
    cutter = _socket_cutter()
    for name, x0, x1, centre in (("male", -28, 0, -14), ("female", 0, 28, 14)):
        base = _box(x0, -28, 0, x1, 28, 4)
        base = base.fuse(getattr(tools, f"{name}_add")).cut(getattr(tools, f"{name}_cut"))
        # Actual native 3-D socket cutters, not a simplified plan-only cavity.
        sockets = [cutter.moved(Location((centre, y, 0))) for y in (-14, 14)]
        shape = base.cut(*sockets)
        assert base.volume - shape.volume == pytest.approx(2 * cutter.volume, abs=1e-6)
        assert shape.is_valid
        assert len(shape.solids()) == 1
        step, stl = tmp_path / f"{name}.step", tmp_path / f"{name}.stl"
        export_step(shape, step)
        export_stl(shape, stl)
        reloaded = import_step(step)
        assert reloaded.is_valid
        assert len(reloaded.solids()) == 1
        assert reloaded.volume == pytest.approx(shape.volume, abs=1e-5)
        assert _volume(reloaded.cut(shape)) < 1e-5
        assert _volume(shape.cut(reloaded)) < 1e-5
        mesh = trimesh.load_mesh(stl)
        assert mesh.is_watertight
        assert mesh.is_winding_consistent
        assert len(mesh.split()) == 1
        assert mesh.volume == pytest.approx(shape.volume, abs=1e-3)
