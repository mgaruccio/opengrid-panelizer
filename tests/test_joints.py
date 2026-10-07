"""Key retention, rejection, and post-boolean safety with synthetic slot envelopes."""

from dataclasses import replace
from itertools import combinations

import pytest
from shapely import union_all
from shapely.affinity import rotate, translate
from shapely.geometry import GeometryCollection, LineString, Polygon, box

from opengrid.joints import add_joints, puzzle_profiles
from opengrid.layout import (
    LayoutError,
    find_print_placement,
    plan_installation,
    transformed_print_footprint,
)
from opengrid.models import InstallationSpec, JointSpec, Panel, PrinterSpec, PrintPlacement


def projection(x, y):
    # Conservative whole negative envelope of a synthetic 24 mm test socket only.
    return box(x - 12, y - 12, x + 12, y + 12)


def test_puzzle_has_a_retaining_head_not_a_widening_neck_at_the_seam():
    spec = JointSpec()
    male, female = puzzle_profiles(spec)
    neck = LineString([(0.1 * spec.depth, -10), (0.1 * spec.depth, 10)])
    head = LineString([(0.85 * spec.depth, -10), (0.85 * spec.depth, 10)])
    assert male.intersection(neck).length == pytest.approx(spec.neck_width)
    assert male.intersection(head).length == pytest.approx(spec.head_width)
    assert female.intersection(neck).length < male.intersection(head).length
    assert female.equals(male.buffer(spec.clearance, join_style="mitre"))
    assert female.contains_properly(male)
    assert male.bounds[0] < 0 < male.bounds[2]


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_profiles_use_installation_coordinates_and_preserve_clearance(angle):
    male, female = puzzle_profiles(JointSpec(), 224, 56, angle)
    original_male, original_female = puzzle_profiles(JointSpec())
    assert male.equals(translate(rotate(original_male, angle, origin=(0, 0)), 224, 56))
    assert female.equals(translate(rotate(original_female, angle, origin=(0, 0)), 224, 56))
    assert female.contains_properly(male)


@pytest.mark.parametrize(
    "patch",
    [
        {"head_width": 1.2},
        {"head_width": 1.3},
        {"clearance": -0.1},
        {"clearance": 0.6},
        {"depth": 0},
        {"head_width": float("nan")},
    ],
)
def test_nonretaining_or_invalid_key_parameters_fail_clearly(patch):
    with pytest.raises(ValueError, match="Invalid puzzle"):
        puzzle_profiles(replace(JointSpec(), **patch))


def test_non_quarter_turn_profile_is_rejected():
    with pytest.raises(ValueError, match="normal"):
        puzzle_profiles(JointSpec(), angle=45)


def test_joint_changes_and_clearance_stay_inside_the_original_two_panel_union():
    spec = InstallationSpec("two", box(0, 0, 112, 56), printer=PrinterSpec(max_panel_span=56))
    layout = plan_installation(spec, socket_projection=projection)
    assert len(layout.joints) == 1
    joint = layout.joints[0]
    panels = {p.id: p for p in layout.panels}
    male, female = panels[joint.male_panel], panels[joint.female_panel]
    assert joint.angle == 0
    original_male, original_female = box(0, 0, 56, 56), box(56, 0, 112, 56)
    original_pair = original_male.union(original_female)
    assert original_pair.covers(joint.male)
    assert original_pair.covers(joint.female)
    assert male.footprint.equals(original_male.union(joint.male))
    assert female.footprint.equals(original_female.difference(joint.female))
    assert male.footprint.intersection(female.footprint).area == 0
    expected_gap = joint.female.intersection(original_female).difference(joint.male)
    assert layout.unused.equals(expected_gap)
    slots = union_all([projection(c.x, c.y) for p in layout.panels for c in p.cells])
    assert joint.female.disjoint(slots)
    for panel in layout.panels:
        assert isinstance(panel.footprint.difference(slots), Polygon)
        assert spec.printer.usable_bed.covers(
            transformed_print_footprint(panel.footprint, panel.placement)
        )


def test_horizontal_seam_normal_points_from_male_to_female():
    spec = InstallationSpec(
        "horizontal", box(0, 0, 56, 112), printer=PrinterSpec(max_panel_span=56)
    )
    layout = plan_installation(spec, socket_projection=projection)
    assert len(layout.joints) == 1
    assert layout.joints[0].angle == 90
    assert (layout.joints[0].x, layout.joints[0].y) == (28, 56)


def test_crowded_slot_projections_block_keys_and_report_the_seam():
    spec = InstallationSpec("crowded", box(0, 0, 448, 112))

    def wide_negative(x, y):
        return box(x - 13.9, y - 13.9, x + 13.9, y + 13.9)

    layout = plan_installation(spec, socket_projection=wide_negative)
    assert len(layout.panels) == 2
    assert not layout.joints
    assert any("no safe puzzle joint" in w for w in layout.warnings)
    for panel in layout.panels:
        for cell in panel.cells:
            assert panel.footprint.contains_properly(wide_negative(cell.x, cell.y))


def test_short_seam_without_safe_lattice_crossing_is_reported_not_faked():
    spec = InstallationSpec(
        "single-row", box(0, 0, 112, 28), printer=PrinterSpec(max_panel_span=56)
    )
    layout = plan_installation(spec, socket_projection=projection)
    assert len(layout.panels) == 2
    assert not layout.joints
    assert any("no safe puzzle joint" in w for w in layout.warnings)


def test_finished_tab_that_exceeds_bed_is_rejected_without_partial_mutation():
    printer = PrinterSpec(width=56, depth=56, margin=0, max_panel_span=56, exclusions=())
    spec = InstallationSpec("exact-bed", box(0, 0, 112, 56), printer=printer)
    layout = plan_installation(spec, socket_projection=projection)
    assert len(layout.panels) == 2
    assert not layout.joints
    assert layout.panels[0].footprint.equals(box(0, 0, 56, 56))
    assert layout.panels[1].footprint.equals(box(56, 0, 112, 56))
    assert any("usable-bed space" in warning for warning in layout.warnings)
    for panel in layout.panels:
        assert printer.usable_bed.covers(
            transformed_print_footprint(panel.footprint, panel.placement)
        )


def test_failed_female_cut_that_would_disconnect_material_tries_the_other_gender():
    # Receiver's two lobes are joined only by a bridge exactly as deep as the
    # clearance tool. Cutting it disconnects the panel, but reversing genders is safe.
    left = box(-28, -28, 0, 28)
    right = union_all([box(0, -28, 28, -0.1), box(0, 0.1, 28, 28), box(0, -0.1, 1.65, 0.1)])
    spec = InstallationSpec("bridge", left.union(right))
    panels = [
        Panel("P001", left, (), PrintPlacement(0, 0, 0)),
        Panel("P002", right, (), PrintPlacement(0, 0, 0)),
    ]
    joints, warnings = add_joints(
        panels,
        spec,
        spec.surface,
        GeometryCollection(),
        lambda shape: find_print_placement(shape, spec.printer),
    )
    assert len(joints) == 1
    assert joints[0].male_panel == "P002"
    assert joints[0].angle == 180
    assert not warnings
    assert all(
        isinstance(panel.footprint, Polygon) and panel.footprint.is_valid for panel in panels
    )
    assert panels[0].footprint.intersection(panels[1].footprint).area == 0


def test_multiple_seams_never_transfer_material_to_a_third_panel_or_a_socket():
    spec = InstallationSpec(
        "six-panels", box(0, 0, 168, 112), printer=PrinterSpec(max_panel_span=56)
    )
    layout = plan_installation(spec, socket_projection=projection)
    assert len(layout.panels) == 6
    assert len(layout.joints) == 7
    slots = union_all([projection(c.x, c.y) for p in layout.panels for c in p.cells])
    for first, second in combinations(layout.panels, 2):
        assert first.footprint.intersection(second.footprint).area == 0
    for joint in layout.joints:
        assert joint.female.disjoint(slots)
        for other in layout.panels:
            if other.id not in (joint.male_panel, joint.female_panel):
                assert other.footprint.intersection(joint.female).area == 0
    for panel in layout.panels:
        assert isinstance(panel.footprint.difference(slots), Polygon)
        assert spec.printer.usable_bed.covers(
            transformed_print_footprint(panel.footprint, panel.placement)
        )


def test_requested_no_joints_does_not_claim_or_warn_about_interlocks():
    spec = InstallationSpec("no-keys", box(0, 0, 448, 112), joints=JointSpec(style="none"))
    layout = plan_installation(spec, socket_projection=projection)
    assert len(layout.panels) == 2
    assert not layout.joints
    assert not any("Seam" in warning for warning in layout.warnings)
    assert layout.unused.is_empty


@pytest.mark.parametrize("width", [56, 448])
def test_clearance_erased_retention_blocks_planning_before_native_import(width):
    spec = InstallationSpec("nonretaining", box(0, 0, width, 56), joints=JointSpec(head_width=1.3))
    with pytest.raises(LayoutError, match="Invalid puzzle"):
        plan_installation(spec)
