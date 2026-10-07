import pytest
from shapely.geometry import box

from opengrid import JointSpec, Keepout, PrinterSpec, generate_installation
from opengrid.models import Layout
from opengrid.spec import SpecError


def test_expert_boundary_accepts_precise_geometry_and_keepout_provenance():
    obstacle = Keepout(
        "known-obstacle",
        box(56, 0, 84, 28),
        clearance=1,
        confidence=0.98,
        source={"kind": "manufacturer_drawing", "ref": "fixture"},
    )
    layout = generate_installation(
        usable_surface=box(0, 0, 112, 56),
        keepouts=[obstacle],
        printer=PrinterSpec(max_panel_span=56),
        joint_strategy=JointSpec(style="none"),
    )
    assert isinstance(layout, Layout)
    assert layout.spec.keepouts[0].source == obstacle.source
    assert layout.spec.keepouts[0].confidence == 0.98
    assert layout.panels and not layout.joints
    for p in layout.panels:
        assert layout.usable.covers(p.footprint)
        assert p.footprint.intersection(obstacle.expanded).area == 0


def test_expert_grid_phase_and_raw_geometry_input():
    origin = (3.2, -2.8)
    layout = generate_installation(
        box(3.2, -2.8, 115.2, 53.2),
        keepouts=[box(3.2, -2.8, 31.2, 25.2)],
        grid_origin=origin,
        joint_strategy=JointSpec(style="none"),
    )
    assert layout.spec.keepouts[0].source["kind"] == "explicit_geometry"
    for p in layout.panels:
        for c in p.cells:
            assert c.x == pytest.approx(origin[0] + (c.ix + 0.5) * 28)
            assert c.y == pytest.approx(origin[1] + (c.iy + 0.5) * 28)


def test_mounting_constraints_are_not_silently_ignored():
    with pytest.raises(SpecError, match="mounting"):
        generate_installation(box(0, 0, 56, 56), mounting_constraints={"screw_locations": []})
