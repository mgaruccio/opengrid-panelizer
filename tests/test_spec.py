import copy
import math
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon, box

from opengrid.models import InstallationSpec, Keepout, PrinterSpec
from opengrid.spec import SpecError, ensure_ready, from_dict, geometry, load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def document():
    return {
        "schema_version": 1,
        "installation": {
            "id": "fixture",
            "status": "ready",
            "units": "mm",
            "board": "lite",
            "surface": {"type": "rectangle", "width": 84, "depth": 56},
        },
    }


def test_synthetic_draft_is_explicitly_unresolved():
    spec = load_spec(ROOT / "examples/draft-installation.yaml")
    assert spec.surface.bounds == (0, 0, 1200, 600)
    assert spec.metadata["desktop_thickness"] == 25
    assert spec.metadata["mounting"] == "user_handled"
    with pytest.raises(SpecError, match="Printable export blocked"):
        ensure_ready(spec)


def test_precise_fixture_ready_and_provenance_preserved():
    spec = load_spec(ROOT / "examples/irregular.yaml")
    ensure_ready(spec)
    k = spec.keepouts[0]
    assert k.source["kind"] == "explicit_geometry"
    assert k.confidence == 1 and k.clearance == 4
    assert spec.surface.contains(Point(84, 84))


@pytest.mark.parametrize("field,value", [("units", "inch"), ("board", "unknown"), ("oops", 1)])
def test_explicit_units_board_and_unknown_fields(field, value):
    raw = document()
    raw["installation"][field] = value
    with pytest.raises(SpecError):
        from_dict(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "84", -1, 0])
def test_bad_surface_dimensions(value):
    raw = document()
    raw["installation"]["surface"]["width"] = value
    with pytest.raises(SpecError):
        from_dict(raw)


def test_self_intersection_is_not_silently_repaired():
    with pytest.raises(SpecError, match="Invalid"):
        geometry({"type": "polygon", "points": [[0, 0], [20, 20], [0, 20], [20, 0]]})


def test_geometry_holes_disconnected_regions_and_rounding():
    p = geometry(
        {
            "type": "polygon",
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "holes": [[[20, 20], [40, 20], [40, 40], [20, 40]]],
        }
    )
    assert p.area == 9600 and len(p.interiors) == 1
    rounded = geometry({"type": "rounded_rectangle", "width": 84, "depth": 56, "radius": 8})
    assert rounded.bounds == (0, 0, 84, 56) and rounded.area < 84 * 56
    disconnected = geometry(
        {
            "type": "multi_polygon",
            "polygons": [
                {"type": "rectangle", "width": 28, "depth": 28},
                {"type": "rectangle", "x": 56, "y": 0, "width": 28, "depth": 28},
            ],
        }
    )
    assert len(disconnected.geoms) == 2


def test_circle_keepout_encloses_true_circle_between_vertices():
    g = geometry({"type": "circle", "x": 0, "y": 0, "radius": 10}, conservative=True)
    for i in range(512):
        a = i * 2 * math.pi / 512
        assert g.buffer(1e-10).covers(Point(10 * math.cos(a), 10 * math.sin(a)))


def test_confidence_is_not_dimensional_uncertainty():
    k = Keepout(
        "obstacle",
        box(20, 20, 40, 40),
        clearance=2,
        uncertainty=3,
        confidence=0.9,
        source={"kind": "user_photo"},
    )
    assert k.expanded.bounds == pytest.approx((15, 15, 45, 45))
    spec = InstallationSpec("fixture", box(0, 0, 84, 84), keepouts=(k,))
    ensure_ready(spec)
    uncertain = copy.copy(k)
    object.__setattr__(uncertain, "confidence", 0.7)
    with pytest.raises(SpecError, match="confidence below"):
        ensure_ready(InstallationSpec("fixture", spec.surface, keepouts=(uncertain,)))


@pytest.mark.parametrize(
    "spec",
    [
        InstallationSpec("fixture", box(0, 0, 56, 56), grid_origin=(float("nan"), 0)),
        InstallationSpec("fixture", box(0, 0, 56, 56), printer=PrinterSpec(margin=200)),
        InstallationSpec("fixture", box(0, 0, 56, 56), printer=PrinterSpec(width=float("inf"))),
        InstallationSpec("fixture", Polygon([(0, 0, 1), (56, 0, 1), (56, 56, 1), (0, 56, 1)])),
    ],
)
def test_expert_api_cannot_bypass_geometry_validation(spec):
    with pytest.raises(SpecError):
        validate_spec(spec)


def test_duplicate_keepouts_and_ready_unresolved_are_blocked():
    k = Keepout("duplicate", box(10, 10, 20, 20))
    with pytest.raises(SpecError, match="unique"):
        ensure_ready(InstallationSpec("fixture", box(0, 0, 84, 84), keepouts=(k, k)))
    with pytest.raises(SpecError, match="measure rail"):
        ensure_ready(InstallationSpec("fixture", box(0, 0, 84, 84), unresolved=("measure rail",)))


def test_yaml_load_is_safe(tmp_path):
    f = tmp_path / "unsafe.yaml"
    f.write_text("!!python/object/apply:os.system ['echo bad']")
    with pytest.raises(SpecError, match="Cannot read"):
        load_spec(f)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"origin": "front_left", "x": "left", "y": "back"},
        {"origin": "rear_left", "x": "left", "y": "front"},
        {"origin": "rear_left", "x": "right", "y": "back"},
    ],
)
def test_opposite_coordinate_declarations_are_not_silently_mislabelled(coordinates):
    raw = document()
    raw["installation"]["coordinate_system"] = coordinates
    with pytest.raises(SpecError, match="coordinate_system"):
        from_dict(raw)


def test_origin_only_input_normalizes_the_supported_axes():
    raw = document()
    raw["installation"]["coordinate_system"] = {"origin": "rear_left"}
    spec = from_dict(raw)
    ensure_ready(spec)
    assert spec.coordinate_system == {"origin": "rear_left", "x": "right", "y": "front"}


def test_conservative_rounded_keepout_covers_true_corner_arcs():
    nominal = {"type": "rounded_rectangle", "width": 500, "depth": 400, "radius": 100}
    g = geometry(nominal, conservative=True)
    assert g.bounds == (0, 0, 500, 400)
    for cx, cy, start in [
        (100, 100, math.pi),
        (400, 100, -math.pi / 2),
        (400, 300, 0),
        (100, 300, math.pi / 2),
    ]:
        for i in range(129):
            a = start + i * math.pi / 256
            assert g.buffer(1e-8).covers(Point(cx + 100 * math.cos(a), cy + 100 * math.sin(a)))
