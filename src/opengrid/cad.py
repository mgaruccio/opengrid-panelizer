"""Native openGrid Lite CAD construction using build123d.

The public functions keep the panel in installation XY with its bottom at Z=0.
The native accessory and mounting-hole cutters are derived from official Lite
STEP geometry; panel boundaries and adapted joint profiles remain owned by the
caller's Shapely footprint.
"""

from __future__ import annotations

from functools import lru_cache
from math import isclose
from pathlib import Path

from build123d import Axis, Face, Location, Part, Plane, Solid, Wire, extrude, import_step
from shapely.geometry import Polygon

from .models import THICKNESS, Panel
from .native import mounting_holes

_NATIVE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "native"
SOCKET_CUTTER_ASSET = _NATIVE_ASSET_DIR / "lite4mm_socket_cutter.step"
MOUNTING_CUTTER_ASSET = _NATIVE_ASSET_DIR / "lite4mm_mounting_cutter.step"


def _ring_wire(coordinates: object) -> Wire:
    """Convert one Shapely ring to a planar build123d wire."""

    points = list(coordinates)  # type: ignore[arg-type]
    if points and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise ValueError("a footprint ring needs at least three points")
    return Wire.make_polygon([(float(point[0]), float(point[1]), 0.0) for point in points])


def _footprint_face(footprint: Polygon) -> Face:
    if footprint.is_empty or footprint.area <= 0:
        raise ValueError("panel footprint must have positive area")
    if not footprint.is_valid:
        raise ValueError(f"panel footprint is invalid: {footprint.is_valid}")

    outer = _ring_wire(footprint.exterior.coords)
    holes = [_ring_wire(interior.coords) for interior in footprint.interiors]
    return Face(outer, holes) if holes else Face(outer)


@lru_cache(maxsize=1)
def _socket_cutter() -> Solid:
    """Load and validate the exact 3-D native accessory cavity once."""

    if not SOCKET_CUTTER_ASSET.is_file():
        raise FileNotFoundError(f"native socket cutter asset is missing: {SOCKET_CUTTER_ASSET}")
    imported = import_step(SOCKET_CUTTER_ASSET)
    solids = imported.solids()
    if len(solids) != 1:
        raise RuntimeError("native socket cutter STEP must contain exactly one solid")
    cutter = solids[0]
    bounds = cutter.bounding_box()
    tolerance = 1e-5
    if (
        abs(bounds.min.X + 13.2) > tolerance
        or abs(bounds.max.X - 13.2) > tolerance
        or abs(bounds.min.Y + 13.2) > tolerance
        or abs(bounds.max.Y - 13.2) > tolerance
        or abs(bounds.min.Z) > tolerance
        or abs(bounds.max.Z - THICKNESS) > tolerance
    ):
        raise RuntimeError("native socket cutter has unexpected normalized bounds")
    if not cutter.is_valid:
        raise RuntimeError("native socket cutter is invalid")
    return cutter


@lru_cache(maxsize=1)
def _mounting_cutter() -> Solid:
    """Native screw profile reflected so its head recess opens at Z=4, not the bed."""
    shape = import_step(MOUNTING_CUTTER_ASSET)
    solids = shape.solids()
    if len(solids) != 1 or not shape.is_valid:
        raise RuntimeError("native mounting cutter must be one valid solid")
    bounds = shape.bounding_box()
    if any(
        abs(actual - expected) > 1e-6
        for actual, expected in (
            (bounds.min.X, -3.6),
            (bounds.max.X, 3.6),
            (bounds.min.Y, -3.6),
            (bounds.max.Y, 3.6),
            (bounds.min.Z, 0),
            (bounds.max.Z, THICKNESS),
        )
    ):
        raise RuntimeError("native mounting cutter has unexpected normalized bounds")
    # Reflect only the screw cutter; preserve all accessory socket orientation.
    return solids[0].mirror(Plane.XY).translate((0, 0, THICKNESS))


def _single_volume(shape: object, *, context: str):
    solids = shape.solids()  # type: ignore[attr-defined]
    if len(solids) != 1:
        raise RuntimeError(f"{context} must be one connected solid; got {len(solids)}")
    if not shape.is_valid:  # type: ignore[attr-defined]
        raise RuntimeError(f"{context} is not a valid CAD solid")
    if shape.volume <= 0:  # type: ignore[attr-defined]
        raise RuntimeError(f"{context} has no positive volume")
    return shape


def build_panel(panel: Panel):
    """Build one native Lite panel in installation coordinates.

    ``panel.footprint`` supplies the complete outline, including any adapted
    joint profiles.  Every listed cell receives the measured native 3-D
    accessory cavity cutter. Complete interior nodes also receive the exact
    native screw-hole cutter; hole selection avoids split/clipped seam holes.
    Screw choice and wood attachment engineering remain user-handled.
    """

    # Boolean operations commonly reverse Shapely ring winding. Extrusion must
    # use global +Z, not the face normal, so every socket cutter overlaps the slab.
    footprint = panel.base_footprint if panel.base_footprint is not None else panel.footprint
    base = extrude(_footprint_face(footprint), amount=THICKNESS, dir=(0, 0, 1))
    for joint in panel.joints:
        if joint.spec.style != "under_desk":
            continue
        from .underdesk import underdesk_solids

        features = underdesk_solids(joint.spec.clearance)
        role = "male" if joint.male_panel == panel.id else "female"
        addition = getattr(features, f"{role}_add")
        cut = getattr(features, f"{role}_cut")
        def posed(shape, joint=joint):
            return shape.rotate(Axis.Z, joint.angle).translate((joint.x, joint.y, 0))

        if addition is not None:
            base = base.fuse(posed(addition))
        if cut is not None:
            base = base.cut(posed(cut))
    if panel.edges:
        from .edges import lip_solids

        for edge in panel.edges:
            addition, cut = lip_solids(edge)
            base = base.fuse(addition) if edge.male_panel == panel.id else base.cut(cut)
    cutter = _socket_cutter()
    cutters = [cutter.moved(Location((cell.x, cell.y, 0.0))) for cell in panel.cells]
    result = base.cut(*cutters) if cutters else base
    removed = base.volume - result.volume
    if not isclose(removed, len(cutters) * cutter.volume, rel_tol=1e-8, abs_tol=1e-5):
        raise RuntimeError(f"panel {panel.id!r} did not preserve every complete native cavity")
    holes = mounting_holes(panel)
    if holes:
        mount = _mounting_cutter()
        mounted = result.cut(*[mount.moved(Location((x, y, 0))) for x, y in holes])
        if not isclose(
            result.volume - mounted.volume, len(holes) * mount.volume, rel_tol=1e-8, abs_tol=1e-5
        ):
            raise RuntimeError(f"panel {panel.id!r} has clipped or overlapping mounting holes")
        result = mounted
    validated = _single_volume(result, context=f"panel {panel.id!r}")
    bounds = validated.bounding_box()
    if abs(bounds.min.Z) > 1e-6 or abs(bounds.max.Z - THICKNESS) > 1e-6:
        raise RuntimeError(f"panel {panel.id!r} must have bottom Z=0 and top Z=4")
    return Part([validated.solids()[0]])


def printable_panel(panel: Panel):
    """Build a panel and apply rotation about global origin, then XY translation."""

    shape = build_panel(panel)
    rotation = float(panel.placement.rotation)
    if rotation:
        shape = shape.rotate(Axis.Z, rotation)
    if panel.placement.x or panel.placement.y:
        shape = shape.translate((float(panel.placement.x), float(panel.placement.y), 0.0))
    return _single_volume(shape, context=f"printable panel {panel.id!r}")
