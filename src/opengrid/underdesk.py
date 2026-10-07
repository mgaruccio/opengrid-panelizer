"""Compact, thickness-captured connector in a native 4 mm Lite lattice node.

The seated male occupies X <= 0, the female X >= 0, both Z = 0..4. Opposite-
owner, bottom-wide rails stop both signs of Z motion. Their outer receiver
walls taper at 0.5 mm/mm between Z = 2..3.2 (vertical elsewhere), so Z play is
only twice the XY clearance. Their inner sides open into the
snap relief rather than leaving thin fins. Print both panels flat, Z = 0 on
the bed: no roofs, supports, raised islands, hardware or separate parts.

Assembly: align Y/Z, start the male 4.2 mm left of its seated position and slide
+X. Only the 1.0 mm XY spring bends, towards -Y; its 0.5 mm root fillet and
6.2 mm effective length limit the bending proxy despite the deeper 0.5 mm tooth.
A 0.05 mm tooth-corner round leaves 0.45 mm of straight retaining face (0.35 mm
engaged at maximum clearance); the 0.6 mm plateau seats in an unloaded recess.
Seam faces stop insertion and the flat tooth shoulder stops withdrawal.
Release: push the exposed tooth at (2.3, 1.0) towards -Y by 0.55 mm, through
either open Z face with a fingernail, then withdraw 4.2 mm along -X.

Apply each side as slab.fuse(add).cut(cut), ignoring None tools. Do NOT extrude
the plan as the finished connector: adds are XY projections, cuts ONLY the
through-slots around the spring; capture pockets are exclusively 3-D cutters.
Reserve the diamond-shaped reservation as solid stock, with no mounting hole.
It lies entirely between the four native sockets centred at (+/-14, +/-14);
no accessory socket needs omitting. Rotate/translate both APIs together in XY;
never mirror or flip Z.

Clearance is finite, per-side horizontal running clearance, 0..0.10 mm; the
default is 0.05. Zero is a nominal-contact calibration specimen, not guaranteed
sliding fit. Vertical play is clearance / 0.5 in either direction. Print the
0/0.05/0.10 ladder first: extrusion, elephant foot, material and fatigue matter.
The spring's bounded displacement checks are kinematic, not a load rating or
a material/strain certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from numbers import Real
from typing import TYPE_CHECKING

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

if TYPE_CHECKING:
    from build123d import Shape


THICKNESS = 4.0
MIN_CLEARANCE = 0.0
MAX_CLEARANCE = 0.1
ASSEMBLY_TRAVEL = 4.2
TONGUE_DEPTH = 1.5
TONGUE_ROOT = 0.8
TONGUE_CENTRES = (-2.0, 2.0)
TONGUE_BOTTOM_HALF_WIDTH = 1.0
TONGUE_TOP_HALF_WIDTH = 0.4
# Concentrating the taper keeps the rail robust without magnifying Z play.
# The 0.8 mm narrow top has four full 0.2 mm layers; no horizontal roof.
TONGUE_TAPER_BOTTOM_Z = 2.0
TONGUE_TAPER_TOP_Z = 3.2
LATCH_ROOT_X = -4.4
LATCH_ROOT_RADIUS = 0.5
LATCH_HALF_WIDTH = 0.5
LATCH_SHOULDER_X = 2.3
LATCH_TOOTH_RISE = 0.5
LATCH_SHOULDER_Y = LATCH_HALF_WIDTH + LATCH_TOOTH_RISE
LATCH_TOOTH_RADIUS = 0.05
LATCH_TOOTH_PLATEAU = 0.6
LATCH_RAMP_START_X = LATCH_SHOULDER_X + LATCH_TOOTH_RADIUS + LATCH_TOOTH_PLATEAU
LATCH_TIP_X = 3.7
LATCH_RELEASE_TRAVEL = 0.55
LATCH_EFFECTIVE_LENGTH = LATCH_SHOULDER_X - (LATCH_ROOT_X + LATCH_ROOT_RADIUS)
# The longer spring sweeps 0.26 mm at the seam during release.
LATCH_ROOT_RELIEF = 0.30
# The envelope includes the free tip's larger displacement, not only the tooth travel.
FLEX_ENVELOPE = box(LATCH_ROOT_X, -1.35, LATCH_TIP_X, 1.15)
RESERVATION_RADIUS = 5.85
RESERVATION_BOUNDS = (-RESERVATION_RADIUS,) * 2 + (RESERVATION_RADIUS,) * 2
_TOOL_OVERRUN = 0.05
# Open each receiver's inner side into the spring relief at EVERY height.
# Otherwise the bottom-wide pocket would leave sub-nozzle internal fins.
_INNER_RELIEF = 1.7


@dataclass(frozen=True)
class UnderdeskPlan:
    """CAD-independent projections and mandatory solid-stock reservation."""

    male_add: BaseGeometry
    male_cut: BaseGeometry
    female_add: BaseGeometry
    female_cut: BaseGeometry
    reservation: Polygon


@dataclass(frozen=True)
class UnderdeskSolids:
    """Additions and cutters; multi-solid tools are intentional, not panels."""

    male_add: Shape | None
    male_cut: Shape | None
    female_add: Shape | None
    female_cut: Shape | None


@dataclass(frozen=True)
class _Rail:
    x0: float
    x1: float
    y: float
    clearance: float = 0.0
    overrun: float = 0.0

    def half_width(self, z: float) -> float:
        band = TONGUE_TAPER_TOP_Z - TONGUE_TAPER_BOTTOM_Z
        fraction = min(1.0, max(0.0, (z - TONGUE_TAPER_BOTTOM_Z) / band))
        return (
            TONGUE_BOTTOM_HALF_WIDTH
            - fraction * (TONGUE_BOTTOM_HALF_WIDTH - TONGUE_TOP_HALF_WIDTH)
            + self.clearance
        )

    @property
    def projection(self) -> Polygon:
        half_width = self.half_width(-self.overrun)
        return box(self.x0, self.y - half_width, self.x1, self.y + half_width)

    @property
    def section(self) -> tuple[tuple[float, float, float], ...]:
        heights = (
            -self.overrun,
            TONGUE_TAPER_BOTTOM_Z,
            TONGUE_TAPER_TOP_Z,
            THICKNESS + self.overrun,
        )
        return tuple(
            (self.x0, self.y + sign * self.half_width(z), z)
            for sign, levels in ((1, heights), (-1, reversed(heights)))
            for z in levels
        )


@dataclass(frozen=True)
class _Profiles:
    male_rail: _Rail
    female_rail: _Rail
    male_pocket: _Rail
    female_pocket: _Rail
    latch: Polygon
    male_slots: BaseGeometry
    female_slot: Polygon


def _latch_profile() -> Polygon:
    """A one-mm XY spring with a relieved root and lightly rounded tooth corner.

    Root arcs use 16 chords per quadrant (less than 0.0005 mm sagitta); the same
    polygon supplies both CAD and planning, including those root fillets.  The
    small outer-corner round retains a straight catch face even at maximum
    clearance, rather than making withdrawal bear only on a camming arc.
    The longer beam offsets the deeper tooth's release travel without exceeding
    the previous one-mm spring's bending proxy.
    """
    radius = LATCH_ROOT_RADIUS
    root_end = LATCH_ROOT_X + radius
    root_half_width = LATCH_HALF_WIDTH + radius
    upper = [
        (
            root_end + radius * cos(pi + i * pi / 32),
            root_half_width + radius * sin(pi + i * pi / 32),
        )
        for i in range(17)
    ]
    tooth_entry = [(LATCH_SHOULDER_X, LATCH_HALF_WIDTH)]
    tooth_entry += [
        (
            LATCH_SHOULDER_X + LATCH_TOOTH_RADIUS + LATCH_TOOTH_RADIUS * cos(pi - i * pi / 24),
            LATCH_SHOULDER_Y - LATCH_TOOTH_RADIUS + LATCH_TOOTH_RADIUS * sin(pi - i * pi / 24),
        )
        for i in range(13)
    ]
    lower = [(x, -y) for x, y in reversed(upper)]
    return Polygon(
        upper
        + tooth_entry
        + [
            (LATCH_RAMP_START_X, LATCH_SHOULDER_Y),
            (LATCH_TIP_X, LATCH_HALF_WIDTH),
            (LATCH_TIP_X, -LATCH_HALF_WIDTH),
        ]
        + lower
    )


def _profiles(clearance: float) -> _Profiles:
    if (
        isinstance(clearance, bool)
        or not isinstance(clearance, Real)
        or not isfinite(clearance)
        or not MIN_CLEARANCE <= clearance <= MAX_CLEARANCE
    ):
        raise ValueError("underdesk clearance must be finite and between 0 and 0.10 mm")
    clearance = float(clearance)
    latch = _latch_profile()
    relief_half_width = LATCH_HALF_WIDTH + LATCH_ROOT_RADIUS + LATCH_ROOT_RELIEF
    root_guard_half_width = LATCH_HALF_WIDTH + LATCH_ROOT_RADIUS + 0.2
    rounded_root_relief = unary_union(
        [
            latch.buffer(LATCH_ROOT_RELIEF, quad_segs=16, join_style=1).intersection(
                box(LATCH_ROOT_X, -relief_half_width, 0, relief_half_width)
            ),
            # Keep a continuous fixed-root land; the relief rounds away from this
            # guard rather than making a thin stock notch.
            box(
                LATCH_ROOT_X,
                -root_guard_half_width,
                LATCH_ROOT_X + LATCH_ROOT_RADIUS,
                root_guard_half_width,
            ),
        ]
    )
    male_relief = unary_union(
        [
            # Round the spring-side relief instead of leaving a sharp stock notch;
            # clipping at the reservation edge keeps the native diagonal land intact.
            rounded_root_relief,
            box(-TONGUE_DEPTH - clearance, 0, 0, _INNER_RELIEF),
        ]
    )
    male_slots = male_relief.difference(latch)
    slot_top = LATCH_SHOULDER_Y + 0.5
    female_slot = Polygon(
        [
            (-_TOOL_OVERRUN, -_INNER_RELIEF),
            # Only the rail pocket needs the deeper inner relief. Beyond it,
            # retain receiver stock below the entire bent-tip envelope.
            (TONGUE_DEPTH + clearance, -_INNER_RELIEF),
            (TONGUE_DEPTH + clearance, FLEX_ENVELOPE.bounds[1]),
            (LATCH_TIP_X + clearance, FLEX_ENVELOPE.bounds[1]),
            (LATCH_TIP_X + clearance, slot_top),
            (LATCH_SHOULDER_X - clearance, slot_top),
            (LATCH_SHOULDER_X - clearance, LATCH_HALF_WIDTH + clearance),
            (-_TOOL_OVERRUN, LATCH_HALF_WIDTH + clearance),
        ]
    )
    return _Profiles(
        male_rail=_Rail(-TONGUE_ROOT, TONGUE_DEPTH, TONGUE_CENTRES[0]),
        female_rail=_Rail(-TONGUE_DEPTH, TONGUE_ROOT, TONGUE_CENTRES[1]),
        male_pocket=_Rail(
            -TONGUE_DEPTH - clearance,
            _TOOL_OVERRUN,
            TONGUE_CENTRES[1],
            clearance,
            _TOOL_OVERRUN,
        ),
        female_pocket=_Rail(
            -_TOOL_OVERRUN,
            TONGUE_DEPTH + clearance,
            TONGUE_CENTRES[0],
            clearance,
            _TOOL_OVERRUN,
        ),
        latch=latch,
        male_slots=male_slots,
        female_slot=female_slot,
    )


def underdesk_plan(clearance: float = 0.05) -> UnderdeskPlan:
    """Return local projections without importing build123d or OpenCascade."""
    profiles = _profiles(clearance)
    radius = RESERVATION_RADIUS
    return UnderdeskPlan(
        male_add=unary_union([profiles.male_rail.projection, profiles.latch]),
        male_cut=profiles.male_slots,
        female_add=profiles.female_rail.projection,
        female_cut=profiles.female_slot,
        reservation=Polygon([(-radius, 0), (0, -radius), (radius, 0), (0, radius)]),
    )


def underdesk_solids(clearance: float = 0.05) -> UnderdeskSolids:
    """Build the Z-varying rails and receivers; CAD imports are deliberately lazy."""
    profiles = _profiles(clearance)
    from build123d import Compound, Face, ShapeList, Solid, Wire

    def rail(profile: _Rail) -> Solid:
        return Solid.extrude(
            Face(Wire.make_polygon(profile.section)), (profile.x1 - profile.x0, 0, 0)
        )

    def prisms(footprint: BaseGeometry, z: float, height: float) -> list[Solid]:
        polygons = [footprint] if isinstance(footprint, Polygon) else footprint.geoms
        result = []
        for polygon in polygons:
            outer = Wire.make_polygon([(x, y, z) for x, y in polygon.exterior.coords])
            holes = [
                Wire.make_polygon([(x, y, z) for x, y in ring.coords]) for ring in polygon.interiors
            ]
            result.append(Solid.extrude(Face(outer, holes), (0, 0, height)))
        return result

    def cutter(pocket: _Rail, slots: BaseGeometry) -> Shape:
        # The reliefs intentionally intersect the capture pockets. Boolean-union
        # them first: overlapping solids in a Compound are not a valid cut tool.
        union = rail(pocket).fuse(*prisms(slots, -_TOOL_OVERRUN, THICKNESS + 2 * _TOOL_OVERRUN))
        return Compound(children=union) if isinstance(union, ShapeList) else union

    return UnderdeskSolids(
        male_add=Compound(
            children=[rail(profiles.male_rail), *prisms(profiles.latch, 0, THICKNESS)]
        ),
        male_cut=cutter(profiles.male_pocket, profiles.male_slots),
        female_add=rail(profiles.female_rail),
        female_cut=cutter(profiles.female_pocket, profiles.female_slot),
    )
