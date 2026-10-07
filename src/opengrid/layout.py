"""Deterministic planar panelization on one native 28 mm lattice.

The grid origin is a cell corner, not a cell centre. Only whole lattice squares
qualify for sockets; irregular edges are retained as solid border/filler. Packing
and seam selection are bounded heuristics, not claims of global optimality.
"""

import math
from collections.abc import Callable, Iterator

from shapely import union_all
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep
from shapely.strtree import STRtree

from .joints import add_joints, puzzle_profiles, single_material
from .models import PITCH, Cell, InstallationSpec, Layout, Panel, PrinterSpec, PrintPlacement
from .spec import ensure_ready

# Bound work on accidental enormous coordinate envelopes. The actual 72 x 30 in
# installation has fewer than 2,000 lattice candidates.
_MAX_CELL_CANDIDATES = 100_000
_MAX_PLACEMENT_COORDINATES = 64


class LayoutError(ValueError):
    """No safe, useful printable layout could be certified."""


def transformed_print_footprint(footprint: BaseGeometry, placement: PrintPlacement) -> BaseGeometry:
    """Apply the same global-origin rotation then translation used for CAD export."""
    return translate(
        rotate(footprint, placement.rotation, origin=(0, 0)), xoff=placement.x, yoff=placement.y
    )


def _polygons(geometry: BaseGeometry) -> Iterator[Polygon]:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygons(part)


def _coordinates(geometry: BaseGeometry, axis: int) -> set[float]:
    return {
        coord[axis]
        for polygon in _polygons(geometry)
        for ring in (polygon.exterior, *polygon.interiors)
        for coord in ring.coords
    }


def _translations(
    bed: BaseGeometry, footprint: BaseGeometry, axis: int, low: float, high: float
) -> list[float]:
    middle = (low + high) / 2
    values = {low, middle, high}
    for b in _coordinates(bed, axis):
        for p in _coordinates(footprint, axis):
            value = b - p
            if low <= value <= high:
                values.add(value)
    values = sorted(values)
    if len(values) > _MAX_PLACEMENT_COORDINATES:
        # Even sampling keeps the bounded search spread over the whole interval.
        last = len(values) - 1
        values = [
            values[round(i * last / (_MAX_PLACEMENT_COORDINATES - 1))]
            for i in range(_MAX_PLACEMENT_COORDINATES)
        ]
    return sorted(values, key=lambda v: (abs(v - middle), v))


def _placement_offsets(
    bed: BaseGeometry,
    footprint: BaseGeometry,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Iterator[tuple[float, float]]:
    yield from (
        (xmin, ymin),
        (xmax, ymin),
        (xmin, ymax),
        (xmax, ymax),
        ((xmin + xmax) / 2, (ymin + ymax) / 2),
    )
    xs = _translations(bed, footprint, 0, xmin, xmax)
    ys = _translations(bed, footprint, 1, ymin, ymax)
    for x in xs:
        for y in ys:
            yield x, y


def find_print_placement(footprint: BaseGeometry, printer: PrinterSpec) -> PrintPlacement | None:
    """Certify placement against the actual margin/exclusion-subtracted bed.

    Try four quarter turns, bounding-box corners/centre, then at most 64 x 64
    feature-aligned translations per orientation. None means this bounded search
    found no placement, not a mathematical impossibility proof for arbitrary beds.
    Every success is checked using the complete footprint, never its bounds alone.
    """
    if not isinstance(footprint, Polygon) or footprint.is_empty or not footprint.is_valid:
        return None
    bed = printer.usable_bed
    if bed.is_empty or not bed.is_valid:
        return None
    bx0, by0, bx1, by1 = bed.bounds
    prepared = prep(bed)
    for angle in (0, 90, 180, 270):
        turned = rotate(footprint, angle, origin=(0, 0))
        x0, y0, x1, y1 = turned.bounds
        xmin, xmax = bx0 - x0, bx1 - x1
        ymin, ymax = by0 - y0, by1 - y1
        if xmin > xmax or ymin > ymax:
            continue
        tried = set()
        for x, y in _placement_offsets(bed, turned, xmin, xmax, ymin, ymax):
            if (x, y) in tried:
                continue
            tried.add((x, y))
            if prepared.covers(translate(turned, xoff=x, yoff=y)):
                return PrintPlacement(angle, x, y)
    return None


def _cell_square(cell: Cell, origin: tuple[float, float]) -> Polygon:
    # Build edges from lattice indices, just like partition cuts. Reconstructing
    # edges by centre +/- 14 can lose full edge rows at decimal origins to ulps.
    ox, oy = origin
    return box(
        ox + cell.ix * PITCH,
        oy + cell.iy * PITCH,
        ox + (cell.ix + 1) * PITCH,
        oy + (cell.iy + 1) * PITCH,
    )


def plan_installation(
    spec: InstallationSpec,
    *,
    socket_projection: Callable[[float, float], BaseGeometry] | None = None,
) -> Layout:
    """Compile a ready installation to safe planar panels and provisional joints.

    The injected projection is a testing/geometry-provider seam. Production lazily
    imports the native CAD provider, whose negative includes all socket chamfers.
    Draft or unresolved installations are rejected *before* invoking that provider.
    """
    ensure_ready(spec)
    if spec.joints.style in {"puzzle", "wall", "under_desk_puzzle"}:
        try:
            puzzle_profiles(spec.joints)
        except ValueError as error:
            raise LayoutError(str(error)) from error
    if socket_projection is None:
        from .native import socket_projection

    keepouts = union_all([keepout.expanded for keepout in spec.keepouts])
    usable = spec.surface.difference(keepouts)
    if usable.is_empty or usable.area <= 0:
        raise LayoutError("Keepouts leave no usable surface")
    ox, oy = spec.grid_origin
    x0, y0, x1, y1 = usable.bounds
    ix0, iy0 = math.floor((x0 - ox) / PITCH), math.floor((y0 - oy) / PITCH)
    ix1, iy1 = math.ceil((x1 - ox) / PITCH), math.ceil((y1 - oy) / PITCH)
    if (ix1 - ix0) * (iy1 - iy0) > _MAX_CELL_CANDIDATES:
        raise LayoutError(f"Surface envelope exceeds {_MAX_CELL_CANDIDATES} lattice candidates")
    chunk = min(math.floor(spec.printer.max_panel_span / PITCH), max(ix1 - ix0, iy1 - iy0))
    if chunk < 1:
        raise LayoutError("printer.max_panel_span cannot hold one native 28 mm cell")

    cells, projections = [], {}
    prepared = prep(usable)
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            cell = Cell(ix, iy, ox + (ix + 0.5) * PITCH, oy + (iy + 0.5) * PITCH)
            square = _cell_square(cell, spec.grid_origin)
            if not prepared.covers(square):
                continue
            negative = socket_projection(cell.x, cell.y)
            if (
                not isinstance(negative, (Polygon, MultiPolygon))
                or negative.is_empty
                or not negative.is_valid
                or negative.has_z
                or not square.contains_properly(negative)
            ):
                raise LayoutError(
                    f"Socket projection at cell ({ix}, {iy}) must be a valid negative "
                    "strictly inside its full native 28 mm cell"
                )
            cells.append(cell)
            projections[ix, iy] = negative
    if not cells:
        raise LayoutError("Usable surface contains no complete native 28 mm cells")
    slots = union_all(list(projections.values()))
    warnings = []
    pieces = []

    def place(footprint: BaseGeometry) -> PrintPlacement | None:
        return find_print_placement(footprint, spec.printer)

    def partition(
        region: BaseGeometry,
        candidates: list[Cell],
        left: int,
        bottom: int,
        right: int,
        top: int,
    ) -> None:
        for polygon in sorted(_polygons(region), key=lambda g: (g.bounds, g.wkb_hex)):
            retained = [
                cell for cell in candidates if polygon.covers(_cell_square(cell, spec.grid_origin))
            ]
            if not retained:
                warnings.append(
                    f"Omitted cell-less border/scrap at {polygon.bounds} "
                    f"({polygon.area:.3f} mm²); no functioning native cells."
                )
                continue
            if not single_material(polygon, slots):
                raise LayoutError(f"Projected sockets disconnect material at {polygon.bounds}")
            placement = place(polygon)
            if placement is not None:
                pieces.append((polygon, tuple(retained), placement))
                continue
            width, height = right - left, top - bottom
            if width == height == 1:
                raise LayoutError(
                    f"No usable-bed placement found for native cell "
                    f"({retained[0].ix}, {retained[0].iy}); check printer margin/exclusions."
                )
            # Retry only on lattice lines; the integer range halves at every step.
            if width > 1 and (width >= height or height == 1):
                middle = (left + right) // 2
                bounds = ((left, bottom, middle, top), (middle, bottom, right, top))
            else:
                middle = (bottom + top) // 2
                bounds = ((left, bottom, right, middle), (left, middle, right, top))
            for l, b, r, t in bounds:
                clip = box(ox + l * PITCH, oy + b * PITCH, ox + r * PITCH, oy + t * PITCH)
                subset = [c for c in retained if l <= c.ix < r and b <= c.iy < t]
                partition(polygon.intersection(clip), subset, l, b, r, t)

    # Keep all cuts phased to the same global lattice, even for negative indices
    # and disconnected usable areas. Solid border areas may be omitted, never used
    # as a reason to invent partial cells.
    groups = {}
    stagger = chunk // 2 if spec.joints.style in {"wall", "under_desk_puzzle"} else 0
    for cell in cells:
        gy = cell.iy // chunk
        shift = (gy % 2) * stagger
        groups.setdefault(((cell.ix - shift) // chunk, gy), []).append(cell)
    for gy in range(iy0 // chunk, (iy1 - 1) // chunk + 1):
        shift = (gy % 2) * stagger
        for gx in range((ix0 - shift) // chunk, (ix1 - 1 - shift) // chunk + 1):
            l, b = gx * chunk + shift, gy * chunk
            r, t = l + chunk, b + chunk
            clip = box(ox + l * PITCH, oy + b * PITCH, ox + r * PITCH, oy + t * PITCH)
            partition(usable.intersection(clip), groups.get((gx, gy), []), l, b, r, t)
    pieces.sort(
        key=lambda part: (
            part[0].bounds[1],
            part[0].bounds[0],
            part[0].bounds,
            part[0].normalize().wkb_hex,
        )
    )
    panels = [
        Panel(f"P{i:03d}", polygon, panel_cells, placement, board=spec.board,
              grid_row=panel_cells[0].iy // chunk if spec.joints.style == "under_desk_puzzle" else None)
        for i, (polygon, panel_cells, placement) in enumerate(pieces, 1)
    ]
    assigned = [(cell.ix, cell.iy) for panel in panels for cell in panel.cells]
    if len(assigned) != len(cells) or set(assigned) != set(projections):
        raise LayoutError("Partitioning could not preserve every complete native cell exactly once")

    joints, limitations = add_joints(
        panels, spec, usable, slots, place, socket_projection=socket_projection
    )
    warnings.extend(limitations)
    # Structural connectors may deliberately reserve cells as solid land.
    slots = union_all([projections[c.ix, c.iy] for p in panels for c in p.cells])
    for panel in panels:
        if not usable.covers(panel.footprint) or not single_material(panel.footprint, slots):
            raise LayoutError(f"Unsafe final material in {panel.id}")
        if not spec.printer.usable_bed.covers(
            transformed_print_footprint(panel.footprint, panel.placement)
        ):
            raise LayoutError(f"Final joint bounds do not fit the usable bed for {panel.id}")
        for cell in panel.cells:
            if not panel.footprint.contains_properly(projections[cell.ix, cell.iy]):
                raise LayoutError(f"Final panel {panel.id} clips a native socket")
        # A conservative warning, not an invented strength/load certification.
        # 0.8 mm is two nominal 0.4 mm nozzle lines on the specified P1S setup.
        inset = panel.footprint.difference(slots).buffer(-0.4)
        if inset.is_empty or isinstance(inset, MultiPolygon):
            warnings.append(
                f"{panel.id}: conservative 0.8 mm projected-ligament check "
                "flags narrow land/ligament near the screening limit; inspect before printing. "
                "This is not a measured strength result."
            )
    combined = union_all([panel.footprint for panel in panels])
    # Captured 3-D lips overlap in XY projection, but must not intersect in 3-D.
    # Export verifies the actual assembled solids; only registered stock permits XY overlap.
    from .edges import edge_reservation

    allowed = union_all([
        *[j.reservation for j in joints if j.spec.style == "under_desk"],
        *[edge_reservation(e) for panel in panels for e in panel.edges],
    ])
    tree = STRtree([p.footprint for p in panels])
    for i, panel in enumerate(panels):
        for k in tree.query(panel.footprint, predicate="intersects"):
            if k > i and panel.footprint.intersection(panels[k].footprint).difference(allowed).area > 1e-7:
                raise LayoutError("Final panel footprints overlap outside captured connectors")
    return Layout(spec, usable, keepouts, panels, joints, usable.difference(combined), warnings)
