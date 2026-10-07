"""Shallow supporting lips in the native socket edge web.

Local male stock is X<=0, receiver X>=0. Under-desk lips capture both signs
of Z; the node spring still stops X withdrawal. Wall ledges are bottom-open:
insert the male from below (+Z), or lower the receiver over it. They support
one Z sign only, not a standalone lock. Both print flat, with 45-degree ramps.
"""

from shapely import union_all
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring

from .models import EdgeInterface

CORNER_MARGIN = 1.25
NODE_MARGIN = 0.8
MIN_LENGTH = 2.0


def lip_sections(style, clearance):
    """Return local X/Z tongue and cutter polygons, never full-height outlines."""
    if style == "under_desk_puzzle":
        style = "wall"  # Mounting-side tongue, never the closed under-desk clip or a Z mirror.
    if style not in {"wall", "under_desk"} or not 0 <= clearance <= 0.1:
        raise ValueError("Supporting lips require wall/under_desk clearance in 0..0.1")
    reach = 0.45 if style == "under_desk" else 0.3 - clearance
    bottom, top = 0.1, 1.9
    tongue = Polygon([
        (-0.5, bottom), (0, bottom), (reach, bottom + reach),
        (reach, top - reach), (0, top), (-0.5, top),
    ])
    # Explicit 45-degree tool faces avoid the tiny horizontal roof created
    # by buffering the tongue's root corner. Clearance is perpendicular
    # to the bevel and horizontal at the nose, matching the mating surfaces.
    offset = 2**0.5 * clearance
    nose = reach + clearance
    if style == "wall":
        cutter = Polygon([(-0.05, -0.05), (nose, -0.05),
                          (nose, top - nose + offset), (-0.05, top + 0.05 + offset)])
    else:
        cutter = Polygon([(-0.05, bottom - 0.05 - offset),
                          (nose, bottom + nose - offset),
                          (nose, top - nose + offset), (-0.05, top + 0.05 + offset)])
    return tongue, cutter


def lip_solids(edge):
    """A tongue addition for the male, a pocket cutter for the receiver."""
    from build123d import Axis, Face, Wire, extrude

    tongue, cutter = lip_sections(edge.style, edge.clearance)

    def prism(section):
        face = Face(Wire.make_polygon(
            [(x, -edge.length / 2, z) for x, z in list(section.exterior.coords)[:-1]],
            close=True,
        ))
        return extrude(face, amount=edge.length, dir=(0, 1, 0)).rotate(
            Axis.Z, edge.angle,
        ).translate((edge.x, edge.y, 0))

    return prism(tongue), prism(cutter)


def _pose(shape, x, y, angle):
    return translate(rotate(shape, angle, origin=(0, 0)), xoff=x, yoff=y)


def edge_reservation(edge):
    tongue, cutter = lip_sections(edge.style, edge.clearance)
    return _pose(box(-0.5, -edge.length / 2, max(tongue.bounds[2], cutter.bounds[2]),
                     edge.length / 2), edge.x, edge.y, edge.angle)


def _lines(shape):
    if isinstance(shape, LineString):
        yield shape
    elif hasattr(shape, "geoms"):
        for part in shape.geoms:
            yield from _lines(part)


def add_edge_interfaces(seams, joints, spec, usable, place, warnings):
    """Fill safe remaining seam intervals, keeping nodes and corners accessible."""
    if spec.joints.clearance > 0.1:
        warnings.append("Supporting lips omitted: clearance exceeds their calibrated 0..0.1 mm range.")
        return
    reserved = union_all([j.reservation for j in joints if j.reservation is not None])
    occupied = reserved.buffer(NODE_MARGIN)
    edges = []
    for first, second, seam in seams:
        nodes = [j for j in joints if {j.male_panel, j.female_panel} == {first.id, second.id}
                 and seam.distance(Point(j.x, j.y)) < 1e-7]
        if not nodes:
            continue  # A lip alone does not replace the seam's locking connector.
        node = nodes[0]
        if spec.joints.style == "under_desk_puzzle":
            # Lip polarity is independent of the full-height puzzle key. Odd
            # global rows offer tongues to both even neighbours; within a row
            # every tongue points east. Fragmented horizontal seams stay bare.
            if first.grid_row is None or second.grid_row is None:
                continue
            if first.grid_row == second.grid_row and node.angle % 180 == 0:
                male, female = sorted((first, second), key=lambda p: min(c.ix for c in p.cells))
                angle = 0
            elif abs(first.grid_row - second.grid_row) == 1 and node.angle % 180 == 90:
                male, female = (first, second) if first.grid_row % 2 else (second, first)
                angle = 90 if female.grid_row > male.grid_row else 270
            else:
                warnings.append(f"Seam {first.id}/{second.id}: fragmented row lips omitted.")
                continue
        else:
            if any((j.male_panel, j.angle) != (node.male_panel, node.angle) for j in nodes):
                warnings.append(f"Seam {first.id}/{second.id}: lips omitted for inconsistent connector ownership.")
                continue
            male, female = (first, second) if first.id == node.male_panel else (second, first)
            angle = node.angle
        for p0, p1 in zip(seam.coords, list(seam.coords)[1:]):
            straight = LineString([p0, p1])
            if straight.length <= 2 * CORNER_MARGIN + MIN_LENGTH:
                continue
            free = substring(straight, CORNER_MARGIN, straight.length - CORNER_MARGIN).difference(occupied)
            for interval in _lines(free):
                if interval.length < MIN_LENGTH:
                    continue
                centre = interval.interpolate(0.5, normalized=True)
                edge = EdgeInterface(f"E{len(edges) + 1:03d}", male.id, female.id,
                                     centre.x, centre.y, angle, interval.length,
                                     spec.joints.style, spec.joints.clearance)
                region = edge_reservation(edge)
                stock_a = _pose(box(-0.5, -edge.length / 2, 0, edge.length / 2),
                                edge.x, edge.y, edge.angle)
                tongue, cutter = lip_sections(edge.style, edge.clearance)
                stock_b = _pose(box(0, -edge.length / 2, cutter.bounds[2], edge.length / 2),
                                edge.x, edge.y, edge.angle)
                a_base = male.base_footprint if male.base_footprint is not None else male.footprint
                b_base = female.base_footprint if female.base_footprint is not None else female.footprint
                if not usable.covers(region) or not a_base.covers(stock_a) or not b_base.covers(stock_b):
                    continue
                addition = _pose(box(-0.5, -edge.length / 2, tongue.bounds[2], edge.length / 2),
                                 edge.x, edge.y, edge.angle)
                outline = male.footprint.union(addition)
                placement = place(outline)
                if not isinstance(outline, Polygon) or placement is None:
                    continue
                allowed = union_all([reserved, region, *(edge_reservation(e) for e in edges)])
                if outline.intersection(female.footprint).difference(allowed).area > 1e-7:
                    continue
                for panel in (male, female):
                    if panel.base_footprint is None:
                        panel.base_footprint = panel.footprint
                    panel.edges += (edge,)
                male.footprint, male.placement = outline, placement
                edges.append(edge)
    if edges:
        capture = ("Puzzle support ledges capture one Z sign with mounting-side tongues. "
                   if spec.joints.style == "under_desk_puzzle" else
                   "Wall ledges capture one Z sign; under-desk lips capture both. ")
        warnings.append(f"{len(edges)} supporting edge lips added; native sockets retained. "
                        + capture + "Physical load and fatigue testing still required.")
