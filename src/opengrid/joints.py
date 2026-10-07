"""Planar puzzle/wall joints and reserved stock for 3-D under-desk connectors.

Profiles and placements are in installation coordinates. The male profile includes
an anchoring root; the female is its clearance-expanded cutting tool. Final panel
footprints already include these booleans. Physical fit still needs a coupon.
"""

import math
from collections.abc import Callable, Iterator

from shapely import line_merge, union_all
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .models import PITCH, InstallationSpec, Joint, JointSpec, Panel, PrintPlacement


def _pose(shape: BaseGeometry, x: float, y: float, angle: int) -> BaseGeometry:
    return translate(rotate(shape, angle, origin=(0, 0)), xoff=x, yoff=y)


def puzzle_profiles(
    spec: JointSpec, x: float = 0, y: float = 0, angle: int = 0
) -> tuple[Polygon, Polygon]:
    """Return an anchored key and its clearance tool; +X points into the receiver.

    The wider head lies beyond the narrow neck, giving in-plane retention rather
    than merely making a tapered registration peg. This is a custom profile.
    """
    values = (spec.depth, spec.neck_width, spec.head_width, spec.clearance)
    if not all(math.isfinite(v) for v in values) or not (
        spec.depth > 0
        and 0 < spec.neck_width < spec.head_width - 2 * spec.clearance
        and 0 <= spec.clearance < min(spec.depth, spec.neck_width) / 2
    ):
        raise ValueError("Invalid puzzle depth, neck, head or clearance")
    if angle not in (0, 90, 180, 270):
        raise ValueError("Puzzle normal must be 0, 90, 180 or 270 degrees")
    d, n, h = spec.depth, spec.neck_width / 2, spec.head_width / 2
    root = min(spec.depth, spec.neck_width) / 2
    wall_puzzle = spec.style in {"wall", "under_desk_puzzle"}
    shoulder = 0.45 if wall_puzzle else 0.65
    male = Polygon(
        [
            (-root, -n),
            (0.35 * d, -n),
            (shoulder * d, -h),
            (d, -h),
            (d, h),
            (shoulder * d, h),
            (0.35 * d, n),
            (-root, n),
        ]
    )
    if wall_puzzle and spec.fillet_radius:
        r = spec.fillet_radius
        male = male.buffer(-r).buffer(2 * r).buffer(-r)
    female = male.buffer(
        spec.clearance, join_style="round" if wall_puzzle else "mitre"
    )
    return _pose(male, x, y, angle), _pose(female, x, y, angle)


def single_material(footprint: BaseGeometry, slots: BaseGeometry) -> bool:
    """Conservative connectivity check even at the widest projected socket cut."""
    if not isinstance(footprint, Polygon) or footprint.is_empty or not footprint.is_valid:
        return False
    material = footprint.difference(slots)
    return isinstance(material, Polygon) and material.is_valid and material.area > 0


def _lines(shape: BaseGeometry) -> Iterator[LineString]:
    if isinstance(shape, LineString):
        if shape.length > 0:
            yield shape
    elif hasattr(shape, "geoms"):
        for part in shape.geoms:
            yield from _lines(part)


def _candidates(seam: LineString, origin: tuple[float, float]) -> list[tuple[float, float, int]]:
    """Use lattice crossings, where neighbouring sockets leave solid corner land."""
    ox, oy = origin
    candidates = set()
    for (x1, y1), (x2, y2) in zip(seam.coords, list(seam.coords)[1:]):
        if abs(x1 - x2) < 1e-8:
            x = ox + round((x1 - ox) / PITCH) * PITCH
            if abs(x - x1) > 1e-8:
                continue
            for iy in range(
                math.ceil((min(y1, y2) - oy) / PITCH), math.floor((max(y1, y2) - oy) / PITCH) + 1
            ):
                candidates.add((x, oy + iy * PITCH, 0))
        elif abs(y1 - y2) < 1e-8:
            y = oy + round((y1 - oy) / PITCH) * PITCH
            if abs(y - y1) > 1e-8:
                continue
            for ix in range(
                math.ceil((min(x1, x2) - ox) / PITCH), math.floor((max(x1, x2) - ox) / PITCH) + 1
            ):
                candidates.add((ox + ix * PITCH, y, 90))
    middle = seam.interpolate(0.5, normalized=True)
    return sorted(candidates, key=lambda p: (Point(p[:2]).distance(middle), p))


def _try_joint(
    male_panel: Panel,
    female_panel: Panel,
    spec: JointSpec,
    slots: BaseGeometry,
    usable: BaseGeometry,
    place: Callable[[BaseGeometry], PrintPlacement | None],
    x: float,
    y: float,
    angle: int,
) -> tuple[BaseGeometry, BaseGeometry] | None:
    male, female = puzzle_profiles(spec, x, y, angle)
    root_depth = min(spec.depth, spec.neck_width) / 2
    root = _pose(box(-root_depth, -spec.neck_width / 2, 0, spec.neck_width / 2), x, y, angle)
    # Requiring the entire root and protrusion prevents a reversed peg, a detached
    # head, or a key attached only by a point from being mistaken for an interlock.
    if not male_panel.footprint.covers(root):
        return None
    if not female_panel.footprint.covers(male.difference(root)):
        return None
    pair = male_panel.footprint.union(female_panel.footprint)
    if not pair.covers(female) or not usable.covers(female) or female.intersects(slots):
        return None
    new_male = male_panel.footprint.union(male)
    new_female = female_panel.footprint.difference(female)
    if not single_material(new_male, slots) or not single_material(new_female, slots):
        return None
    if new_male.intersection(new_female).area > 0:
        return None
    male_placement, female_placement = place(new_male), place(new_female)
    if male_placement is None or female_placement is None:
        return None
    male_panel.footprint, male_panel.placement = new_male, male_placement
    female_panel.footprint, female_panel.placement = new_female, female_placement
    return male, female


def add_joints(
    panels: list[Panel],
    spec: InstallationSpec,
    usable: BaseGeometry,
    slots: BaseGeometry,
    place: Callable[[BaseGeometry], PrintPlacement | None],
    *,
    socket_projection=None,
) -> tuple[list[Joint], list[str]]:
    """Modify footprints only when a whole socket-safe, printable joint is proven.

    At most two keys per connected shared seam. A failed seam is explicitly
    reported, never advertised as interlocking. Bed checks use the final shapes,
    including previous keys; geometry is not committed until both panels fit.
    """
    if spec.joints.style == "none":
        return [], []
    if spec.joints.style in {"wall", "under_desk", "under_desk_puzzle"}:
        return _add_structural_joints(panels, spec, usable, slots, place, socket_projection)
    seams = []
    tree = STRtree([panel.footprint for panel in panels])
    for i, first in enumerate(panels):
        for j in sorted(
            int(j) for j in tree.query(first.footprint, predicate="intersects") if j > i
        ):
            second = panels[j]
            shared = first.footprint.boundary.intersection(second.footprint.boundary)
            merged = line_merge(union_all(list(_lines(shared))))
            for line in sorted(_lines(merged), key=lambda g: (g.bounds, g.wkb_hex)):
                seams.append((first, second, line))

    joints, warnings = [], []
    for first, second, seam in seams:
        count = 0
        for x, y, normal in _candidates(seam, spec.grid_origin):
            accepted = False
            for angle in (normal, (normal + 180) % 360):
                for male_panel, female_panel in ((first, second), (second, first)):
                    profiles = _try_joint(
                        male_panel, female_panel, spec.joints, slots, usable, place, x, y, angle
                    )
                    if profiles is None:
                        continue
                    joints.append(
                        Joint(
                            f"J{len(joints) + 1:03d}",
                            male_panel.id,
                            female_panel.id,
                            x,
                            y,
                            angle,
                            *profiles,
                        )
                    )
                    count += 1
                    accepted = True
                    break
                if accepted:
                    break
            if count == 2:
                break
        if count == 0:
            warnings.append(
                f"Seam {first.id}/{second.id} at {seam.bounds}: no safe puzzle joint; "
                "insufficient solid land, socket clearance, connectivity or usable-bed space."
            )
    return joints, warnings


def _add_structural_joints(panels, spec, usable, slots, place, projection_provider=None):
    """Use intact grid-corner land; never remove a socket to accommodate a joint."""
    if spec.joints.style == "under_desk":
        from .underdesk import underdesk_plan

        plan = underdesk_plan(spec.joints.clearance)
        a_add, a_cut = plan.male_add, plan.male_cut
        b_add, b_cut = plan.female_add, plan.female_cut
        reservation = plan.reservation
    else:
        a_add, b_cut = puzzle_profiles(spec.joints)
        a_cut = b_add = Polygon()
        # One millimetre of stock beyond the connector/cutting tool, not a strength rating.
        reservation = a_add.union(b_cut).buffer(1.0)

    seams = []
    tree = STRtree([p.footprint for p in panels])
    for i, first in enumerate(panels):
        for j in sorted(int(j) for j in tree.query(first.footprint, predicate="intersects") if j > i):
            second = panels[j]
            shared = first.footprint.boundary.intersection(second.footprint.boundary)
            for seam in _lines(line_merge(union_all(list(_lines(shared))))):
                seams.append((first, second, seam))

    joints, warnings = [], []
    stock = []
    for first, second, seam in seams:
        count = 0
        for x, y, normal in _candidates(seam, spec.grid_origin):
            accepted = False
            for angle in (normal, (normal + 180) % 360):
                posed = [_pose(g, x, y, angle) for g in (a_add, a_cut, b_add, b_cut, reservation)]
                aa, ac, ba, bc, reserved = posed
                if (
                    not usable.covers(reserved)
                    or reserved.intersects(slots)
                    or any(reserved.intersects(g) for g in stock)
                ):
                    continue
                x0, y0, x1, y1 = reservation.bounds
                left = _pose(reservation.intersection(box(x0, y0, 0, y1)), x, y, angle)
                right = _pose(reservation.intersection(box(0, y0, x1, y1)), x, y, angle)
                for male, female in ((first, second), (second, first)):
                    if not male.footprint.covers(left) or not female.footprint.covers(right):
                        continue
                    ma = male.footprint.union(aa).difference(ac)
                    fe = female.footprint.union(ba).difference(bc)
                    if not all(single_material(g, slots) for g in (ma, fe)):
                        continue
                    if not all(usable.covers(g) for g in (ma, fe)):
                        continue
                    allowed_overlap = union_all([*stock, reserved])
                    if ma.intersection(fe).difference(allowed_overlap).area > 1e-7:
                        continue
                    placements = [place(g) for g in (ma, fe)]
                    if any(p is None for p in placements):
                        continue
                    joint = Joint(
                        f"J{len(joints) + 1:03d}", male.id, female.id, x, y, angle,
                        aa, bc, spec.joints, reserved,
                    )
                    if spec.joints.style == "under_desk":
                        for panel in (male, female):
                            if panel.base_footprint is None:
                                panel.base_footprint = panel.footprint
                    male.footprint, female.footprint = ma, fe
                    male.placement, female.placement = placements
                    male.joints += (joint,)
                    female.joints += (joint,)
                    stock.append(reserved)
                    joints.append(joint)
                    warnings.append(
                        f"{joint.id} ({spec.joints.style}): all native sockets preserved. "
                        "Physical fit/strength require testing."
                    )
                    accepted = True
                    count += 1
                    break
                if accepted:
                    break
        if not count:
            warnings.append(
                f"Seam {first.id}/{second.id}: no safe {spec.joints.style} connector; "
                "insufficient solid land, retained sockets or usable-bed space."
            )
    if spec.joints.style == "under_desk_puzzle":
        _add_puzzle_t_joints(panels, seams, joints, stock, spec, usable, slots, place, warnings)
    from .edges import add_edge_interfaces

    add_edge_interfaces(seams, joints, spec, usable, place, warnings)
    return joints, warnings


def _add_puzzle_t_joints(panels, seams, joints, stock, spec, usable, slots, place, warnings):
    """Atomically capture two split corners in one spanning-panel receiver.

    Bisect the existing full-height puzzle profile longitudinally.
    Each corner retains a 1.7 mm neck and an outward-facing shoulder. The
    mating half blocks inward motion; the spanning pocket blocks withdrawal.
    No Z catch is added: the existing row-directed supporting lips still apply.
    Only complete horizontal brick-row T nodes and the calibrated profile are
    eligible. Irregular/missing stock leaves all three panels unchanged.
    """
    preset = JointSpec(style="under_desk_puzzle", clearance=spec.joints.clearance)
    if spec.joints != preset:
        warnings.append("T-junction keys omitted: require the calibrated puzzle profile.")
        return
    male, female = puzzle_profiles(spec.joints)
    reservation = female.buffer(1.0)
    x0, y0, x1, y1 = reservation.bounds
    right = reservation.intersection(box(0, y0, x1, y1))
    quarters = [box(x0, y0, 0, 0), box(x0, 0, 0, y1)]
    roots = [reservation.intersection(q) for q in quarters]
    halves = [male.intersection(box(x0, lo, x1, hi)) for lo, hi in ((y0, 0), (0, y1))]
    nodes = sorted({(x, y) for _, _, seam in seams
                    for x, y, normal in _candidates(seam, spec.grid_origin) if normal == 90})
    tree = STRtree([p.footprint for p in panels])
    for x, y in nodes:
        meeting = [panels[int(i)] for i in tree.query(Point(x, y), predicate="intersects")]
        if len(meeting) != 3:
            continue
        accepted = False
        for angle in (90, 270):
            reserved = _pose(reservation, x, y, angle)
            if (not usable.covers(reserved) or reserved.intersects(slots)
                    or any(reserved.intersects(g) for g in stock)):
                continue
            receivers = [p for p in meeting if p.footprint.covers(_pose(right, x, y, angle))]
            if len(receivers) != 1:
                continue
            receiver = receivers[0]
            owners = [[p for p in meeting if p is not receiver
                       and p.footprint.covers(_pose(root, x, y, angle))] for root in roots]
            if any(len(group) != 1 for group in owners):
                continue
            corners = [group[0] for group in owners]
            if (corners[0] is corners[1] or receiver.grid_row is None
                    or corners[0].grid_row != corners[1].grid_row
                    or corners[0].grid_row is None
                    or abs(corners[0].grid_row - receiver.grid_row) != 1):
                continue
            keys = [_pose(half, x, y, angle) for half in halves]
            cuts = [_pose(half.buffer(spec.joints.clearance), x, y, angle) for half in halves]
            changed = [p.footprint.union(key) for p, key in zip(corners, keys)]
            changed.append(receiver.footprint.difference(union_all(cuts)))
            if not all(single_material(g, slots) and usable.covers(g) for g in changed):
                continue
            if any(changed[i].intersection(changed[j]).area > 1e-7
                   for i in range(3) for j in range(i + 1, 3)):
                continue
            placements = [place(g) for g in changed]
            if any(p is None for p in placements):
                continue
            # Commit both halves together, including their shared stock reservation.
            for panel, outline, placement in zip([*corners, receiver], changed, placements):
                panel.footprint, panel.placement = outline, placement
            for corner, key, cut in zip(corners, keys, cuts):
                joint = Joint(f"J{len(joints) + 1:03d}", corner.id, receiver.id, x, y,
                              angle, key, cut, spec.joints, reserved)
                corner.joints += (joint,)
                receiver.joints += (joint,)
                joints.append(joint)
            stock.append(reserved)
            warnings.append(f"T-junction ({x:g}, {y:g}): two integral half-keys captured by "
                            f"{receiver.id}; assemble the split row first. No independent Z lock "
                            "or strength rating; retain supporting lips and test physical fit.")
            accepted = True
            break
        if not accepted:
            warnings.append(f"T-junction ({x:g}, {y:g}): keys omitted; insufficient complete "
                            "corner/receiver stock, native socket clearance or usable-bed space.")
