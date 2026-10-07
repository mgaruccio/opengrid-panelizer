"""Experimental native Lite closure comparison; does not change production joints.

Run from the repository root:
    uv run --extra dev python experiments/closure_study.py --output <directory>

A: unadorned full-depth wall puzzle. B: the same puzzle with local 8 mm collars
(4 mm extra on the socket face; the mounting back remains flat).
C: a squared H/dogbone rigid key, top-loaded then slid 2.5 mm under a sloping
roof. Flat across-X bearing shoulders have no Y normal component. Reverse
sliding remains the intentional release: NO detent, friction lock, all-six-DOF
retention or measured stiffness is claimed. CAD clearances are not physical fit;
0.2 mm layer quantization and extrusion width require print calibration.

Research supplied by the lead: Prusa's modeling-with-3d-printing-in-mind_164135
(0.45 mm extrusion, calibrate fit), Princeton Sun 2023 connector_scf.pdf
(contact geometry matters; resin results are not PLA allowables), Formlabs snap
fit guidance (long beams reduce strain, not the simplification pursued here).
The geometry/assembly gate below uses the actual exported STEP and STL. The lead
owns the final Orca 0.4/0.2 mm support-free slice and any physical printing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

# A direct script invocation must resolve this checkout, not another editable install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import trimesh
from build123d import (
    Axis,
    Compound,
    Location,
    export_step,
    export_stl,
    extrude,
    import_step,
    loft,
)
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from opengrid.cad import (
    _footprint_face,
    _mounting_cutter,
    _single_volume,
    _socket_cutter,
    build_panel,
)
from opengrid.joints import puzzle_profiles
from opengrid.models import Cell, JointSpec, Panel, PrintPlacement
from opengrid.native import mounting_holes, socket_projection

TOL = 1e-5  # cubic millimetres, Boolean noise, NOT a manufacturing tolerance
PUZZLE = JointSpec(style="wall", depth=2.2, neck_width=3.4, head_width=4.6,
                   fillet_radius=0.4, clearance=0.05)
KEY_TRAVEL = 2.5
POCKET_ALLOWANCE = 0.20  # Fixed pocket offset; variants change only the key.
KEY_CLEARANCE = 0.05  # Positive XY gap per mating side, not a friction lock.
KEY_CLEARANCES = (KEY_CLEARANCE, 0.12, 0.20)
KEY_HALF_X = 2.05
KEY_WAIST_X = 1.15  # Leaves 0.95 mm seam-side stock below the retaining roof.
KEY_HALF_HEAD_Y = 1.20
KEY_HALF_NECK_Y = 0.70  # Smallest stem neck >=1.4 mm, ~three 0.45 mm lines.
KEY_BOTTOM = 1.25
KEY_FLANGE_TOP = 3.30
KEY_BEVEL_TOP = 4.15
KEY_TOP = 7.0
HEIGHTS = {"A": 4.0, "B": 8.0, "C": 6.0}


@dataclass(frozen=True)
class JointSite:
    id: str
    left: str
    right: str
    x: float
    y: float
    angle: int


@dataclass
class Sample:
    concept: str
    panel: Panel
    shape: object


def prism(outline: Polygon, z0: float, z1: float):
    return extrude(_footprint_face(outline), amount=z1 - z0, dir=(0, 0, 1)).translate(
        (0, 0, z0)
    )


def transition(bottom: Polygon, top: Polygon, z0: float, z1: float):
    return loft([_footprint_face(bottom).moved(Location((0, 0, z0))),
                 _footprint_face(top).moved(Location((0, 0, z1)))], ruled=True)


def pose(shape, site: JointSite):
    return shape.rotate(Axis.Z, site.angle).translate((site.x, site.y, 0))


def local(shape, site: JointSite):
    return shape.translate((-site.x, -site.y, 0)).rotate(Axis.Z, -site.angle)


def overlap(a, b) -> float:
    """Removed volume is robust to STEP round-trip coincident-face intersections."""
    ba, bb = a.bounding_box(), b.bounding_box()
    if any(min(ahi, bhi) - max(alo, blo) < 1e-7 for alo, ahi, blo, bhi in (
        (ba.min.X, ba.max.X, bb.min.X, bb.max.X),
        (ba.min.Y, ba.max.Y, bb.min.Y, bb.max.Y),
        (ba.min.Z, ba.max.Z, bb.min.Z, bb.max.Z),
    )):
        return 0.0
    remainder = a.cut(b)
    volume = a.volume - (remainder.volume if remainder is not None else 0.0)
    if volume < -TOL:
        raise ValueError("Boolean subtraction unexpectedly increased volume")
    return max(0.0, volume)


def require_clear(a, b, context: str) -> float:
    volume = overlap(a, b)
    if volume > TOL:
        raise ValueError(f"{context}: {volume:.6g} mm3 interference")
    return volume


@lru_cache(maxsize=1)
def collar_outline() -> Polygon:
    # Native sockets at (+/-14,+/-14) leave |x|+|y| < 5.897056 mm.
    # Rounding inward preserves that entire socket projection at every height.
    d = Polygon([(5.8, 0), (0, 5.8), (-5.8, 0), (0, -5.8)])
    return d.buffer(-0.6).buffer(0.6)


def bowtie(half_x: float = KEY_HALF_X, travel: float = 0, offset: float = 0) -> Polygon:
    """Squared H profile (legacy function name); exact continuous Y sweep.

    The old diagonal shoulders had |ny/nx|=3.4 and could cam the free key
    toward -Y as panels separated. These straight shoulders have normals +/-X.
    Each vertical section is an interval, so expanding its two Y ends by
    travel/2 is the Minkowski sum with that Y segment, also through the loft.
    """
    waist_x = KEY_WAIST_X * half_x / KEY_HALF_X
    w, h = KEY_HALF_NECK_Y + travel / 2, KEY_HALF_HEAD_Y + travel / 2
    p = Polygon([(-half_x, -h), (-waist_x, -h), (-waist_x, -w),
                 (waist_x, -w), (waist_x, -h), (half_x, -h),
                 (half_x, h), (waist_x, h), (waist_x, w),
                 (-waist_x, w), (-waist_x, h), (-half_x, h)])
    return p.buffer(offset, join_style="mitre") if offset else p


@lru_cache(maxsize=8)
def key_solid(clearance: float = KEY_CLEARANCE, travel: float = 0):
    """One rigid key, or its exact continuous tangential swept volume.

    Seated key centre is Y=+1.25; travel=2.5 gives the sweep centred at Y=0.
    Variant clearance changes the key only; all panels have identical pockets.
    """
    if not 0.05 <= clearance <= POCKET_ALLOWANCE or travel < 0:
        raise ValueError("key clearance must be 0.05..0.20 mm and travel nonnegative")
    offset = POCKET_ALLOWANCE - clearance
    flange, stem = bowtie(offset=offset, travel=travel), bowtie(1.2, travel, offset)
    result = prism(flange, KEY_BOTTOM, KEY_FLANGE_TOP).fuse(
        transition(flange, stem, KEY_FLANGE_TOP, KEY_BEVEL_TOP),
        prism(stem, KEY_BEVEL_TOP, KEY_TOP),
    )
    centre = KEY_TRAVEL / 2 if not travel else 0
    return _single_volume(result.translate((0, centre, 0)), context="rigid key")


@lru_cache(maxsize=1)
def key_cutter():
    """Open loading bay, straight floor, and support-free sloping retaining roof."""
    flange = bowtie(travel=KEY_TRAVEL, offset=POCKET_ALLOWANCE)
    throat = bowtie(1.2, KEY_TRAVEL, POCKET_ALLOWANCE)
    channel = prism(flange, 1.2, 3.35).fuse(
        transition(flange, throat, 3.35, 4.2), prism(throat, 4.2, 8)
    )
    # Full projection of every section, including the sloping Z shoulder.
    half_x = KEY_HALF_X + POCKET_ALLOWANCE
    half_y = KEY_HALF_HEAD_Y + POCKET_ALLOWANCE
    bay = box(-half_x, -KEY_TRAVEL / 2 - half_y, half_x, -KEY_TRAVEL / 2 + half_y)
    return channel.fuse(prism(bay, 1.2, 8))


def build_grid(concept: str, nx: int, ny: int, *, cells_x: int = 2,
               cells_y: int = 2, prefix: str | None = None):
    """Internal seams only, at complete lattice nodes, never at a four-way corner."""
    if concept not in HEIGHTS or min(nx, ny, cells_x, cells_y) < 1:
        raise ValueError("invalid concept or grid size")
    prefix = prefix or concept
    width, height = 28 * cells_x, 28 * cells_y
    panels = {}
    for iy in range(ny):
        for ix in range(nx):
            x, y = (ix - 1) * width, (iy - 1) * height
            name = f"{prefix}{iy * nx + ix + 1}"
            cells = tuple(Cell((ix - 1) * cells_x + cx, (iy - 1) * cells_y + cy,
                               x + 14 + 28 * cx, y + 14 + 28 * cy)
                          for cy in range(cells_y) for cx in range(cells_x))
            panels[ix, iy] = Panel(name, box(x, y, x + width, y + height), cells,
                                   PrintPlacement(0, 0, 0))
    sites = []
    for (ix, iy), panel in panels.items():
        x0, y0, x1, y1 = panel.footprint.bounds
        for dx, dy, count, angle in ((1, 0, cells_y, 0), (0, 1, cells_x, 90)):
            if (ix + dx, iy + dy) not in panels:
                continue
            other = panels[ix + dx, iy + dy]
            for k in range(1, count):
                x, y = (x1, y0 + 28 * k) if dx else (x0 + 28 * k, y1)
                sites.append(JointSite(f"{prefix}J{len(sites) + 1}", panel.id,
                                       other.id, x, y, angle))
    by_id = {p.id: p for p in panels.values()}
    if concept != "C":
        for site in sites:
            male, female = puzzle_profiles(PUZZLE, site.x, site.y, site.angle)
            by_id[site.left].footprint = by_id[site.left].footprint.union(male)
            by_id[site.right].footprint = by_id[site.right].footprint.difference(female)
    samples = []
    for panel in panels.values():
        solid = build_panel(panel)  # exact native sockets and interior mounting recesses
        if concept in ("B", "C"):
            for site in sites:
                if panel.id not in (site.left, site.right):
                    continue
                collar = translate(collar_outline(), xoff=site.x, yoff=site.y)
                solid = solid.fuse(prism(collar.intersection(panel.footprint), 4, HEIGHTS[concept]))
                if concept == "C":
                    solid = solid.cut(pose(key_cutter(), site))
        samples.append(Sample(concept, panel, _single_volume(solid, context=panel.id)))
    return samples, sites


def bounds(shape):
    b = shape.bounding_box()
    return [[b.min.X, b.min.Y, b.min.Z], [b.max.X, b.max.Y, b.max.Z]]


def mesh_check(path: Path, cad):
    mesh = trimesh.load_mesh(path)
    result = {"watertight": bool(mesh.is_watertight), "positive_volume": bool(mesh.is_volume),
              "components": len(mesh.split()), "stl_volume_mm3": float(mesh.volume),
              "step_volume_mm3": cad.volume}
    if not result["watertight"] or not result["positive_volume"] or result["components"] != 1:
        raise ValueError(f"invalid exported STL: {path}: {result}")
    if not math.isclose(mesh.volume, cad.volume, rel_tol=0.003, abs_tol=0.01):
        raise ValueError(f"STL/STEP volume mismatch: {path}")
    return result


def export_part(output: Path, name: str, shape, x: float, y: float, rotation: int = 0):
    rotated = shape.rotate(Axis.Z, rotation)
    b = rotated.bounding_box()
    offset = (x - b.min.X, y - b.min.Y, -b.min.Z)
    printable = rotated.translate(offset)
    directory = output / "parts"
    directory.mkdir(parents=True, exist_ok=True)
    step_path, stl_path = directory / f"{name}.step", directory / f"{name}.stl"
    export_step(printable, step_path)
    export_stl(printable, stl_path, tolerance=0.015, angular_tolerance=0.08)
    imported = import_step(step_path)
    _single_volume(imported, context=f"exported {name}")
    # Discard the reader's assembly wrapper; re-exporting transformed nested
    # STEP assemblies (not their geometry) can fail in OCCT's STEP writer.
    imported = imported.solids()[0]
    result = mesh_check(stl_path, imported)
    record = {"id": name, "step": f"parts/{name}.step", "stl": f"parts/{name}.stl",
              "print_placement": {"rotation": rotation, "translation": list(offset)},
              "print_bounds_mm": bounds(imported), "print_orientation": "flat, native Z=0 on bed"}
    seated = imported.translate(tuple(-v for v in offset)).rotate(Axis.Z, -rotation)
    return record, seated, imported, result


def verify_native(sample: Sample, solid):
    sockets, projections, mounts = [], [], []
    for cell in sample.panel.cells:
        sockets.append(require_clear(solid, _socket_cutter().translate((cell.x, cell.y, 0)),
                                     f"{sample.panel.id} native socket"))
        projections.append(require_clear(solid, prism(socket_projection(cell.x, cell.y), 4, 9),
                                         f"{sample.panel.id} extended socket projection"))
    for x, y in mounting_holes(sample.panel):
        mounts.append(require_clear(solid, _mounting_cutter().translate((x, y, 0)),
                                    f"{sample.panel.id} mounting recess"))
    return {"socket_count": len(sockets), "mount_count": len(mounts),
            "max_socket_interference_mm3": max(sockets, default=0),
            "max_above_base_projection_interference_mm3": max(projections, default=0),
            "max_mount_interference_mm3": max(mounts, default=0)}


def measure_key_play(site, shapes, key):
    """First CAD contact at fixed seated Y and orientation; NOT stiffness/retention.

    Z is the key's total travel against both fixed panels. X is the sum of
    outward panel travel against the fixed key. Coupled/rotating escape modes
    and friction are deliberately not inferred from these one-axis tests.
    """
    key = local(key, site)
    panels = [local(shapes[getattr(site, role)], site) for role in ("left", "right")]

    def contact_distance(moving, fixed, direction):
        lo, hi = 0.0, 0.4
        require_clear(moving, fixed, "play measurement seated fit")
        if overlap(moving.translate(tuple(hi * d for d in direction)), fixed) <= TOL:
            raise ValueError("no first-contact bracket within 0.4 mm")
        for _ in range(14):
            mid = (lo + hi) / 2
            if overlap(moving.translate(tuple(mid * d for d in direction)), fixed) > TOL:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    both = Compound(children=panels)
    down = contact_distance(key, both, (0, 0, -1))
    up = contact_distance(key, both, (0, 0, 1))
    left = contact_distance(panels[0], key, (-1, 0, 0))
    right = contact_distance(panels[1], key, (1, 0, 0))
    return {"z_minus_mm": round(down, 4), "z_plus_mm": round(up, 4),
            "pure_z_total_mm": round(down + up, 4),
            "x_left_outward_mm": round(left, 4), "x_right_outward_mm": round(right, 4),
            "across_x_separation_mm": round(left + right, 4),
            "search_resolution_mm": 0.4 / 2**14, "collision_tolerance_mm3": TOL,
            "scope": "first CAD contact, fixed seated Y/orientation; not stiffness or all-DOF retention"}


def verify_assembly(samples, sites, shapes, keys=None, *, key_clearance=KEY_CLEARANCE):
    """Whole-panel insertion and continuous key paths, not isolated coupon proxies.

    Panel insertion is certified with a conservative full-footprint vertical
    prism: every actual panel must be contained in it, and the prism must clear
    every other actual panel. This proves all Z positions, including insertion
    into an already completed 2-D array. Actual displaced STEP solids are also
    tested. The key Y sweep is exact for these Y-monotone loft sections.
    """
    report = {"panel_count": len(samples), "joint_count": len(sites),
              "seated_max_interference_mm3": 0.0, "panel_insertion": [], "key_insertion": []}
    for a, b in combinations(shapes.values(), 2):
        report["seated_max_interference_mm3"] = max(
            report["seated_max_interference_mm3"], require_clear(a, b, "seated panels"))
    for sample in samples:
        name = sample.panel.id
        solid = shapes[name]
        envelope = prism(sample.panel.footprint, 0, 20)
        outside = solid.volume - overlap(solid, envelope)
        if abs(outside) > TOL:
            raise ValueError(f"{name} escapes its vertical insertion certificate")
        max_interference = 0.0
        for other_id, other in shapes.items():
            if other_id == name:
                continue
            max_interference = max(max_interference,
                                   require_clear(envelope, other, f"{name} continuous Z insertion"))
            for dz in (8.0, 3.0, 0.2):
                require_clear(solid.translate((0, 0, dz)), other, f"{name} actual Z={dz}")
        report["panel_insertion"].append({"id": name, "direction": [0, 0, -1],
                                          "continuous_envelope_interference_mm3": max_interference})
    if samples[0].concept != "C":
        report["positive_z_capture"] = False
        report["release"] = "lift any panel vertically; no edge lips, springs or Z lock"
        return report
    keys = keys or {site.id: pose(key_solid(key_clearance), site) for site in sites}
    projections = [prism(socket_projection(c.x, c.y), 0, 9)
                   for sample in samples for c in sample.panel.cells]
    for site in sites:
        key = keys[site.id]
        all_panels = Compound(children=list(shapes.values())
                              + [k for name, k in keys.items() if name != site.id])
        require_clear(key, all_panels, f"{site.id} seated key")
        projected = max(require_clear(key, cavity, f"{site.id} key socket projection")
                        for cavity in projections)
        # Full rectangular projection is conservative for every section of the
        # actual exported key, so this certifies continuous top-entry as well.
        b = local(key, site).bounding_box()
        vertical = pose(prism(box(b.min.X, b.min.Y - KEY_TRAVEL,
                                  b.max.X, b.max.Y - KEY_TRAVEL), KEY_BOTTOM, 20), site)
        entry = require_clear(vertical, all_panels, f"{site.id} vertical key entry")
        slide = require_clear(pose(key_solid(key_clearance, travel=KEY_TRAVEL), site), all_panels,
                              f"{site.id} continuous tangential key insertion")
        for dy, dz in ((-KEY_TRAVEL, 8), (-KEY_TRAVEL, 0), (-1.25, 0), (0, 0)):
            moved = pose(local(key, site).translate((0, dy, dz)), site)
            require_clear(moved, all_panels, f"{site.id} actual key insertion")
        # Each half has a real floor, upper shoulder, and across-seam shoulder.
        # A displaced-key collision is a geometric constraint, NOT stiffness.
        capture = {}
        for role, sign in (("left", -1), ("right", 1)):
            panel = shapes[getattr(site, role)]
            for direction, delta in (("z_minus", (0, 0, -0.5)), ("z_plus", (0, 0, 0.5)),
                                     ("across_seam", (sign * 1.0, 0, 0))):
                shifted = pose(local(panel, site).translate(delta), site)
                contact = overlap(key, shifted)
                if contact < 0.01:
                    raise ValueError(f"{site.id} missing {role} {direction} capture")
                capture[f"{role}_{direction}_collision_mm3"] = contact
        report["key_insertion"].append({"joint": site.id, "top_entry_interference_mm3": entry,
                                        "continuous_slide_interference_mm3": slide,
                                        "socket_projection_interference_mm3": projected,
                                        "travel_mm": KEY_TRAVEL, "capture": capture})
    report["positive_z_capture"] = True
    report["release"] = "reverse the 2.5 mm tangential slide, then lift; friction unproven, no detent"
    return report


# Font-independent single-stroke labels, engraved on separate tags, never sockets.
_GLYPHS = {
    "A": [[(0, 0), (0, 4), (2, 4), (2, 0)], [(0, 2), (2, 2)]],
    "B": [[(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)], [(0, 2), (2, 2)]],
    "C": [[(2, 4), (0, 4), (0, 0), (2, 0)]],
    "1": [[(1, 0), (1, 4), (0, 3)]],
    "2": [[(0, 4), (2, 4), (2, 2), (0, 2), (0, 0), (2, 0)]],
    "3": [[(0, 4), (2, 4), (2, 0), (0, 0)], [(0, 2), (2, 2)]],
    "4": [[(0, 4), (0, 2), (2, 2)], [(2, 4), (2, 0)]],
}


def label_tag(label: str):
    from shapely.geometry import LineString

    tag = prism(box(0, 0, 3 * len(label) + 2, 7), 0, 1.2)
    for i, char in enumerate(label):
        for stroke in _GLYPHS[char]:
            tool = LineString([(x + 1.5 + 3 * i, y + 1.5) for x, y in stroke]).buffer(
                0.25, cap_style="square", join_style="mitre")
            tag = tag.cut(prism(tool, 0.8, 1.4))
    return _single_volume(tag, context=f"tag {label}")


def export_fea(output, concept, site, shapes, key=None):
    directory = output / "fea" / concept
    directory.mkdir(parents=True, exist_ok=True)
    outside = prism(box(-300, -300, 300, 300).difference(box(-5, -6, 5, 6)), -1, 20)
    volumes = {}
    for role in ("left", "right"):
        # Cut the complement: OCC common/intersect can return an empty result
        # for nearly coincident STEP round-trip faces despite a valid overlap.
        piece = local(shapes[getattr(site, role)], site).cut(outside)
        _single_volume(piece, context=f"FEA {concept}/{role}")
        export_step(piece, directory / f"{role}.step")
        reread = import_step(directory / f"{role}.step")
        _single_volume(reread, context=f"reimport FEA {concept}/{role}")
        volumes[role] = reread.volume
    if key is not None:
        piece = local(key, site).solids()[0]
        export_step(piece, directory / "key.step")
        volumes["key"] = piece.volume
    return {"directory": f"fea/{concept}", "seam": "X=0", "clamp_planes_mm": [-5, 5],
            "y_bounds_mm": [-6, 6], "volumes_mm3": volumes,
            "contact": "separate actual bodies; do not tie/fuse mating faces",
            "caution": "local ranking specimen, not whole-panel stiffness or calibrated PLA strength"}


def generate(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"units": "mm", "experimental": True, "parts": [], "concepts": {},
                "plate": {"size_mm": [256, 256], "stl": "plate.stl", "step": "plate.step"},
                "print_settings": {"nozzle_mm": 0.4, "layer_mm": 0.2, "supports": False,
                                   "note": "Orca validation and physical fit pending; brim tiny keys"},
                "limitations": ["No measured stiffness, FEA solution, friction or strength claim.",
                                ("A/B have no positive Z lock; B is 8 mm local height, with 4 mm extra "
                                 "on the socket face; the mounting back stays flat (no wood stand-off)."),
                                "C intentionally releases by reverse sliding; no friction lock or all-DOF retention.",
                                "Positive CAD gaps are not guaranteed physical fit at 0.2 mm layers.",
                                "Keep screw/mount positions and native socket orientation unchanged."]}
    verification = {"parts": {}, "assemblies": {}, "two_by_two": {}, "spare_keys": {}}
    plate_shapes, plate_boxes = [], []
    for concept in "ABC":
        samples, sites = build_grid(concept, 2, 2 if concept == "C" else 1,
                                    cells_y=2 if concept == "C" else 4)
        shapes, keys = {}, {}
        for i, sample in enumerate(samples):
            x = 4 + 62 * i if concept == "C" else 4 + 124 * i
            y = {"A": 4, "B": 70, "C": 136}[concept]
            record, seated, printed, checks = export_part(
                output, sample.panel.id, sample.shape, x, y, 0 if concept == "C" else 90)
            record.update({"concept": concept, "kind": "panel",
                           "cells": [asdict(c) for c in sample.panel.cells],
                           "mounting_holes": mounting_holes(sample.panel),
                           "joint_ids": [s.id for s in sites
                                         if sample.panel.id in (s.left, s.right)],
                           "assembly_direction": [0, 0, -1]})
            manifest["parts"].append(record)
            verification["parts"][sample.panel.id] = checks | verify_native(sample, seated)
            shapes[sample.panel.id] = seated
            plate_shapes.append(printed)
            plate_boxes.append((sample.panel.id, record["print_bounds_mm"]))
            tag, _, tag_print, tag_check = export_part(
                output, f"tag-{sample.panel.id}", label_tag(sample.panel.id), 4 + 12 * len(shapes)
                + {"A": 0, "B": 30, "C": 60}[concept], 204)
            tag.update({"kind": "label_tag", "concept": concept, "labels": sample.panel.id})
            manifest["parts"].append(tag)
            verification["parts"][tag["id"]] = tag_check
            plate_shapes.append(tag_print)
            plate_boxes.append((tag["id"], tag["print_bounds_mm"]))
        if concept == "C":
            for i, site in enumerate(sites):
                name = f"key-{site.id}"
                record, seated, printed, check = export_part(output, name, pose(key_solid(), site),
                                                            8 + i * 14, 224, -site.angle)
                record.update({"concept": "C", "kind": "key", "joint_id": site.id,
                               "xy_clearance_per_side_mm": KEY_CLEARANCE,
                               "print_orientation": "flat flange bottom on bed; small-key brim advised"})
                manifest["parts"].append(record)
                verification["parts"][name] = check
                keys[site.id] = seated
                plate_shapes.append(printed)
                plate_boxes.append((name, record["print_bounds_mm"]))
            site = sites[0]
            verification["key_play"] = {
                f"{KEY_CLEARANCE:.2f}": measure_key_play(site, shapes, keys[site.id])}
            # Four keys of every fit size let the entire 2x2 be assembled even
            # if the tightest nominal fit needs more clearance on this printer.
            for i in range(8):
                variant, copy = divmod(i, 4)
                clearance = KEY_CLEARANCES[variant + 1]
                name = f"key-spare-{clearance:.2f}-{copy + 1}"
                key = key_solid(clearance)
                record, seated, printed, check = export_part(
                    output, name, key, 78 + 70 * variant + 14 * copy, 224)
                key = seated  # All spare-key checks use the actual STEP round-trip.
                if copy == 0:
                    verification["key_play"][f"{clearance:.2f}"] = measure_key_play(
                        site, shapes, pose(key, site))
                record.update({"concept": "C", "kind": "spare_key",
                               "xy_clearance_per_side_mm": clearance,
                               "print_orientation": "flat flange bottom on bed; small-key brim advised"})
                manifest["parts"].append(record)
                verification["parts"][name] = check
                both = Compound(children=list(shapes.values()))
                site = sites[0]
                b = key.bounding_box()
                entry = prism(box(b.min.X, b.min.Y - KEY_TRAVEL, b.max.X, b.max.Y - KEY_TRAVEL),
                              KEY_BOTTOM, 20)
                verification["spare_keys"][name] = {
                    "entry_interference_mm3": require_clear(pose(entry, site), both, name),
                    "slide_interference_mm3": require_clear(
                        pose(key_solid(clearance, KEY_TRAVEL), site), both, name)}
                plate_shapes.append(printed)
                plate_boxes.append((name, record["print_bounds_mm"]))
        assembly = Compound(children=list(shapes.values()) + list(keys.values()))
        export_step(assembly, output / f"assembly-{concept}.step")
        verification["assemblies"][concept] = verify_assembly(samples, sites, shapes, keys)
        if concept == "C":
            verification["two_by_two"][concept] = verification["assemblies"][concept]
        else:
            # Same features on BOTH axes. Not on the print plate, but a real CAD
            # array that exercises the two-neighbour corner insertion problem.
            proof, proof_sites = build_grid(concept, 2, 2, prefix=f"{concept}-proof-")
            proof_path = output / f"proof-2x2-{concept}.step"
            export_step(Compound(children=[s.shape for s in proof]), proof_path)
            imported = import_step(proof_path)
            solids = imported.solids()
            if len(solids) != 4:
                raise ValueError(f"{concept} 2x2 STEP lost its four separate panels")
            # STEP child order is not an identity guarantee. Match each exported
            # solid by volume and centre, and require an unambiguous one-to-one map.
            proof_shapes = {}
            remaining = list(solids)
            for sample in proof:
                matches = [s for s in remaining if (s.center() - sample.shape.center()).length < 1e-4
                           and abs(s.volume - sample.shape.volume) < TOL]
                if len(matches) != 1:
                    raise ValueError("ambiguous 2x2 STEP solid identity")
                proof_shapes[sample.panel.id] = matches[0]
                remaining.remove(matches[0])
            verification["two_by_two"][concept] = verify_assembly(proof, proof_sites, proof_shapes)
        manifest["concepts"][concept] = {
            "name": {"A": "plain full-depth puzzle", "B": "8 mm reinforced puzzle collar",
                     "C": "top-loaded tangential squared rigid key"}[concept],
            "assembly_step": f"assembly-{concept}.step", "height_mm": HEIGHTS[concept],
            "joints": [asdict(s) for s in sites],
            "assembly": ("place panels at final XY, insert keys from +Z at local Y=-1.25, "
                         "then slide +local Y 2.5 mm; rotate both vectors by joint angle"
                         if concept == "C" else "vertical placement; no tangential movement required"),
            "fea": export_fea(output, concept, sites[0], shapes, keys.get(sites[0].id)),
        }
    for name, b in plate_boxes:
        if min(b[0][:2]) < -1e-5 or max(b[1][:2]) > 256 or abs(b[0][2]) > 1e-5:
            raise ValueError(f"{name} does not fit the flat 256 mm plate")
    for (a, ba), (b, bb) in combinations(plate_boxes, 2):
        if min(ba[1][0], bb[1][0]) > max(ba[0][0], bb[0][0]) and \
                min(ba[1][1], bb[1][1]) > max(ba[0][1], bb[0][1]):
            raise ValueError(f"plate bounding boxes overlap: {a}, {b}")
    plate = Compound(children=plate_shapes)
    export_step(plate, output / "plate.step")
    export_stl(plate, output / "plate.stl", tolerance=0.015, angular_tolerance=0.08)
    mesh = trimesh.load_mesh(output / "plate.stl")
    if not mesh.is_volume or len(mesh.split()) != len(plate_shapes):
        raise ValueError("combined plate STL lost watertight separate components")
    verification["plate"] = {"components": len(plate_shapes), "watertight": bool(mesh.is_watertight),
                             "bounds_mm": mesh.bounds.tolist(), "box_overlap": False}
    manifest["parameters"] = {"puzzle": asdict(PUZZLE), "collar_diamond_radius_mm": 5.8,
                              "collar_corner_radius_mm": 0.6, "key_travel_mm": KEY_TRAVEL,
                              "key_floor_mm": 1.2, "key_bottom_mm": KEY_BOTTOM,
                              "key_flange_thickness_mm": KEY_FLANGE_TOP - KEY_BOTTOM,
                              "key_flange_top_mm": KEY_FLANGE_TOP, "key_bevel_top_mm": KEY_BEVEL_TOP,
                              "key_roof_nominal_slope_deg": 45,
                              "key_pocket_allowance_mm": POCKET_ALLOWANCE,
                              "key_flange_retaining_land_mm": KEY_WAIST_X - POCKET_ALLOWANCE,
                              "key_min_stem_neck_mm": 2 * KEY_HALF_NECK_Y,
                              "key_xy_clearance_per_side_mm": KEY_CLEARANCE,
                              "key_clearance_variants_mm": list(KEY_CLEARANCES)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "geometry-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return manifest, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("opengrid/verification/closure-study/generated"))
    args = parser.parse_args()
    manifest, _ = generate(args.output)
    print(f"Generated and geometry-verified {len(manifest['parts'])} parts in {args.output}")
    print("Experimental: slice in Orca and calibrate fit before printing; no stiffness claim.")


if __name__ == "__main__":
    main()
