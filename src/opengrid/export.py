"""CAD and assembly exports. The manifest is written only after all exports succeed."""

import json
import math
from dataclasses import asdict
from pathlib import Path

from shapely import union_all
from shapely.geometry import mapping
from shapely.strtree import STRtree

from .drawings import write_pdf, write_svg
from .models import PITCH, THICKNESS, Layout, board_thickness
from .native import mounting_holes
from .spec import SpecError, ensure_ready
from .underdesk import ASSEMBLY_TRAVEL

NATIVE_SOURCE = (
    "https://www.printables.com/model/1214361-opengrid-walldesk-mounting-framework-and-ecosystem"
)


def _attribution(board: str) -> str:
    family = "Full" if board == "full" else "Lite"
    geometry = "accessory" if board == "full" else "accessory and screw-hole"
    mounting = (
        "Full uses separate native mounting tiles/snaps; no integrated screw holes are added."
        if board == "full" else
        "The native screw-hole profile is reflected to open its head recess at Z=4."
    )
    return f"""Native openGrid {family} {geometry} geometry is derived from David D's openGrid,
licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Source: https://www.printables.com/model/1214361-opengrid-walldesk-mounting-framework-and-ecosystem
Panel design contributions: mgaruccio / openGrid Panelizer, CC BY 4.0.
Project: https://github.com/mgaruccio/opengrid-panelizer
Compiler software is separately MIT licensed; native and generated CAD retain CC BY 4.0.
Changes: custom installation outlines, panelization, and adapted puzzle joints;
these joints are not an official openGrid board-to-board interface.
{mounting}
No endorsement by the source author is implied. Physical print fit remains
unverified until a coupon is printed and checked with the intended accessories.
"""


ATTRIBUTION = _attribution("lite")  # Retain the existing Lite attribution constant.


def prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SpecError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _joint_assembly_axis(joint, layout: Layout) -> list[int]:
    """Relative seating motion of the puzzle/clip male, not always the lip male."""
    if layout.spec.joints.style == "under_desk":
        angle = math.radians(joint.angle)
        return [round(math.cos(angle)), round(math.sin(angle)), 0]
    if layout.spec.joints.style == "under_desk_puzzle":
        pair = {joint.male_panel, joint.female_panel}
        for panel in layout.panels:
            if panel.id == joint.male_panel:
                for edge in panel.edges:
                    if {edge.male_panel, edge.female_panel} == pair:
                        return [0, 0, 1 if edge.male_panel == joint.male_panel else -1]
    return [0, 0, 1]


def _puzzle_support(layout: Layout) -> dict:
    """Conservative candidates from actual lips/holes, not a screw-count/load model."""
    panels = {p.id: p for p in layout.panels}
    if any(p.grid_row is None for p in panels.values()):
        raise SpecError("Puzzle support layout is missing its global row indices")
    rows = {}
    for panel in panels.values():
        rows.setdefault(panel.grid_row, []).append(panel)
    for row in rows.values():
        row.sort(key=lambda p: (min(c.ix for c in p.cells), min(c.iy for c in p.cells), p.id))
    order = sorted(rows, key=lambda r: (not r % 2, r))  # All infill before anchor rows.
    rank = {p.id: i for i, p in enumerate(p for r in order for p in rows[r])}
    edges = {e.id: e for p in panels.values() for e in p.edges}
    for edge in edges.values():
        if rank[edge.male_panel] >= rank[edge.female_panel]:
            raise SpecError(f"Puzzle lip {edge.id} contradicts the receiver-lowering order")
    holes = {p.id: mounting_holes(p) for p in panels.values()}
    records, blockers = [], []
    for panel in panels.values():
        row = panel.grid_row
        supports = {"north": [], "south": []}  # +Y / -Y, in installation coordinates.
        if layout.spec.board == "full":
            records.append({
                "panel": panel.id, "grid_row": row, "role": "infill" if row % 2 else "anchor",
                "fastening": "native_tiles_required", "supporting_lips": supports,
                "direct_reasons": [("Full requires native mounting tiles/snaps on every panel; "
                                    "indirect-only support is not approved")],
            })
            blockers.append(f"{panel.id}: user must arrange and verify native Full mounting tiles/snaps; "
                            "tile placement and attachment are not verified by this generator")
            continue
        for edge in panel.edges:
            if edge.male_panel != panel.id or edge.angle not in (90, 270):
                continue
            anchor = panels[edge.female_panel]
            delta = 1 if edge.angle == 90 else -1
            if (row % 2 and anchor.grid_row == row + delta and not anchor.grid_row % 2
                    and edge in anchor.edges and holes[anchor.id]):
                supports["north" if delta == 1 else "south"].append(
                    {"panel": anchor.id, "lip": edge.id}
                )
        base = panel.base_footprint if panel.base_footprint is not None else panel.footprint
        interior = base.distance(layout.usable.boundary) > 1e-7
        # A clipped or recursively split nominal brick is never treated as an infill candidate.
        whole_brick = len(panel.cells) == math.floor(layout.spec.printer.max_panel_span / PITCH) ** 2
        candidate = bool(row % 2 and interior and whole_brick
                         and supports["north"] and supports["south"])
        reasons = []
        if not row % 2:
            reasons.append("anchor row")
        if not interior:
            reasons.append("touches usable-region boundary")
        if not whole_brick:
            reasons.append("partial or fragmented brick")
        if not supports["north"] or not supports["south"]:
            reasons.append("both north AND south native-holed anchors are required")
        if not candidate and not holes[panel.id]:
            blockers.append(f"{panel.id}: direct fastening required but no complete native mounting hole")
        records.append({
            "panel": panel.id, "grid_row": row, "role": "infill" if row % 2 else "anchor",
            "fastening": "indirect_support_candidate" if candidate else "direct_required",
            "supporting_lips": supports, "direct_reasons": reasons if not candidate else [],
        })
    steps = []
    def step(action, instruction, **data):
        steps.append({"number": len(steps) + 1, "action": action, **data, "instruction": instruction})
    thickness = board_thickness(layout.spec.board)
    step("prepare", "Preassemble the entire board away from the desk. Z=0 is the mounting face; "
         f"Z={thickness:g} is the accessory face. Keep all parts in this pose; do not mirror or flex them. "
         + ("Do not fasten anchor rows to the desk first." if layout.spec.board == "full" else
            "Do not screw anchor rows to the desk first."))
    for row in order:
        ids = [p.id for p in rows[row]]
        step("build_row", f"Build row {row} west-to-east: {', '.join(ids)}. "
             "Where a registered lip exists, lower the next receiver along -Z onto the western tongue; "
             "otherwise position the loose piece without assuming capture. "
             "Keep other rows separate while building this row; hold unconnected fragments aligned.",
             grid_row=row, panels=ids)
        if row % 2:
            step("position_infill", f"Position infill row {row} ({', '.join(ids)}) at its final XY location.",
                 grid_row=row, panels=ids)
        else:
            adjacent = sorted({panels[e.male_panel].grid_row for e in edges.values()
                               if e.female_panel in ids and e.angle in (90, 270)})
            neighbours = [p.id for r in adjacent for p in rows[r]]
            step("lower_anchor_row", f"Lower whole anchor row {row} ({', '.join(ids)}) along -Z "
                 f"onto adjacent infill panels {', '.join(neighbours) or '(none)'}. "
                 "Keep the complete row rigid and aligned; do not insert a final panel diagonally.",
                 grid_row=row, panels=ids, adjacent_infill_panels=neighbours)
    direct = [r["panel"] for r in records if r["fastening"] != "indirect_support_candidate"]
    if layout.spec.board == "full":
        step("mount", f"Only after full-board preassembly, mount every panel: {', '.join(direct)}. "
             "Use separate native Full mounting tiles/snaps, never drill sockets or add Lite screw holes. "
             "The user must arrange and verify suitable tile placement and attachment for each panel. "
             "Resolve every listed mounting blocker before installation; support all panels during handling.",
             panels=direct)
        limitation = (
            "Full requires native mounting tiles/snaps on every panel; indirect-only support is not approved. "
            "Mounting tiles/snaps are not generated or positioned here. The user must arrange and verify them. "
            "Lips and puzzle keys are not load-rated; physical fit, creep, substrate and load testing remain necessary."
        )
    else:
        step("mount", f"Only after full-board preassembly, mount anchors and exceptions: {', '.join(direct)}. "
             "Use their existing native mounting holes listed in panels; never drill sockets. "
             "Stop if a mounting blocker is listed. Support all panels during handling and installation.",
             panels=direct)
        limitation = (
            "Indirect support candidates are not load-rated or a safe/minimum screw count. "
            "Native connectors primarily align; physical lip, creep, fastener, substrate and load "
            "testing remain necessary. Add direct fastening as needed; reductions are not guaranteed."
        )
    return {
        "mounting_face_z_mm": 0, "accessory_face_z_mm": thickness,
        "tongue_z_mm": [0.1, 1.9], "north": "+Y", "south": "-Y",
        "panels": records, "assembly_steps": steps, "mounting_blockers": blockers,
        "limitation": limitation,
    }


def export_layout(layout: Layout, output: str | Path) -> dict:
    ensure_ready(layout.spec)
    from build123d import Axis, Compound, export_step, export_stl

    from .cad import build_panel

    support = _puzzle_support(layout) if layout.spec.joints.style == "under_desk_puzzle" else None
    path = Path(output)
    prepare_output(path)
    panel_records = []
    assembly = []
    for panel in layout.panels:
        print(f"Building {panel.id}: {len(panel.cells)} cells", flush=True)
        shape = build_panel(panel)
        shape.label = panel.id
        assembly.append(shape)
        placement = panel.placement
        printed = shape.rotate(Axis.Z, placement.rotation).translate((placement.x, placement.y, 0))
        if not export_step(printed, path / f"{panel.id}.step"):
            raise SpecError(f"STEP export failed for {panel.id}")
        if not export_stl(printed, path / f"{panel.id}.stl", tolerance=0.01, angular_tolerance=0.1):
            raise SpecError(f"STL export failed for {panel.id}")
        panel_records.append(
            {
                "id": panel.id,
                "footprint": mapping(panel.footprint),
                "installation_bounds": list(panel.footprint.bounds),
                "cells": [{"ix": c.ix, "iy": c.iy, "x": c.x, "y": c.y} for c in panel.cells],
                "mounting_holes": [{"x": x, "y": y} for x, y in mounting_holes(panel)],
                "print_placement": {
                    "rotation": placement.rotation,
                    "x": placement.x,
                    "y": placement.y,
                },
                "volume_mm3": shape.volume,
                "step": f"{panel.id}.step",
                "stl": f"{panel.id}.stl",
            }
        )
    if not panel_records:
        raise SpecError("No printable panels")
    if layout.spec.joints.style in {"wall", "under_desk", "under_desk_puzzle"}:
        tree = STRtree([p.footprint for p in layout.panels])
        for i, panel in enumerate(layout.panels):
            for k in tree.query(panel.footprint, predicate="intersects"):
                if k > i and assembly[i].volume - assembly[i].cut(assembly[k]).volume > 1e-5:
                    raise SpecError(f"Connectors collide: {panel.id}/{layout.panels[k].id}")
    if not export_step(Compound(children=assembly), path / "assembly.step"):
        raise SpecError("Assembly STEP export failed")
    write_svg(layout.spec, path / "assembly.svg", layout)
    write_pdf(layout.spec, path / "assembly.pdf", layout)
    spec = layout.spec
    (path / "ATTRIBUTION.txt").write_text(_attribution(spec.board))
    covered = union_all([p.footprint for p in layout.panels])
    manifest = {
        "schema_version": 1,
        "installation": {
            "id": spec.id,
            "status": spec.status,
            "units": "mm",
            "board": spec.board,
            "coordinate_system": spec.coordinate_system,
            "grid_origin": list(spec.grid_origin),
            "source": spec.source,
            "metadata": spec.metadata,
            "surface": mapping(spec.surface),
            "keepouts": [
                {
                    "id": k.id,
                    "geometry": mapping(k.geometry),
                    "clearance": k.clearance,
                    "uncertainty": k.uncertainty,
                    "confidence": k.confidence,
                    "source": k.source,
                    "critical": k.critical,
                }
                for k in spec.keepouts
            ],
        },
        "native_interface": {
            "board": spec.board,
            "pitch_mm": PITCH,
            "thickness_mm": board_thickness(spec.board),
            **({"mounting_hole": {
                "through_diameter_mm": 4.1,
                "head_diameter_mm": 7.2,
                "head_face_z_mm": THICKNESS,
                "pattern_pitch_mm": 56,
                "placement": "complete interior odd lattice nodes only; no split seam holes",
            }} if spec.board == "lite" else {}),
            "source": NATIVE_SOURCE,
            "author": "David D",
            "license": "CC-BY-4.0",
            "physical_fit": "requires_user_coupon_check",
        },
        "printer": {
            "name": spec.printer.name,
            "width": spec.printer.width,
            "depth": spec.printer.depth,
            "margin": spec.printer.margin,
            "usable_bed": mapping(spec.printer.usable_bed),
        },
        "coverage": {
            "usable_area_mm2": layout.usable.area,
            "covered_footprint_area_mm2": covered.area,
            "unused_area_mm2": layout.unused.area,
            "functional_cells": sum(len(p.cells) for p in layout.panels),
        },
        "panels": panel_records,
        "edge_interfaces": [asdict(e) for e in {e.id: e for p in layout.panels for e in p.edges}.values()],
        "joints": [
            {
                "id": j.id,
                "male_panel": j.male_panel,
                "female_panel": j.female_panel,
                "x": j.x,
                "y": j.y,
                "angle": j.angle,
                "style": spec.joints.style,
                "assembly_axis": _joint_assembly_axis(j, layout),
            }
            for j in layout.joints
        ],
        "joint_strategy": {
            **{k: v for k, v in asdict(spec.joints).items() if v is not None},
            "panel_pattern": (
                "brick_staggered" if spec.joints.style in {"wall", "under_desk_puzzle"} else "aligned"
            ),
            "clearance_definition": "millimetres per mating side",
            "assembly": (
                "Follow puzzle_support.assembly_steps / ASSEMBLY.md: build infill rows first, then lower "
                "complete anchor rows along -Z onto adjacent infill rows. Preassemble before mounting."
                if support is not None else
                f"Push together across the seam in XY by {ASSEMBLY_TRAVEL:g} mm until the integral latch clicks; "
                "press the exposed spring shoulder towards its relief slot to release. "
                "Connect panels within each row first, then slide complete rows together. "
                "Do not insert the final corner panel diagonally. Assemble before mounting."
                if spec.joints.style == "under_desk" else
                "Seat each male puzzle head from below along +Z, or lower its receiver over it. "
                "Build rows left-to-right, then lower each new row over the preceding row. "
                "The shallow ledge supports one thickness direction only; wall mounting is still required. "
                "Puzzle shoulders retain across the seam in XY. Assemble before mounting."
                if spec.joints.style == "wall" and any(p.edges for p in layout.panels) else
                "Seat the puzzle head through Z; the shoulders retain across the seam in XY."
            ),
            "physical_fit": "requires_coupon",
        },
        "warnings": layout.warnings,
        "mounting": (
            "native Full mounting tiles/snaps required on every panel; no integrated screw holes; "
            "user must arrange and verify tile placement and attachment"
            if spec.board == "full" else
            "native Lite screw holes included; screws and attachment user_handled"
        ),
    }
    if support is not None:
        manifest["puzzle_support"] = support
        guidance = ["# Puzzle support assembly and mounting", "", support["limitation"], ""]
        guidance += [f"{s['number']}. {s['instruction']}" for s in support["assembly_steps"]]
        guidance += ["", *[f"BLOCKER: {b}" for b in support["mounting_blockers"]]]
        (path / "ASSEMBLY.md").write_text("\n".join(guidance) + "\n")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
