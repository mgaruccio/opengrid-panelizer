"""Experimental, frictionless 3-D closure comparison; NOT a strength/retention rating.

Run without changing project dependencies::

    uv run --no-project --with gmsh python experiments/closure_fem.py \
      --left left.step --right right.step --key key.step --output study \
      --ccx /path/to/ccx_2.23

Default study: two independent meshes, E=1500/2500/3500 MPa, separation,
Z sliding and rotation about Y. Units: mm, N, MPa, radians. All bodies are
meshed separately, never fused/tied. Left outer X face is rigidly clamped;
right outer X face receives a rigid displacement/rotation, with its other
motions restrained. Only those faces are clamped, not entire bodies. This
is a local, rigid-fixture coupon test, not a full-panel or insertion model.

An optional key is entirely unsupported. A free key can make the static
problem singular, especially before clearance closes. No weak springs,
contact adjustment, preload, friction or other hidden stabilization is used.
A failed solve is inconclusive, never evidence of stiffness or retention.
Even a successful solve cannot certify that all free-key modes are stable.

Official solver: https://www.dhondt.de/ccx_2.23.tar.bz2 (local install only).
Official 2.23 manual: https://www.dhondt.de/ccx_2.23.pdf, *CONTACT PAIR,
*SURFACE BEHAVIOR, *CONTACT PRINT: face-to-face LINEAR penalty contact is
strictly compression-only; no ADJUST preserves the supplied clearances.
https://gmsh.info/doc/texinfo/gmsh.html documents OCC STEP import and meshing.
C3D10 is the default; surface-to-surface contact avoids quadratic
node-to-face contact's problematic nodal force distribution.

Public-boundary controls (actual STEP export/reimport, not mock solves)::

    ... --control monolith --cases separation --nu 0 --motion .02 \
        --mesh-sizes 2 1.5 --youngs-moduli 2500 --output monolith
    ... --control free --cases separation --output free
    ... --control contact --cases compression --nu 0 --motion .02 \
        --mesh-sizes 2 1.5 --youngs-moduli 2500 --output contact

For nu=0 the monolith small-strain axial reaction is E*48*motion/10 N.
With NLGEOM, its exact uniform axial reaction is E*48*(lambda**2-1)*lambda/2,
lambda=1+motion/10. The touching compression blocks approach the monolith
compression reaction as penalty stiffness increases. Separated blocks in
tension must have essentially zero reaction. Outputs
include original STEP controls, Gmsh meshes/logs, .inp/.dat/.frd/.sta/.cvg,
solver stdout, and results.json. Use a fresh output directory for each study.
Delete the output directory to clean up; nothing is installed system-wide.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path

Vec = tuple[float, float, float]
# CalculiX tetrahedral face numbers; midside nodes follow the same faces.
FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
GMSH_TO_C3D10 = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)


@dataclass
class Body:
    name: str
    nodes: dict[int, Vec]
    elements: dict[int, tuple[int, ...]]
    surface: list[tuple[int, int]]
    volume: float


def external_faces(elements: dict[int, tuple[int, ...]]) -> list[tuple[int, int]]:
    counts = Counter(tuple(sorted(nodes[i] for i in face))
                     for nodes in elements.values() for face in FACES)
    if any(count > 2 for count in counts.values()):
        raise ValueError("Non-manifold tetrahedral mesh")
    return [(tag, side) for tag, nodes in elements.items()
            for side, face in enumerate(FACES, 1)
            if counts[tuple(sorted(nodes[i] for i in face))] == 1]


def fixture_nodes(body: Body, x: float | None, left: bool) -> list[int]:
    if x is None:
        x = (min if left else max)(p[0] for p in body.nodes.values())
    nodes = [tag for tag, p in body.nodes.items() if abs(p[0] - x) < 1e-6]
    # A fixture must contain a real planar mesh face, not a corner/edge.
    selected = set(nodes)
    if not any(all(body.elements[e][i] in selected for i in FACES[s - 1])
               for e, s in body.surface):
        raise ValueError(f"{body.name}: no exterior face at fixture X={x}; set --left-x/--right-x")
    return sorted(nodes)


def control_steps(gmsh, control: str, output: Path) -> dict[str, Path]:
    directory = output / "control-step"
    directory.mkdir()
    boxes = {"MONOLITH": (-5.0, 10.0)} if control == "monolith" else {
        "LEFT": (-5.0, 4.9 if control == "free" else 5.0),
        "RIGHT": (0.1 if control == "free" else 0.0, 4.9 if control == "free" else 5.0),
    }
    paths = {}
    for name, (start, length) in boxes.items():
        gmsh.model.add(name)
        gmsh.model.occ.addBox(start, -6, 0, length, 12, 4)
        gmsh.model.occ.synchronize()
        paths[name] = directory / f"{name.lower()}.step"
        gmsh.write(str(paths[name]))
        gmsh.model.remove()
    return paths


def check_solids(gmsh, paths: dict[str, Path]) -> None:
    """Reject overlap rather than hiding it with initial contact adjustment."""
    gmsh.model.add("initial-geometry")
    volumes = {}
    for name, path in paths.items():
        imported = gmsh.model.occ.importShapes(str(path))
        solids = [entity for entity in imported if entity[0] == 3]
        if len(solids) != 1:
            raise ValueError(f"{name}: expected exactly one STEP solid, found {len(solids)}")
        volumes[name] = solids
    for a, b in combinations(volumes, 2):
        common, _ = gmsh.model.occ.intersect(gmsh.model.occ.copy(volumes[a]),
                                            gmsh.model.occ.copy(volumes[b]))
        overlap = sum(gmsh.model.occ.getMass(d, tag) for d, tag in common if d == 3)
        if overlap > 1e-6:
            raise ValueError(f"Initial {a}/{b} overlap: {overlap:.9g} mm^3; no adjustment allowed")
        if common:
            gmsh.model.occ.remove(common, recursive=True)
    gmsh.model.remove()


def mesh_bodies(gmsh, paths: dict[str, Path], h: float, order: int, output: Path) -> list[Body]:
    bodies = []
    node_offset = element_offset = 0
    for name, path in paths.items():
        gmsh.model.add(name)
        gmsh.logger.start()
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        volume = sum(gmsh.model.occ.getMass(d, t) for d, t in gmsh.model.getEntities(3))
        for option, value in {
            "Mesh.MeshSizeMin": h, "Mesh.MeshSizeMax": h,
            "Mesh.MeshSizeFromCurvature": 0, "Mesh.ElementOrder": order,
            "Mesh.Algorithm3D": 1, "Mesh.Optimize": 1,
        }.items():
            gmsh.option.setNumber(option, value)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(output / f"{name.lower()}.msh"))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = {int(tag) + node_offset: tuple(float(v) for v in coords[3*i:3*i+3])
                 for i, tag in enumerate(tags)}
        types, element_tags, connectivity = gmsh.model.mesh.getElements(3)
        elements = {}
        for kind, ids, flat in zip(types, element_tags, connectivity):
            if int(kind) != (11 if order == 2 else 4):
                raise ValueError(f"Unexpected Gmsh element type {kind}")
            width = 10 if order == 2 else 4
            permutation = GMSH_TO_C3D10 if order == 2 else range(4)
            for i, tag in enumerate(ids):
                elements[int(tag) + element_offset] = tuple(
                    int(flat[i*width+j]) + node_offset for j in permutation)
        if not elements:
            raise ValueError(f"{name}: empty mesh")
        bodies.append(Body(name, nodes, elements, external_faces(elements), volume))
        node_offset = max(nodes)
        element_offset = max(elements)
        (output / f"{name.lower()}-gmsh.log").write_text("\n".join(gmsh.logger.get()))
        gmsh.logger.stop()
        gmsh.model.remove()
    return bodies


def imposed_displacement(point: Vec, center: Vec, case: str, motion: float, angle: float) -> Vec:
    if case == "bend":
        x, _, z = (point[i] - center[i] for i in range(3))
        return ((math.cos(angle)-1)*x + math.sin(angle)*z, 0.0,
                -math.sin(angle)*x + (math.cos(angle)-1)*z)
    if case == "zslide":
        return (0.0, 0.0, motion)
    return (-motion if case == "compression" else motion, 0.0, 0.0)


def write_deck(bodies: list[Body], output: Path, args, young: float, h: float, case: str) -> dict:
    left, right = bodies[0], bodies[0] if len(bodies) == 1 else bodies[1]
    fixed = fixture_nodes(left, args.left_x, True)
    driven = fixture_nodes(right, args.right_x, False)
    if set(fixed) & set(driven):
        raise ValueError("Fixture faces overlap")
    center = tuple((min(right.nodes[n][i] for n in driven) +
                    max(right.nodes[n][i] for n in driven))/2 for i in range(3))
    prescribed = {n: imposed_displacement(right.nodes[n], center, case, args.motion, args.angle)
                  for n in driven}
    lines = ["*HEADING", "Experimental separate-body frictionless closure coupon; mm,N,MPa",
             "** No ADJUST, tie, weak springs, key support, preload or friction.", "*NODE,NSET=NALL"]
    for body in bodies:
        lines.extend(f"{n}," + ",".join(f"{x:.12g}" for x in p) for n, p in body.nodes.items())
    for body in bodies:
        width = len(next(iter(body.elements.values())))
        lines.append(f"*ELEMENT,TYPE=C3D{width},ELSET={body.name}")
        lines.extend(f"{e}," + ",".join(map(str, nodes)) for e, nodes in body.elements.items())
        lines.extend([f"*SOLID SECTION,ELSET={body.name},MATERIAL=PLA"])
        lines.append(f"*SURFACE,NAME={body.name}_SURF,TYPE=ELEMENT")
        lines.extend(f"{e},S{s}" for e, s in body.surface)
    lines += ["*ELSET,ELSET=EALL", ",".join(b.name for b in bodies),
              "*MATERIAL,NAME=PLA", "*ELASTIC", f"{young:.12g},{args.nu:.12g}"]
    penalty = args.penalty_factor * young / h  # N/mm^3, not MPa.
    if len(bodies) > 1:
        lines += ["*SURFACE INTERACTION,NAME=FRICTIONLESS",
                  "*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR", f"{penalty:.12g}"]
        for master, slave in combinations(bodies, 2):
            lines += ["*CONTACT PAIR,INTERACTION=FRICTIONLESS,TYPE=SURFACE TO SURFACE",
                      f"{slave.name}_SURF,{master.name}_SURF"]
    for name, nodes in [("FIXED", fixed), ("DRIVEN", driven)]:
        lines.append(f"*NSET,NSET={name}")
        lines.extend(",".join(map(str, nodes[i:i+16])) for i in range(0, len(nodes), 16))
    lines += ["*BOUNDARY", "FIXED,1,3,0", "*STEP,NLGEOM,INC=200", "*STATIC",
              "0.05,1,0.00001,0.1", "*BOUNDARY"]
    for n, displacement in prescribed.items():
        lines.extend(f"{n},{d},{d},{u:.12g}" for d, u in enumerate(displacement, 1))
    lines += ["*NODE PRINT,NSET=NALL,FREQUENCY=1", "U",
              "*NODE PRINT,NSET=FIXED", "RF", "*NODE PRINT,NSET=DRIVEN", "RF",
              "*NODE FILE", "U,RF", "*EL FILE", "S,E",
              "*EL PRINT,ELSET=EALL,TOTALS=ONLY", "ELSE"]
    if len(bodies) > 1:
        lines += ["*CONTACT PRINT,TOTALS=YES", "CDIS,CSTR,CELS,CNUM"]
    lines += ["*END STEP"]
    (output / "job.inp").write_text("\n".join(lines) + "\n")
    return {"fixed_nodes": fixed, "driven_nodes": driven, "rotation_center_mm": center,
            "prescribed": prescribed, "penalty_N_per_mm3": penalty}


def dat_tables(text: str) -> list[tuple[str, float, list[list[float]]]]:
    """Read named .dat blocks, preserving output times (including partial solves)."""
    blocks = []
    current = None
    for line in text.splitlines():
        match = re.search(r"^(.*?)\s+(?:and |for )?time\s+([\d.EeDd+\-]+)\s*$", line.strip())
        if match:
            current = (match[1].strip().lower(), float(match[2].replace("D", "E")), [])
            blocks.append(current)
        elif line.strip() and current is not None:
            try:
                # Fortran omits E when a fixed-width exponent needs three digits.
                row = [float(re.sub(r"(?<=\d)([+-]\d{3})$", r"E\1", x.replace("D", "E")))
                       for x in line.split()]
                if all(math.isfinite(value) for value in row):
                    current[2].append(row)
            except ValueError:
                current = None
    return blocks


def norm(vector) -> float:
    return math.sqrt(sum(v*v for v in vector))


def summarize(output: Path, bodies: list[Body], fixture: dict, case: str, args,
              returncode: int | None, timed_out: bool) -> dict:
    stdout = (output / "solver.log").read_text(errors="replace")
    dat = output / "job.dat"
    tables = dat_tables(dat.read_text(errors="replace") if dat.exists() else "")
    final_time = max((t for _, t, _ in tables), default=0.0)
    last = [(name, rows) for name, t, rows in tables if abs(t-final_time) < 1e-8]
    def nodes(prefix, set_name):
        return {int(row[0]): row[1:4] for name, rows in last
                if name.startswith(prefix) and f"set {set_name.lower()}" in name
                for row in rows if len(row) == 4}
    displacements = nodes("displacements", "NALL")
    left_rf, right_rf = nodes("forces", "FIXED"), nodes("forces", "DRIVEN")
    left_force = [sum(row[i] for row in left_rf.values()) for i in range(3)]
    right_force = [sum(row[i] for row in right_rf.values()) for i in range(3)]
    imbalance = norm([a+b for a, b in zip(left_force, right_force)])
    reaction_scale = max(norm(left_force), norm(right_force), 1e-6)
    coordinates = {n: p for body in bodies for n, p in body.nodes.items()}
    center = fixture["rotation_center_mm"]
    moment_y = sum((coordinates[n][2]+displacements.get(n, [0]*3)[2]-center[2])*f[0] -
                   (coordinates[n][0]+displacements.get(n, [0]*3)[0]-center[0])*f[2]
                   for n, f in right_rf.items())
    error = max((abs(displacements.get(n, [math.inf]*3)[i] - u[i])
                 for n, u in fixture["prescribed"].items() for i in range(3)), default=math.inf)
    max_u = max((norm(u) for u in displacements.values()), default=None)
    expected_max = max(norm(u) for u in fixture["prescribed"].values())
    complete = (returncode == 0 and not timed_out and "Job finished" in stdout
                and "*ERROR" not in stdout and final_time >= 1-1e-6
                and len(displacements) == len(coordinates)
                and len(left_rf) == len(fixture["fixed_nodes"])
                and len(right_rf) == len(fixture["driven_nodes"]))
    balance_ok = imbalance <= 1e-5 + .01*reaction_scale
    bounded = max_u is not None and max_u < max(1.0, 20*expected_max)
    valid = complete and balance_ok and error < 1e-6 and bounded
    signal = moment_y if case == "bend" else right_force[2 if case == "zslide" else 0]
    imposed = args.angle if case == "bend" else (-args.motion if case == "compression" else args.motion)
    status = "completed" if valid else "inconclusive_failed_or_unrestrained"
    if valid and abs(signal) < 1e-6:
        status = "free_motion_at_tested_stroke"
    scalars = {name: rows[-1][-1] for name, rows in last if rows and len(rows[-1]) == 1}
    residuals = re.findall(r"largest residual force\s*[=:]?\s*([\d.Ee+\-]+)", stdout)
    cvg = output / "job.cvg"
    convergence = []
    if cvg.exists():
        for line in cvg.read_text(errors="replace").splitlines():
            values = line.split()
            if len(values) == 9 and all(v.isdigit() for v in values[:5]):
                convergence.append([float(v) for v in values])
    normal_motion = [abs(row[2]) for name, rows in last
                     if name.startswith("relative contact displacement")
                     for row in rows if len(row) == 5]
    pressure = [row[2] for name, rows in last if name.startswith("contact stress")
                for row in rows if len(row) == 5]
    internal = scalars.get("total internal energy for set eall", 0)
    penalty_energy = scalars.get("total contact spring energy", 0)
    return {"status": status, "solver_returncode": returncode, "timed_out": timed_out,
            "last_output_time": final_time, "completed": complete,
            "force_balance_ok": balance_ok, "reaction_imbalance_N": imbalance,
            "reaction_imbalance_relative": imbalance/reaction_scale,
            "fixed_reaction_N": left_force if left_rf else None,
            "driven_reaction_N": right_force if right_rf else None,
            "driven_moment_y_Nmm": moment_y if right_rf else None,
            "max_displacement_mm": max_u, "prescribed_displacement_error_mm": error if math.isfinite(error) else None,
            "secant_N_per_mm_or_Nmm_per_rad": signal/imposed if valid else None,
            "last_reported_force_residual_N": float(residuals[-1]) if residuals else None,
            "last_residual_force_percent": convergence[-1][5] if convergence else None,
            "last_displacement_correction_percent": convergence[-1][6] if convergence else None,
            "max_abs_normal_contact_displacement_mm": max(normal_motion, default=None),
            "max_contact_pressure_MPa": max(pressure, default=None),
            "penalty_energy_fraction": penalty_energy/(internal+penalty_energy)
            if internal+penalty_energy > 1e-12 else None,
            "stabilization_energy_Nmm": 0.0,
            "final_scalar_output": scalars,
            "displacements_mm": displacements, "fixed_nodal_reactions_N": left_rf,
            "driven_nodal_reactions_N": right_rf,
            "unsupported_key_warning": len(bodies) == 3}


def run_case(output: Path, bodies: list[Body], args, young: float, h: float, case: str) -> dict:
    output.mkdir()
    fixture = write_deck(bodies, output, args, young, h, case)
    started = time.monotonic()
    timed_out = False
    returncode = None
    with (output / "solver.log").open("w") as log:
        try:
            completed = subprocess.run([args.ccx, "-i", "job"], cwd=output, check=False,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "OMP_NUM_THREADS": "1",
                                            "CCX_NPROC_RESULTS": "1"}, timeout=args.timeout)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    result = summarize(output, bodies, fixture, case, args, returncode, timed_out)
    result.update({"mesh_size_mm": h, "young_MPa": young, "case": case,
                   "directory": str(output), "elapsed_seconds": time.monotonic()-started,
                   "fixture": {k: v for k, v in fixture.items() if k != "prescribed"}})
    return result


def sensitivities(runs: list[dict]) -> list[dict]:
    comparisons = []
    for case, young in sorted({(r["case"], r["young_MPa"]) for r in runs}):
        group = sorted((r for r in runs if r["case"] == case and r["young_MPa"] == young),
                       key=lambda r: r["mesh_size_mm"], reverse=True)
        for coarse, fine in pairwise(group):
            a, b = (r["secant_N_per_mm_or_Nmm_per_rad"] for r in [coarse, fine])
            comparisons.append({"case": case, "young_MPa": young,
                                "coarse_h_mm": coarse["mesh_size_mm"], "fine_h_mm": fine["mesh_size_mm"],
                                "relative_change": abs(a-b)/max(abs(b), 1e-6) if a is not None and b is not None else None,
                                "note": "Two levels are a sensitivity check, not proof of mesh convergence; penalty also scales as E/h."})
    return comparisons


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", type=Path)
    p.add_argument("--right", type=Path)
    p.add_argument("--key", type=Path)
    p.add_argument("--control", choices=["free", "contact", "monolith"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ccx", default="ccx")
    p.add_argument("--mesh-sizes", type=float, nargs="+", default=[1.2, .8])
    p.add_argument("--order", type=int, choices=[1, 2], default=2)
    p.add_argument("--youngs-moduli", type=float, nargs="+", default=[1500, 2500, 3500])
    p.add_argument("--nu", type=float, default=.35)
    p.add_argument("--cases", nargs="+", choices=["separation", "zslide", "bend", "compression"],
                   default=["separation", "zslide", "bend"])
    p.add_argument("--motion", type=float, default=.2, help="Positive translation magnitude in mm")
    p.add_argument("--angle", type=float, default=.02, help="Bend rotation about Y in radians")
    p.add_argument("--left-x", type=float, help="Clamp plane; default left-body minimum X")
    p.add_argument("--right-x", type=float, help="Driven plane; default right-body maximum X")
    p.add_argument("--penalty-factor", type=float, default=50, help="Normal stiffness = factor*E/h, N/mm^3")
    p.add_argument("--timeout", type=float, default=90, help="Seconds per solver process; failure is retained")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if bool(args.control) == bool(args.left or args.right or args.key):
        p.error("Choose --control OR separate --left and --right STEP solids [--key]")
    if not args.control and not (args.left and args.right):
        p.error("Both --left and --right are required")
    positive = [*args.mesh_sizes, *args.youngs_moduli, args.motion, args.angle,
                args.penalty_factor, args.timeout]
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not -1 < args.nu < .5:
        p.error("Mesh sizes, moduli, motion, angle, penalty and timeout must be positive finite; -1 < nu < .5")
    if len(set(args.mesh_sizes)) != len(args.mesh_sizes) or len(set(args.youngs_moduli)) != len(args.youngs_moduli):
        p.error("Duplicate mesh sizes or moduli")
    executable = shutil.which(args.ccx)
    if not executable:
        p.error("CalculiX executable not found; supply --ccx (no automatic installation)")
    args.ccx = str(Path(executable).resolve())
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a fresh/empty --output directory; existing results are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    report = {"study": "Frictionless contact coupon; no ranking or calibrated strength prediction",
              "units": "mm, N, MPa, radians", "gmsh_version": gmsh.__version__,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "limitations": ["Isotropic approximate PLA; no FDM anisotropy, plasticity, damage or fatigue.",
                              "No friction, interference-fit preload, insertion history or initial contact adjustment.",
                              "Rigid outer-X-face fixtures; local coupons, not whole panels or a 2x2 assembly.",
                              "Free keys are unsupported and can be singular; convergence does not certify key stability.",
                              "Secants include clearance travel; free motion at one stroke is not proof of unlimited release.",
                              "All external faces are contact candidates; face-to-face pairing updates each increment.",
                              "Penalty compliance and faceting affect results; inspect .dat contact overlaps and .cvg residuals.",
                              "No stabilization; failed/incomplete or force-imbalanced solves have no stiffness result.",
                              "Physical printed fit and bidirectional load checks remain necessary."],
              "runs": []}
    try:
        paths = control_steps(gmsh, args.control, args.output) if args.control else {
            "LEFT": args.left.resolve(), "RIGHT": args.right.resolve(),
            **({"KEY": args.key.resolve()} if args.key else {})}
        report["inputs"] = {name: str(path) for name, path in paths.items()}
        check_solids(gmsh, paths)
        for h in args.mesh_sizes:
            mesh_dir = args.output / f"h-{h:g}"
            mesh_dir.mkdir()
            bodies = mesh_bodies(gmsh, paths, h, args.order, mesh_dir)
            report.setdefault("meshes", []).append({"h_mm": h, "bodies": [
                {"name": b.name, "nodes": len(b.nodes), "elements": len(b.elements),
                 "external_faces": len(b.surface), "volume_mm3": b.volume} for b in bodies]})
            for young in args.youngs_moduli:
                for case in args.cases:
                    result = run_case(mesh_dir / f"E-{young:g}-{case}", bodies, args, young, h, case)
                    report["runs"].append(result)
                    print(f"h={h:g} E={young:g} {case}: {result['status']}", flush=True)
                    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["mesh_sensitivity"] = sensitivities(report["runs"])
    except Exception as exc:  # noqa: BLE001 -- Gmsh raises Exception; retain failed-run evidence.
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        gmsh.finalize()
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if "error" not in report and all(r["secant_N_per_mm_or_Nmm_per_rad"] is not None
                                             for r in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
