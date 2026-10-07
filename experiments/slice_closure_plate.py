"""Slice experimental closure STLs with existing local P1S profiles, then audit the 3MF.

Does not contact the printer. Profiles contain machine/process/filament.json.
Example: uv run --extra dev python experiments/slice_closure_plate.py \
    --profiles opengrid/verification/connector-v2/profiles \
    --output opengrid/verification/closure-study/sliced --models <actual STLs...>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import box


def audit(package: Path, inputs: list[Path]) -> dict:
    """Inspect actual object transforms and positive-extrusion layers, not CLI exit alone."""
    records = {p.name: p for p in inputs}
    if len(records) != len(inputs):
        raise ValueError("STL basenames must be unique")
    with zipfile.ZipFile(package) as archive:
        settings = json.loads(archive.read("Metadata/project_settings.config"))
        metadata = json.loads(archive.read("Metadata/plate_1.json"))
        gcode = archive.read("Metadata/plate_1.gcode").decode()
        info = ET.fromstring(archive.read("Metadata/slice_info.config"))
        plates = info.findall("plate")
        assert len(plates) == 1, "Study must fit one build plate"
        plate = {p.attrib["key"]: p.attrib["value"] for p in plates[0].findall("metadata")}
        assert plate["support_used"] == "false" and plate["outside"] == "false"
        assert len(metadata["bbox_objects"]) == len(inputs)
        assert metadata["filament_colors"] == ["#898989"]
        assert settings["nozzle_diameter"] == ["0.4"]
        assert settings["nozzle_temperature"] == ["220"]
        assert settings["nozzle_temperature_initial_layer"] == ["220"]
        assert settings["textured_plate_temp"] == ["55"]
        assert settings["textured_plate_temp_initial_layer"] == ["55"]
        assert "; wall_loops = 4" in gcode and "; enable_support = 0" in gcode
        assert "; sparse_infill_density = 100%" in gcode
        assert "M83" in gcode, "Layer audit requires relative extrusion"
        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        build = ET.fromstring(archive.read("3D/3dmodel.model")).find("m:build", ns)
        transforms = {
            item.attrib["objectid"]: list(map(float, item.attrib["transform"].split()))
            for item in build
        }
        config = ET.fromstring(archive.read("Metadata/model_settings.config"))
        bounds, heights, transforms_by_name = {}, {}, {}
        for obj in config.findall("object"):
            md = {p.attrib["key"]: p.attrib["value"] for p in obj.findall("metadata")}
            name = md["name"]
            assert name in records, f"Unexpected object: {name}"
            transform = transforms[obj.attrib["id"]]
            mesh = trimesh.load_mesh(records[name])
            assert isinstance(mesh, trimesh.Trimesh) and mesh.is_watertight and mesh.is_volume
            assert len(mesh.split()) == 1, f"Disconnected printable: {name}"
            # 3MF stores a row-vector 3x3 followed by XYZ translation.
            vertices = mesh.vertices @ np.array(transform[:9]).reshape((3, 3))
            vertices += np.array(transform[9:12])
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            assert abs(low[2]) < 0.02, f"Floating object: {name}"
            bounds[name] = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
            heights[name] = float(high[2])
            transforms_by_name[name] = transform
            footprint = box(*bounds[name])
            assert footprint.within(box(0.5, 0.5, 255.5, 255.5)), name
            assert not footprint.intersects(box(0, 0, 18.5, 28.5)), name
        assert set(bounds) == set(records)
        names = list(bounds)
        for index, name in enumerate(names):
            for other in names[index + 1:]:
                assert not box(*bounds[name]).intersects(box(*bounds[other])), (name, other)
        # Orca's printed bbox includes brim, unlike the transformed model mesh.
        # Bound that difference by the configured brim and verify it separately.
        printed_bounds = {}
        brim = float(settings.get("brim_width", 0))
        brim_margin = brim + float(settings.get("brim_object_gap", 0)) + 0.5
        for obj in metadata["bbox_objects"]:
            name = obj["name"]
            model_box, printed_box = box(*bounds[name]), box(*obj["bbox"])
            assert printed_box.buffer(0.03).covers(model_box), name
            assert model_box.buffer(brim_margin, join_style="mitre").covers(printed_box), name
            assert printed_box.within(box(0.5, 0.5, 255.5, 255.5)), name
            assert not printed_box.intersects(box(0, 0, 18.5, 28.5)), name
            printed_bounds[name] = obj["bbox"]
        layers = {name: set() for name in names}
        extrusion_mm = {name: 0.0 for name in names}
        current, layer = None, 0
        for line in gcode.splitlines():
            if line.startswith("; printing object "):
                current = line.split("; printing object ", 1)[1].split(" id:", 1)[0]
            elif line.startswith("; stop printing object "):
                current = None
            elif line.startswith("; layer num/total_layer_count:"):
                layer = int(line.split(":")[1].split("/")[0])
            elif current in layers and line.startswith(("G1 ", "G2 ", "G3 ")):
                args = {k: float(v) for k, v in re.findall(r"([XYE])([-+]?\d*\.?\d+)", line)}
                if args.get("E", 0) > 0 and ("X" in args or "Y" in args):
                    layers[current].add(layer)
                    extrusion_mm[current] += args["E"]
        for name in names:
            expected = math.ceil((heights[name] - 0.02) / 0.2)
            assert len(layers[name]) >= expected - 1, (name, layers[name], expected)
            assert extrusion_mm[name] > 0, name
        warnings = [w.attrib for w in info.findall("plate/warning")]
        for name in archive.namelist():
            if name.startswith("Metadata/plate_1") and name.endswith(".png"):
                (package.parent / Path(name).name).write_bytes(archive.read(name))
    return {
        "package": str(package), "parts": names, "gray_filament": True,
        "support_used": False, "bed_clearance": True,
        "solid_infill": True, "walls": 4, "layer_height": 0.2,
        "layer_count": max(max(value) for value in layers.values()),
        "object_extrusion_layers": {name: sorted(value) for name, value in layers.items()},
        "object_extrusion_mm": extrusion_mm, "bounds": bounds, "heights": heights,
        "printed_bounds_including_brim": printed_bounds, "brim_width_mm": brim,
        "transforms": transforms_by_name, "plate_metadata": plate, "warnings": warnings,
        "limitations": ["Toolpaths are not a physical adhesion or strength test.",
                        "Solid infill remains anisotropic FFF, not homogeneous bulk PLA."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--orca", default="/opt/orca-slicer/bin/orca-slicer")
    parser.add_argument("--fixed-layout", action="store_true",
                        help="Use the study's labeled rows, shifted clear of the P1S exclusion")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = args.models
    if args.fixed_layout:
        staged = args.output / "positioned-models"
        staged.mkdir(exist_ok=True)
        inputs = []
        for source in args.models:
            mesh = trimesh.load_mesh(source)
            # Source study rows are already collision-free. Keep their identities
            # and order; move keys down a little to reserve an explicit brim margin.
            mesh.apply_translation((0, 24 if source.name.startswith("key-") else 28, 0))
            lo, hi = mesh.bounds
            reserved = box(lo[0], lo[1], hi[0], hi[1]).buffer(2.5, join_style="mitre")
            assert reserved.within(box(0.5, 0.5, 255.5, 255.5)), source.name
            assert not reserved.intersects(box(0, 0, 18.5, 28.5)), source.name
            target = staged / source.name
            mesh.export(target)
            inputs.append(target)
    profiles = args.output / "profiles"
    profiles.mkdir(exist_ok=True)
    for name in ("machine", "process", "filament"):
        data = json.loads((args.profiles / f"{name}.json").read_text())
        if name == "process":
            data.update({"wall_loops": "4", "sparse_infill_density": "100%",
                         "sparse_infill_pattern": "rectilinear", "enable_support": "0",
                         "outer_wall_speed": ["40", "40"],
                         "brim_width": "3", "layer_height": "0.2"})
            if args.fixed_layout:
                data.update({"brim_width": "2", "brim_type": "outer_only"})
        (profiles / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    package = args.output / "closure-study-gray-esun.3mf"
    command = ["xvfb-run", "-a", args.orca, "--load-settings",
               f"{profiles}/machine.json;{profiles}/process.json", "--load-filaments",
               str(profiles / "filament.json"), "--arrange", "0" if args.fixed_layout else "1",
               "--orient", "0",
               "--ensure-on-bed", "--slice", "0", "--outputdir", str(args.output),
               "--export-3mf", package.name, *map(str, inputs)]
    print("RUN", command, flush=True)
    subprocess.run(command, check=True, timeout=300)
    result = json.loads((args.output / "result.json").read_text())
    assert result["return_code"] == 0
    report = audit(package, inputs)
    report["fixed_layout"] = args.fixed_layout
    if args.fixed_layout:
        for source in inputs:
            lo, hi = trimesh.load_mesh(source).bounds
            actual = report["bounds"][source.name]
            assert np.allclose(actual, [lo[0], lo[1], hi[0], hi[1]], atol=0.03), source.name
        report["reserved_brim_margin_mm"] = 2.5
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
