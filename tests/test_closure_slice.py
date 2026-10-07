"""The study's print audit must reject unsafe/incomplete slicer output."""
import json
import runpy
import zipfile
from pathlib import Path

import pytest
import trimesh

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / "experiments/slice_closure_plate.py"))["audit"]


def package(tmp_path, *, x=30, color="#898989", extrusion=True, support="false", temp="220",
            brim=0):
    stl = tmp_path / "sample.stl"
    mesh = trimesh.creation.box(extents=(4, 4, 0.4))
    mesh.apply_translation((2, 2, 0.2))
    mesh.export(stl)
    config = {
        "nozzle_diameter": ["0.4"], "nozzle_temperature": [temp],
        "nozzle_temperature_initial_layer": [temp], "textured_plate_temp": ["55"],
        "textured_plate_temp_initial_layer": ["55"],
        "brim_width": str(brim),
    }
    gcode = "M83\n; wall_loops = 4\n; enable_support = 0\n; sparse_infill_density = 100%\n"
    for layer in (1, 2):
        gcode += f"; layer num/total_layer_count: {layer}/2\n"
        gcode += "; printing object sample.stl id:0\n"
        if extrusion:
            gcode += f"G1 X{x + 1} Y41 E1.2\n"
        gcode += "; stop printing object sample.stl id:0\n"
    model = (
        '<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f'<build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 {x} 40 0"/>'
        '</build></model>'
    )
    info = (
        '<config><plate><metadata key="outside" value="false"/>'
        f'<metadata key="support_used" value="{support}"/></plate></config>'
    )
    files = {
        "Metadata/project_settings.config": json.dumps(config),
        "Metadata/plate_1.json": json.dumps({
            "bbox_objects": [{"name": "sample.stl",
                              "bbox": [x-brim, 40-brim, x+4+brim, 44+brim]}],
            "filament_colors": [color],
        }),
        "Metadata/plate_1.gcode": gcode,
        "Metadata/slice_info.config": info,
        "3D/3dmodel.model": model,
        "Metadata/model_settings.config": (
            '<config><object id="1"><metadata key="name" value="sample.stl"/>'
            '</object></config>'
        ),
    }
    archive = tmp_path / "sample.3mf"
    with zipfile.ZipFile(archive, "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    return archive, [stl]


def test_study_audit_accepts_actual_mesh_and_layer_extrusion(tmp_path):
    result = AUDIT(*package(tmp_path))
    assert result["bed_clearance"]
    assert result["object_extrusion_layers"] == {"sample.stl": [1, 2]}
    assert result["heights"]["sample.stl"] == pytest.approx(0.4)


@pytest.mark.parametrize("invalid", [
    {"x": 254}, {"color": "#17121F"}, {"extrusion": False},
    {"support": "true"}, {"temp": "260"}, {"x": 250, "brim": 2},
])
def test_study_audit_rejects_bad_plate(tmp_path, invalid):
    with pytest.raises(AssertionError):
        AUDIT(*package(tmp_path, **invalid))


def test_study_audit_accounts_for_printed_brim_bounds(tmp_path):
    result = AUDIT(*package(tmp_path, brim=2))
    assert result["bounds"]["sample.stl"] == [30, 40, 34, 44]
    assert result["printed_bounds_including_brim"]["sample.stl"] == [28, 38, 36, 46]
    assert result["brim_width_mm"] == 2
