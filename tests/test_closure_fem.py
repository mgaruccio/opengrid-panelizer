"""Focused checks; set CLOSURE_FEM_CCX to enable real public-CLI solver controls.

    CLOSURE_FEM_CCX=/path/to/ccx uv run --no-project --with gmsh --with pytest \
      python -m pytest -q tests/test_closure_fem.py

No project dependency changes. Solver tests use actual STEP export/import,
Gmsh C3D10 meshes and CalculiX (not mocked contact). Temporary output is
managed by pytest. Retain standalone CLI output when collecting study evidence.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "closure_fem.py"
spec = importlib.util.spec_from_file_location("closure_fem", SCRIPT)
fem = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fem
spec.loader.exec_module(fem)


def test_external_faces_exclude_shared_face():
    elements = {1: (1, 2, 3, 4), 2: (1, 3, 2, 5)}
    assert len(fem.external_faces(elements)) == 6
    assert (1, 1) not in fem.external_faces(elements)
    assert (2, 1) not in fem.external_faces(elements)


def test_fixture_requires_face_not_single_corner():
    elements = {1: (1, 2, 3, 4)}
    body = fem.Body("LEFT", {1: (0, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1),
                              4: (1, 0, 0)}, elements, fem.external_faces(elements), 1/6)
    assert fem.fixture_nodes(body, None, True) == [1, 2, 3]
    with pytest.raises(ValueError, match="no exterior face"):
        fem.fixture_nodes(body, None, False)


def test_bending_is_rigid_rotation_not_shear():
    point, center = (5, 0, 4), (5, 0, 2)
    u = fem.imposed_displacement(point, center, "bend", .2, .1)
    assert u[0] == pytest.approx(2*math.sin(.1))
    assert math.dist(tuple(point[i]+u[i] for i in range(3)), center) == pytest.approx(2)


def test_dat_parser_preserves_time_and_rejects_nonfinite_rows():
    tables = fem.dat_tables("""
 displacements (vx,vy,vz) for set NALL and time  0.5000000E+00

 1 1.000D-01 0 0
 displacements (vx,vy,vz) for set NALL and time  0.1000000E+01

 1 2.000D-01 1.234567-174 -2.345678-172
 2 nan 0 0
 total internal energy for set EALL and time  0.1000000E+01
 2.5
""")
    assert [t for _, t, _ in tables] == [.5, 1, 1]
    assert tables[1][2] == [[1, .2, 1.234567e-174, -2.345678e-172]]
    assert tables[2][2] == [[2.5]]


def test_incomplete_zero_reaction_is_not_a_free_mechanism_result(tmp_path):
    elements = {1: (1, 2, 3, 4)}
    body = fem.Body("MONOLITH", {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0),
                                  4: (0, 0, 1)}, elements, [], 1/6)
    (tmp_path / "solver.log").write_text("*ERROR: no convergence\n")
    fixture = {"fixed_nodes": [1], "driven_nodes": [2], "rotation_center_mm": (1, 0, 0),
               "prescribed": {2: (.2, 0, 0)}}
    result = fem.summarize(tmp_path, [body], fixture, "separation",
                           fem.parser().parse_args(["--output", str(tmp_path)]), 0, False)
    assert result["secant_N_per_mm_or_Nmm_per_rad"] is None
    assert result["status"] == "inconclusive_failed_or_unrestrained"
    json.dumps(result, allow_nan=False)


def real_cli(output, *arguments):
    ccx = os.environ.get("CLOSURE_FEM_CCX")
    if not ccx:
        pytest.skip("Set CLOSURE_FEM_CCX for real Gmsh/CalculiX public-boundary controls")
    pytest.importorskip("gmsh")
    process = subprocess.run([sys.executable, str(SCRIPT), "--ccx", ccx, "--output", str(output),
                              "--timeout", "15", "--youngs-moduli", "2500", *arguments],
                             text=True, capture_output=True, timeout=50, check=False)
    report = json.loads((output / "results.json").read_text())
    assert process.returncode == 0, (process.stdout, process.stderr, report.get("error"))
    for run in report["runs"]:
        assert run["completed"] and run["force_balance_ok"]
        assert run["last_residual_force_percent"] is not None
        assert run["prescribed_displacement_error_mm"] < 1e-6
        assert run["stabilization_energy_Nmm"] == 0
        directory = Path(run["directory"])
        assert all((directory / ("job" + ext)).is_file() for ext in [".inp", ".dat", ".frd", ".sta", ".cvg"])
    return report


def test_real_monolith_matches_finite_strain_analytic_bar(tmp_path):
    report = real_cli(tmp_path / "monolith", "--control", "monolith", "--cases", "separation",
                      "--nu", "0", "--motion", ".02", "--mesh-sizes", "2", "1.5")
    stretch = 1 + .02/10
    force = 2500*48*(stretch**2-1)*stretch/2
    for run in report["runs"]:
        assert run["driven_reaction_N"][0] == pytest.approx(force, rel=1e-5)
    assert report["mesh_sensitivity"][0]["relative_change"] < 1e-5


def test_real_free_bodies_are_not_bonded(tmp_path):
    report = real_cli(tmp_path / "free", "--control", "free", "--mesh-sizes", "2")
    for run in report["runs"]:
        assert run["status"] == "free_motion_at_tested_stroke"
        assert abs(run["secant_N_per_mm_or_Nmm_per_rad"]) < 1e-5


def test_real_contact_carries_compression_but_not_tension(tmp_path):
    output = tmp_path / "contact"
    report = real_cli(output, "--control", "contact", "--cases", "compression",
                      "--nu", "0", "--motion", ".02", "--mesh-sizes", "2", "1.5")
    for run in report["runs"]:
        # Includes penalty compliance and finite compression strain.
        assert run["driven_reaction_N"][0] == pytest.approx(-240, rel=.01)
        assert run["max_contact_pressure_MPa"] > 4
        assert 0 < run["max_abs_normal_contact_displacement_mm"] < .001
        assert 0 < run["penalty_energy_fraction"] < .01
        deck = (Path(run["directory"]) / "job.inp").read_text()
        assert "TYPE=SURFACE TO SURFACE" in deck
        assert "ADJUST=" not in deck and "*TIE" not in deck and "*SPRING" not in deck
    assert report["mesh_sensitivity"][0]["relative_change"] < .01
    # Exercise the requested separate-STEP public input path as well as controls.
    pull = real_cli(tmp_path / "pull", "--left", str(output / "control-step/left.step"),
                    "--right", str(output / "control-step/right.step"),
                    "--cases", "separation", "--mesh-sizes", "2")
    assert pull["runs"][0]["status"] == "free_motion_at_tested_stroke"


def test_real_initial_overlap_is_rejected(tmp_path):
    ccx = os.environ.get("CLOSURE_FEM_CCX")
    if not ccx:
        pytest.skip("Set CLOSURE_FEM_CCX for real STEP checks")
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        paths = fem.control_steps(gmsh, "free", tmp_path)
        with pytest.raises(ValueError, match="overlap"):
            fem.check_solids(gmsh, {"LEFT": paths["LEFT"], "RIGHT": paths["LEFT"]})
    finally:
        gmsh.finalize()
