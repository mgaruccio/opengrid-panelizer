# Historical closure experiments — not recommended print releases

These scripts preserve earlier connector experiments for reproducibility and
regression tests. They are **not** the current `under_desk_puzzle` design, and
should not be treated as installation-ready files.

## Concepts explored

- **A:** a broad, full-thickness puzzle interface without supporting edge lips.
- **B:** a puzzle interface with raised local reinforcement.
- **C:** separate captured keys, including a squared-shoulder revision after an
  inclined design was found to cam toward its release path.

The raised reinforcement and separate-key approaches were rejected for this
application: they complicated assembly and/or intruded into the accessory-side
space. They remain experiments, not the public project's recommended direction.
The production generator preserves the native 4 mm panel envelope; these older
experiment scripts do not necessarily do so.

## Tools

- `closure_study.py`: generates the historical sample geometries.
- `closure_fem.py`: separate-body Gmsh/CalculiX contact experiments and controls.
- `slice_closure_plate.py`: local Orca CLI slicing/auditing helper; it forces four
  walls and 100% infill, so it is not a neutral comparison of arbitrary profiles.
- `render_closure_models.py` and `illustrate_closure_study.py`: optional local
  visualization tools for those samples.

Run each script with `--help` for its interface. These tools do not upload files
or start a printer. Generated models, slicer profiles, solver binaries, telemetry
and machine-specific operational scripts are intentionally not bundled.

## Evidence limits

A connected watertight solid, collision-free assembly path or successful slice
is not a force measurement. The FEM tool uses simplified elastic material and
frictionless contact; failed or incomplete solves are explicitly inconclusive.
A successful local fixture test is not a whole-board, fastener or creep model.
Historical runtime reports are not included in this public source snapshot,
so no historical numerical result is presented here as independently verified
public evidence.

The current supporting-lip investigation also timed out in its bounded contact
solves. It has no validated stiffness, sustained-load or minimum-screw rating.
Physical fit and load testing remain necessary.

## References

- [Gradient-Based Dovetail Joint Shape Optimization for Stiffness](https://gfx.cs.princeton.edu/pubs/Sun_2023_GDJ/connector_scf.pdf): useful contact/load-path ideas, but its resin specimens and assumptions are not calibration data for 4 mm FDM PLA.
- [Official openGrid board guidance](https://www.opengrid.world/guides/board/): native connectors primarily align boards; add mounting points when needed.
- [Prusa design guidance](https://help.prusa3d.com/article/modeling-with-3d-printing-in-mind_164140): orientation, tolerances and thin-feature printability.

For the current prototype, use `examples/under-desk-puzzle.yaml` through the
public `opengrid generate` CLI and follow its generated `ASSEMBLY.md`.
