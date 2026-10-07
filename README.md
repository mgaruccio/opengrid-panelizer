# openGrid Panelizer

Turn a measured surface and keepouts into printer-sized **openGrid Lite** panels,
with native accessory sockets, mounting holes, and optional inter-panel joints.

**Experimental mechanical designs.** Automated geometry and assembly checks do
not establish printed fit, sustained load capacity, or a safe screw count. Print
a fit coupon and test with your actual accessories before making a large board.
This is an independent project, not an official openGrid generator.

## What it does

- Uses the native **28 mm pitch and 4 mm thickness** of openGrid Lite.
- Preserves complete native socket cavities and the native mounting-hole profile.
- Handles polygonal surfaces, keepouts, printer-bed limits and excluded bed areas.
- Exports individual STEP/STL panels, assembly STEP/SVG/PDF and a JSON manifest.
- Offers puzzle joints, mounting-side support lips, and experimental clip joints.
- Blocks printable export of draft or unresolved installations.

It does not infer obstacles from a desk model, select suitable screws, certify
loads, control a printer, or solve the globally optimal mounting arrangement.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.12 or 3.13:

```sh
git clone https://github.com/mgaruccio/opengrid-panelizer.git
cd opengrid-panelizer
uv sync --extra dev --frozen

# Synthetic ready-to-generate puzzle pair.
uv run opengrid preview --spec examples/fit-coupon.yaml --output verification/preview
uv run opengrid generate --spec examples/fit-coupon.yaml --output verification/pair

# Preview an unresolved example: deliberately produces no printable files.
uv run opengrid preview --spec examples/draft-installation.yaml --output verification/draft
```

Output directories must be new or empty. `generate` refuses the draft example;
do not mark a real installation ready until its geometry and keepouts have been
measured. The examples are synthetic, not claims about any particular desk.

## Joint choices

| `joints.style` | Interface | Assembly |
|---|---|---|
| `puzzle` | Through-thickness puzzle outline, no supporting lip | Seat through Z |
| `wall` | Wider puzzle keys plus one-sided, mounting-side lips; brick stagger | Follow the generated seating directions |
| `under_desk_puzzle` | Puzzle keys plus independently oriented one-sided lips; brick stagger | Preassemble infill rows, then lower anchor rows onto them |
| `under_desk` | Experimental spring clips with captured lips | Join within rows, then slide complete rows together |
| `none` | No inter-panel connectors | Independently mount panels |

All styles retain the native 4 mm envelope. **Z=0 is the mounting/print-bed face;
Z=4 is the accessory face and screw-head recess side.** Do not flip native socket
geometry to reverse a lip. Supporting lips may be omitted where there is
insufficient safe stock; read the manifest warnings.

For the puzzle-and-lip under-desk prototype:

```sh
uv run opengrid generate --spec examples/under-desk-puzzle.yaml --output verification/puzzle-rows
```

Read its generated `ASSEMBLY.md` and `manifest.json`. Build the entire board before
mounting: infill rows first, then lower the anchor rows onto their tongues. Keep
all accessory faces aligned. The manifest identifies direct-fastening panels and
**indirect-support candidates**, not a safe number of screws. A candidate must
have actual north **and** south support lips into locally fastened panels with
native holes. Boundaries, fragmented panels and incomplete support paths require
direct fastening; missing native mounting holes are explicit blockers.

The example's candidate selection is a conservative row-layout heuristic—not
structural optimization or proof against tipping, creep, or fastener pullout.
Whole-board preassembly is allowed; inserting infill after fixing both adjacent
anchor rows to a desk is not the documented assembly method.

`clearance` is millimetres **per mating side**. Supporting lips require 0–0.10 mm;
zero is a calibration experiment, not a guaranteed sliding fit. Start with the
example clearance and verify it on your printer. Flat printing with a 0.4 mm
nozzle and 0.2 mm layers is the development process; inspect the actual sliced
lips and mating shoulders rather than relying only on a slicer's success code.

## Outputs and customization

Copy an example YAML, change its surface and printer constraints, add measured
keepouts, then preview before generating. `examples/irregular.yaml` demonstrates
an irregular boundary and keepouts. All dimensions are millimetres.

Each export includes an `ATTRIBUTION.txt`; keep it with redistributed CAD files.
The manifest records panels, native mounting holes, joint/lip ownership, print
placements, coverage and warnings. Complete interior mounting nodes follow the
native 56 mm pattern; clipped or split holes are omitted. Hardware selection and
attachment engineering remain the installer's responsibility.

The Python entry point `opengrid.generate_installation(...)` returns a layout;
CAD export is separate. Automatic mounting constraints are not implemented and
nonempty mounting constraints are rejected rather than silently ignored.

## Development and verification

```sh
uv run pytest -q
uv run ruff check src tests
uv build
```

Tests exercise public CLI exports, reload actual STEP/STL solids, check native
cavities and complete rigid assembly sequences. Optional CalculiX solver tests
skip when their external runtime is unavailable. The `experiments/` directory
contains historical comparison tools and **rejected concepts**, not recommended
printable releases; see [the experiment notes](experiments/CLOSURE_STUDY.md).

The current supporting-lip contact analysis did not converge within its bounded
runs. No stiffness, load rating or long-term PLA result is claimed from it. Check
printed fit, accessory insertion, whitening/cracking, repeated assembly and
representative sustained loads before relying on a reduced-fastener layout.

## License and credits

- Original software: **MIT** ([LICENSE](LICENSE)).
- Native/derived CAD and generated panel designs: **CC BY 4.0**; credit
  **David D / openGrid Lite** and retain the generated attribution.
- [GridFlock](https://github.com/yawkat/GridFlock) inspired the puzzle-segmentation
  approach; it is not bundled or required.

See [NOTICE.md](NOTICE.md) and
[native asset attribution](src/opengrid/assets/native/ATTRIBUTION.md) for sources,
changes and license boundaries. No upstream endorsement is implied.
