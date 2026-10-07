# openGrid Panelizer

Turn a measured surface and keepouts into printer-sized **openGrid Lite or Full**
panels, with native accessory sockets and optional inter-panel joints.
Lite includes native screw holes; Full uses separate native mounting tiles/snaps.

**Experimental mechanical designs.** Automated geometry and assembly checks do
not establish printed fit, sustained load capacity, or a safe screw count. Print
a fit coupon and test with your actual accessories before making a large board.
This is an independent project, not an official openGrid generator.

## What it does

- Uses native **28 mm pitch**, **4 mm Lite** or **6.8 mm Full** thickness.
- Preserves each family's complete native socket cavities (Full is not thickened Lite).
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

`generate` defaults to up to four fresh-process panel builds at a time (bounded
by CPU count). Use `--jobs 2` to choose concurrency, or `--jobs 1` for the original
serial, in-process path. Output ordering and geometry are unchanged. The Python
`export_layout(layout, output, jobs=1)` API stays serial by default; when opting
into `jobs > 1`, call it from an importable script guarded by
`if __name__ == "__main__":` as required by multiprocessing spawn.

## Board selection

Set `installation.board: full` in YAML/JSON, or pass `board="full"` to
`opengrid.generate_installation(...)`. Omit it (or use `lite`) for unchanged Lite
behavior. Full uses the official 6.8 mm socket geometry and **no integrated screw
holes**. Arrange and verify separate native Full mounting tiles/snaps on **every
panel**; tiles/snaps are not generated or positioned here. Full under-desk puzzle
manifests do not approve indirect-only support and list tile verification as a
mounting blocker for each panel. Attachment engineering remains user-handled.

Full supports `none`, `puzzle`, `wall`, and `under_desk_puzzle`. The experimental
`under_desk` spring clip is Lite-only; Full rejects that combination.

## Joint choices

| `joints.style` | Interface | Assembly |
|---|---|---|
| `puzzle` | Through-thickness puzzle outline, no supporting lip | Seat through Z |
| `wall` | Wider puzzle keys plus one-sided, mounting-side lips; brick stagger | Follow the generated seating directions |
| `under_desk_puzzle` | Puzzle keys plus independently oriented one-sided lips; brick stagger | Preassemble infill rows, then lower anchor rows onto them |
| `under_desk` | Experimental spring clips with captured lips | Join within rows, then slide complete rows together |
| `none` | No inter-panel connectors | Independently mount panels |

Supported styles retain the selected family's native envelope. **Z=0 is the
mounting/print-bed face; Z=4 (Lite) or Z=6.8 (Full) is the accessory face.** Lite
screw-head recesses open at Z=4. Mounting-side lips stay at Z=0.1–1.9 for both
families; do not flip socket geometry to reverse a lip. Supporting lips may be
omitted where there is insufficient safe stock; read the manifest warnings.

For the puzzle-and-lip under-desk prototype:

```sh
uv run opengrid generate --spec examples/under-desk-puzzle.yaml --output verification/puzzle-rows
```

Read its generated `ASSEMBLY.md` and `manifest.json`. Build the entire board before
mounting: infill rows first, then lower the anchor rows onto their tongues. Keep
all accessory faces aligned. For Lite, the manifest identifies direct-fastening
panels and **indirect-support candidates**, not a safe number of screws. A candidate must
have actual north **and** south support lips into locally fastened panels with
native holes. Boundaries, fragmented panels and incomplete support paths require
direct fastening; missing native mounting holes are explicit blockers.

With the default `under_desk_puzzle` profile, complete T-junctions have two
integral half-heads captured by the spanning panel. Each corner has a 1.7 mm
neck; preassemble the split row before seating it against the spanning row,
following the same infill/anchor sequence. These keys restrain XY, not Z: keep
the supporting lips and mounting constraints. Missing corner/receiver stock or
custom profile dimensions omit both halves with a warning; no loose key is used.

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
The manifest records panels, native Lite mounting holes, joint/lip ownership, print
placements, coverage and warnings. Complete interior Lite mounting nodes follow
the native 56 mm pattern; clipped or split holes are omitted. Hardware selection and
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
  **David D / openGrid Lite or Full** and retain the generated attribution.
- [GridFlock](https://github.com/yawkat/GridFlock) inspired the puzzle-segmentation
  approach; it is not bundled or required.

See [NOTICE.md](NOTICE.md) and
[native asset attribution](src/opengrid/assets/native/ATTRIBUTION.md) for sources,
changes and license boundaries. No upstream endorsement is implied.
