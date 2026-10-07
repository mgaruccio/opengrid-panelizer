# Licensing and attribution

## Software

The original Python source, tests and documentation in this repository are
licensed under the [MIT License](LICENSE).

## Native and derived CAD

The bundled openGrid Lite STEP models and derived socket/mounting cutters in
`src/opengrid/assets/native/` are by **David D**, licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
These assets are not relicensed under MIT.

Source: [openGrid Wall/Desk mounting framework and ecosystem](https://www.printables.com/model/1214361-opengrid-walldesk-mounting-framework-and-ecosystem).
See [native asset attribution](src/opengrid/assets/native/ATTRIBUTION.md) for
provenance, cutter derivation, and source-file checksums.

Generated panels incorporate that native geometry. Original panel-design
contributions are also offered under CC BY 4.0. Keep the generated
`ATTRIBUTION.txt` with redistributed STEP/STL files, credit David D and this
project, link the license, and identify your modifications. The compiler's
software license is separate from the CAD license.

Changes include custom panel outlines, puzzle joints, supporting lips and
mounting-hole placement. Socket cavities preserve the native geometry; the
mounting cutter is reflected so screw-head recesses face away from the mounting
surface. These modifications are not official openGrid board interfaces.

## References

[GridFlock](https://github.com/yawkat/GridFlock), by Jonas Konrad, inspired the
puzzle-segmentation approach. It is not bundled or required at runtime. Its
42 mm Gridfinity contract is not this project's 28 mm openGrid contract.

This is an independent project. No endorsement by openGrid's creator or other
referenced projects is implied. Attribution does not establish physical fit,
structural capacity, or suitability for a particular installation.
