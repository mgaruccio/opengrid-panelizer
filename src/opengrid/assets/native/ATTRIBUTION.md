# Native openGrid Lite source and attribution

The bundled `openGrid_Lite_2x2.step` and `openGrid_Lite_4x4.step` are the
official openGrid Lite STEP models published by David D in the openGrid
Wall/Desk mounting framework and ecosystem model:

- Source model: <https://www.printables.com/model/1214361-opengrid-walldesk-mounting-framework-and-ecosystem>
- License shown by the publisher: Creative Commons Attribution 4.0
  International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>.
- Required attribution: **David D, openGrid Lite, CC BY 4.0**.

The STEP files are retained as source reference assets. Their SHA-256 values
are recorded here so a replacement download can be checked:

- `openGrid_Lite_2x2.step`: `e0ec305060f05335f0439ef21a47a27d7c4eed9c9b014282ef99366cb1c9ac40`
- `openGrid_Lite_4x4.step`: `d1d21dead2b3e663e4eb78f0ceda30a8ac6edccf10aa289a5d50791891049693`
- `lite4mm_socket_cutter.step` (derived): `8612c1333bf6c38c8646712b7c4d00f89ead09c56fceaee90094f4d78a8a0991`

## Derived socket cutter

`lite4mm_socket_cutter.step` is a derived source asset. It was generated
from the official 4x4 STEP by subtracting the board from a 28 x 28 x 4 mm
box around the complete interior cell centred at `(1176, -1708)` in that
file, selecting the largest resulting solid (the accessory cavity), and
translating that solid so its socket centre is `(0, 0, 0)`. Corner
screw/mounting cuts were not selected. The normalized cutter is the exact
3-D native cavity used by `opengrid.cad`; `opengrid.native` uses its
conservative projected convex hull for CAD-independent layout keepouts.

The derivation can be reproduced with build123d 0.11.1 and the following
operations (all dimensions are millimetres):

```python
from build123d import Align, Box, Location, export_step, import_step

board = import_step("openGrid_Lite_4x4.step")
center = (1176.0, -1708.0)
tile = Box(28, 28, 4, align=(Align.MIN, Align.MIN, Align.MIN)).moved(
    Location((center[0] - 14, center[1] - 14, 0))
)
cavity = max(tile.cut(board).solids(), key=lambda solid: solid.volume)
cavity = cavity.translate((-center[0], -center[1], 0))
export_step(cavity, "lite4mm_socket_cutter.step")
```

The parametric openGrid-Studio SCAD reference was consulted during geometry
investigation but is not embedded in the application. The official STEP
geometry, rather than nominal prose dimensions, is authoritative here.
The reference mounting holes are now preserved separately as described below;
mounting hardware/attachment engineering remains user-handled.

## Derived mounting-hole cutter

`lite4mm_mounting_cutter.step` is derived from the same official 4x4 board.
Subtract the board from an 8 x 8 x 4 mm box centered on `(1162, -1722)`,
select the largest resulting solid (excluding neighbouring socket corners),
and translate by `(-1162, 1722, 0)`. This preserves the native 4.1 mm through
hole, 7.2 mm recessed/countersunk head profile on Z=0, and 4 mm thickness.
Volume: 99.6950394695547 mm³. It has the same David D / CC BY 4.0 attribution.
For generated panels, the CAD engine reflects only this cutter across XY and
translates by 4 mm, placing the head recess at Z=4 away from the print bed.
The bundled source cutter and accessory sockets are not altered.

Hole placement follows the official 2x2/4x4 56 mm pattern at odd lattice nodes
with four complete adjacent cells in the same panel and intact edge land.
No half-holes are cut along seams, irregular edges or keepouts; adapted puzzle
joints remain independent. This geometry does not specify wood screw suitability
or establish mounting strength.


## Native openGrid Full (6.8 mm)

`openGrid_Full_2x2.step` and `openGrid_Full_4x4.step` are the separate official
Full board models from the same David D publication and CC BY 4.0 licence above:

- [Official Full 2x2 STEP](https://files.printables.com/media/prints/1214361/stls/9114185_003e310f-0311-43c7-b9e2-f84adcc08ebf_74f5395c-89ad-4159-87b8-607af959337b/opengrid-2x2.step)
- [Official Full 4x4 STEP](https://files.printables.com/media/prints/1214361/stls/9114207_fce98296-75f1-49d3-a5ea-c4e41023ea98_cb0a1654-89e5-4cfa-97eb-6db3dbdcd743/opengrid-4x4.step)
- [Official tile dimension drawing](https://makerworld.bblmw.com/makerworld/model/USc4ad1ec99528a7/design/2025-08-10_e49c64b993b288.pdf)

Required attribution: **David D, openGrid Full, CC BY 4.0**.

SHA-256:

- `openGrid_Full_2x2.step`: `e69e5a035f577a65f95caec0078778d71b7765be48c1504b9cde486bea403c82`
- `openGrid_Full_4x4.step`: `423a4588ec583d84ac5bbbed4c15b1c79db44d402394c593a2a62d13596932fe`
- `full6p8mm_socket_cutter.step` (derived): `9a169e8944a9324c5d04af64ca9a00e08087d91d10235928c8e84ea2c2938bcc`

The Full cutter is derived with the same box-subtraction procedure as Lite,
but from a **28 x 28 x 6.8 mm** tile centred on **(1176, -1372)** in the Full
4x4 source. Select the largest resulting solid and translate by
`(-1176, 1372, 0)`. Its volume is 4108.003995374757 mm³; its bounds are
±13.2 mm in XY and Z=0..6.8 mm (within CAD tolerance). An independent extraction
from the Full 2x2 source at **(1400, -1596)** has zero Boolean difference in both
directions. Full is not a scaled or thickened Lite cavity.

Full uses native mounting tiles/snaps; no Lite mounting-hole cutter is derived
or added for this board family. Perimeter alignment cutouts are not part of
the socket cutter; custom panel joints remain separate from the native interface.
