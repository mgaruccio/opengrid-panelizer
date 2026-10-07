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
