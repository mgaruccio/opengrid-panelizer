"""CAD-independent native openGrid Lite and Full geometry.

The compiler uses this module from layout and preview code, so it deliberately
contains no build123d/OCC imports. ``socket_projection`` is the conservative
XY projection shared by the official 4 mm Lite and 6.8 mm Full cavities.
Their measured STEP vertices have the same 16-point convex hull over all Z
planes, despite their different 3-D engagement profiles. Keeping the widest
lower and upper chamfers is conservative for both families' keepouts.
"""

from __future__ import annotations

import math

from shapely.affinity import translate
from shapely.geometry import Polygon, box

from .models import Panel

MOUNTING_HEAD_RADIUS = 3.6
PITCH = 28.0
THICKNESS = 4.0

# Coordinates are relative to an accessory socket centre.  The 16-gon is the
# convex hull of the widest points in the bundled derived cutter: the upper
# opening reaches 13.2 mm from centre while the lower 0 mm section reaches the
# 12.9 mm chamfer points.  Keeping both sets is conservative over all Z.
SOCKET_PROJECTION_POINTS: tuple[tuple[float, float], ...] = (
    (-6.923044738, -13.2),
    (-9.202943725, -12.9),
    (-12.9, -9.202943725),
    (-13.2, -6.923044738),
    (-13.2, 6.923044738),
    (-12.9, 9.202943725),
    (-9.202943725, 12.9),
    (-6.923044738, 13.2),
    (6.923044738, 13.2),
    (9.202943725, 12.9),
    (12.9, 9.202943725),
    (13.2, 6.923044738),
    (13.2, -6.923044738),
    (12.9, -9.202943725),
    (9.202943725, -12.9),
    (6.923044738, -13.2),
)

_BASE_SOCKET_PROJECTION = Polygon(SOCKET_PROJECTION_POINTS)
if not _BASE_SOCKET_PROJECTION.is_valid:  # pragma: no cover - source constant
    raise RuntimeError("native socket projection is invalid")


def socket_projection(x: float, y: float) -> Polygon:
    """Return the conservative projected native socket cavity at ``(x, y)``.

    Coordinates are installation XY millimetres.  The returned polygon is a
    fresh Shapely geometry and is suitable for ``difference``/``intersection``
    in layout and preview code without importing a CAD kernel.
    """

    x = float(x)
    y = float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("socket projection coordinates must be finite")
    return translate(_BASE_SOCKET_PROJECTION, xoff=x, yoff=y)


def mounting_holes(panel: Panel) -> tuple[tuple[float, float], ...]:
    """Native 56 mm screw-hole pattern at complete interior four-cell nodes.

    Odd lattice nodes match the official 2x2/4x4 Lite boards. Never split a
    mounting hole between panels or place one in an irregular edge/joint.
    Coordinates share the socket lattice phase, including negative origins.
    Full uses separate native mounting tiles/snaps, never integrated screw holes.
    """
    if panel.board == "full":
        return ()
    cells = {(cell.ix, cell.iy): cell for cell in panel.cells}
    holes = []
    for (ix, iy), cell in sorted(cells.items()):
        nx, ny = ix + 1, iy + 1
        if nx % 2 != 1 or ny % 2 != 1:
            continue
        if not all(key in cells for key in ((nx, iy), (ix, ny), (nx, ny))):
            continue
        x, y = cell.x + PITCH / 2, cell.y + PITCH / 2
        # Bounding square plus 0.8 mm edge land is conservative for the round
        # 7.2 mm native head recess; reject clipped holes, not partial screw cuts.
        r = MOUNTING_HEAD_RADIUS + 0.8
        if panel.footprint.covers(box(x - r, y - r, x + r, y + r)):
            holes.append((x, y))
    return tuple(holes)
