from __future__ import annotations

from math import isclose
from pathlib import Path

import pytest
from build123d import Align, Box, Location, Mesher, export_step, export_stl, import_step
from shapely.geometry import Point, Polygon

from opengrid.cad import SOCKET_CUTTER_ASSET, build_panel, printable_panel
from opengrid.models import Cell, Panel, PrintPlacement
from opengrid.native import socket_projection

ASSET_DIR = Path(__file__).parents[1] / "src" / "opengrid" / "assets" / "native"


def _source_socket(path: Path, centre: tuple[float, float]):
    board = import_step(path)
    tile = Box(28, 28, 4, align=(Align.MIN, Align.MIN, Align.MIN)).moved(
        Location((centre[0] - 14, centre[1] - 14, 0))
    )
    components = tile.cut(board).solids()
    assert components
    return max(components, key=lambda solid: solid.volume).translate((-centre[0], -centre[1], 0))


def _panel(
    footprint: Polygon,
    cells: tuple[Cell, ...],
    *,
    rotation: int = 0,
    x: float = 0,
    y: float = 0,
) -> Panel:
    return Panel(
        id="test-panel",
        footprint=footprint,
        cells=cells,
        placement=PrintPlacement(rotation=rotation, x=x, y=y),
    )


def test_socket_projection_is_conservative_and_cad_independent():
    projection = socket_projection(28, -14)

    assert projection.is_valid
    assert projection.geom_type == "Polygon"
    assert len(projection.exterior.coords) == 17
    assert all(
        isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(projection.bounds, (14.8, -27.2, 41.2, -0.8))
    )

    # The largest 3-D cutter extrema from both the lower and upper chamfers
    # must remain inside the CAD-independent projection.
    cutter = import_step(SOCKET_CUTTER_ASSET)
    for vertex in cutter.vertices():
        point = vertex.center()
        assert projection.buffer(1e-7).covers(Point(point.X + 28, point.Y - 14))


def test_socket_cutter_matches_official_2x2_and_4x4_interior_cells():
    cutter = import_step(SOCKET_CUTTER_ASSET)
    assert len(cutter.solids()) == 1

    references = (
        (ASSET_DIR / "openGrid_Lite_2x2.step", (1400.0, -1932.0)),
        (ASSET_DIR / "openGrid_Lite_4x4.step", (1176.0, -1708.0)),
    )
    for path, centre in references:
        source_socket = _source_socket(path, centre)
        assert isclose(source_socket.volume, cutter.volume, rel_tol=1e-8, abs_tol=1e-6)
        # Equal volume/bounds alone cannot establish the engagement profile.
        assert source_socket.cut(cutter).volume < 1e-5
        assert cutter.cut(source_socket).volume < 1e-5
        source_bounds = source_socket.bounding_box()
        cutter_bounds = cutter.bounding_box()
        for actual, expected in (
            (source_bounds.min.X, cutter_bounds.min.X),
            (source_bounds.min.Y, cutter_bounds.min.Y),
            (source_bounds.min.Z, cutter_bounds.min.Z),
            (source_bounds.max.X, cutter_bounds.max.X),
            (source_bounds.max.Y, cutter_bounds.max.Y),
            (source_bounds.max.Z, cutter_bounds.max.Z),
        ):
            assert isclose(actual, expected, abs_tol=1e-6)


@pytest.mark.parametrize("clockwise", [False, True])
def test_ring_winding_cannot_reverse_extrusion_or_skip_native_cuts(clockwise):
    footprint = Polygon([(0, 0), (28, 0), (28, 28), (0, 28)])
    if clockwise:
        footprint = footprint.reverse()
    shape = build_panel(_panel(footprint, (Cell(0, 0, 14, 14),)))
    bounds = shape.bounding_box()
    assert isclose(bounds.min.Z, 0, abs_tol=1e-6)
    assert isclose(bounds.max.Z, 4, abs_tol=1e-6)
    cutter = import_step(SOCKET_CUTTER_ASSET)
    assert isclose(shape.volume, 28 * 28 * 4 - cutter.volume, abs_tol=1e-5)


def test_build_panel_cuts_multiple_native_sockets_and_supports_concavity():
    footprint = Polygon([(0, 0), (56, 0), (56, 28), (28, 28), (28, 56), (0, 56)])
    cells = (
        Cell(ix=0, iy=0, x=14, y=14),
        Cell(ix=1, iy=0, x=42, y=14),
        Cell(ix=0, iy=1, x=14, y=42),
    )
    panel = _panel(footprint, cells)

    shape = build_panel(panel)

    assert shape.is_valid
    assert len(shape.solids()) == 1
    assert isclose(shape.bounding_box().min.Z, 0, abs_tol=1e-6)
    assert isclose(shape.bounding_box().max.Z, 4, abs_tol=1e-6)
    assert shape.volume < footprint.area * 4
    assert shape.volume > 0


def test_build_panel_preserves_footprint_holes():
    footprint = Polygon(
        [(0, 0), (56, 0), (56, 56), (0, 56)],
        holes=[[(26, 26), (30, 26), (30, 30), (26, 30)]],
    )
    panel = _panel(
        footprint,
        (
            Cell(ix=0, iy=0, x=14, y=14),
            Cell(ix=1, iy=0, x=42, y=14),
            Cell(ix=0, iy=1, x=14, y=42),
            Cell(ix=1, iy=1, x=42, y=42),
        ),
    )

    shape = build_panel(panel)

    assert shape.is_valid
    assert len(shape.solids()) == 1
    assert shape.volume < footprint.area * 4


def test_printable_panel_rotates_about_origin_then_translates():
    footprint = Polygon([(0, 0), (28, 0), (28, 28), (0, 28)])
    panel = _panel(
        footprint,
        (Cell(ix=0, iy=0, x=14, y=14),),
        rotation=90,
        x=100,
        y=200,
    )

    shape = printable_panel(panel)
    bounds = shape.bounding_box()

    assert isclose(bounds.min.X, 72, abs_tol=1e-6)
    assert isclose(bounds.max.X, 100, abs_tol=1e-6)
    assert isclose(bounds.min.Y, 200, abs_tol=1e-6)
    assert isclose(bounds.max.Y, 228, abs_tol=1e-6)
    assert isclose(bounds.min.Z, 0, abs_tol=1e-6)
    assert isclose(bounds.max.Z, 4, abs_tol=1e-6)


def test_native_panel_step_and_stl_round_trip(tmp_path: Path):
    footprint = Polygon([(0, 0), (28, 0), (28, 28), (0, 28)])
    panel = _panel(footprint, (Cell(ix=0, iy=0, x=14, y=14),))
    shape = build_panel(panel)

    step_path = tmp_path / "panel.step"
    stl_path = tmp_path / "panel.stl"
    assert export_step(shape, step_path)
    assert export_stl(shape, stl_path, tolerance=0.01, angular_tolerance=0.1)

    reloaded_step = import_step(step_path)
    reloaded_mesh = Mesher().read(stl_path)
    assert len(reloaded_step.solids()) == 1
    assert len(reloaded_mesh) == 1
    assert isclose(reloaded_step.volume, shape.volume, rel_tol=1e-8, abs_tol=1e-6)
    assert isclose(reloaded_mesh[0].volume, shape.volume, rel_tol=2e-3, abs_tol=0.5)
    assert isclose(reloaded_step.bounding_box().min.Z, 0, abs_tol=1e-6)
    assert isclose(reloaded_step.bounding_box().max.Z, 4, abs_tol=1e-6)
