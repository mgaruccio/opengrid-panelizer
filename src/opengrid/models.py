"""Shared, CAD-independent contracts. All geometry and coordinates are millimetres."""

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

PITCH = 28.0
THICKNESS = 4.0


@dataclass(frozen=True)
class PrinterSpec:
    name: str = "bambu_p1s"
    width: float = 256.0
    depth: float = 256.0
    margin: float = 0.5
    max_panel_span: float = 224.0
    exclusions: tuple[BaseGeometry, ...] = field(default_factory=lambda: (box(0, 0, 18, 28),))

    @property
    def usable_bed(self) -> BaseGeometry:
        bed = box(self.margin, self.margin, self.width - self.margin, self.depth - self.margin)
        for exclusion in self.exclusions:
            bed = bed.difference(exclusion.buffer(self.margin))
        return bed


@dataclass(frozen=True)
class JointSpec:
    style: str = "puzzle"
    depth: float | None = None
    neck_width: float | None = None
    head_width: float | None = None
    clearance: float | None = None
    fillet_radius: float | None = None

    def __post_init__(self) -> None:
        # Resolve presets once, retaining the original puzzle defaults.
        wall_puzzle = self.style in {"wall", "under_desk_puzzle"}
        dimensions = (2.2, 3.4, 4.6, 0.4) if wall_puzzle else (1.5, 1.2, 2.4, 0.0)
        if self.style != "under_desk":
            for name, value in zip(
                ("depth", "neck_width", "head_width", "fillet_radius"), dimensions
            ):
                if getattr(self, name) is None:
                    object.__setattr__(self, name, value)
        if self.clearance is None:
            clearance = 0.0 if self.style == "under_desk" else 0.05 if wall_puzzle else 0.15
            object.__setattr__(self, "clearance", clearance)


@dataclass(frozen=True)
class Keepout:
    id: str
    geometry: BaseGeometry
    clearance: float = 0.0
    uncertainty: float = 0.0
    confidence: float = 1.0
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "explicit_geometry"})
    critical: bool = True

    @property
    def expanded(self) -> BaseGeometry:
        # A high confidence score is not a dimensional tolerance.
        return self.geometry.buffer(self.clearance + self.uncertainty)


@dataclass(frozen=True)
class InstallationSpec:
    id: str
    surface: BaseGeometry
    status: str = "ready"
    grid_origin: tuple[float, float] = (0.0, 0.0)
    keepouts: tuple[Keepout, ...] = ()
    printer: PrinterSpec = field(default_factory=PrinterSpec)
    joints: JointSpec = field(default_factory=JointSpec)
    unresolved: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    coordinate_system: dict[str, str] = field(
        default_factory=lambda: {"origin": "rear_left", "x": "right", "y": "front"}
    )
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "explicit_geometry"})


@dataclass(frozen=True)
class Cell:
    ix: int
    iy: int
    x: float  # centre in installation coordinates
    y: float


@dataclass(frozen=True)
class PrintPlacement:
    rotation: int
    x: float  # translation after rotation about global (0,0)
    y: float


@dataclass
class Panel:
    id: str
    footprint: Polygon
    cells: tuple[Cell, ...]
    placement: PrintPlacement
    joints: tuple["Joint", ...] = ()
    base_footprint: Polygon | None = None
    edges: tuple["EdgeInterface", ...] = ()
    grid_row: int | None = None  # Global brick row, retained through partition fragmentation.


@dataclass(frozen=True)
class Joint:
    id: str
    male_panel: str
    female_panel: str
    x: float
    y: float
    angle: int  # normal from male to female, degrees
    male: BaseGeometry
    female: BaseGeometry
    spec: JointSpec = field(default_factory=JointSpec)
    reservation: BaseGeometry | None = None


@dataclass(frozen=True)
class EdgeInterface:
    """A shared beveled lip, centred on a straight seam interval."""

    id: str
    male_panel: str
    female_panel: str
    x: float
    y: float
    angle: int
    length: float
    style: str
    clearance: float


@dataclass
class Layout:
    spec: InstallationSpec
    usable: BaseGeometry
    keepout_union: BaseGeometry
    panels: list[Panel]
    joints: list[Joint]
    unused: BaseGeometry
    warnings: list[str]
