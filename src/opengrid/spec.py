"""Safe declarative input parsing, with no product lookup or geometric inference."""

import json
import math
from pathlib import Path
from typing import Any

import yaml
from shapely import is_valid_reason
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from .models import InstallationSpec, JointSpec, Keepout, PrinterSpec


class SpecError(ValueError):
    """An invalid or unresolved manufacturing input."""


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{name} must be a finite number in millimetres")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise SpecError(f"{name} must be finite and >= {minimum}")
    return value


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise SpecError(f"{name} must be a mapping")
    return value


def _keys(value: dict, allowed: set[str], name: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise SpecError(f"Unknown {name} fields: {', '.join(sorted(map(str, extra)))}")


def _points(value: Any, name: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise SpecError(f"{name} needs at least three XY points")
    result = []
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise SpecError(f"{name} points must contain exactly X and Y")
        result.append((_number(p[0], name), _number(p[1], name)))
    return result


def geometry(value: Any, *, conservative: bool = False) -> BaseGeometry:
    """Build valid planar geometry; circle keepouts conservatively enclose the circle."""
    raw = _mapping(value, "geometry")
    kind = raw.get("type")
    try:
        if kind in {"rectangle", "rounded_rectangle"}:
            _keys(raw, {"type", "x", "y", "width", "depth", "radius"}, "geometry")
            x, y = _number(raw.get("x", 0), "x"), _number(raw.get("y", 0), "y")
            w = _number(raw["width"], "width", minimum=0.001)
            h = _number(raw["depth"], "depth", minimum=0.001)
            r = _number(raw.get("radius", 0), "radius", minimum=0)
            if r > min(w, h) / 2:
                raise SpecError("corner radius exceeds half the smaller dimension")
            if r:
                # Inscribed arcs are suitable for surfaces, not forbidden space.
                radius = r / math.cos(math.pi / 128) if conservative else r
                g = box(x + r, y + r, x + w - r, y + h - r).buffer(radius, quad_segs=32)
                # Keep the declared straight bounds while enclosing the true corner arcs.
                if conservative:
                    g = g.intersection(box(x, y, x + w, y + h))
            else:
                g = box(x, y, x + w, y + h)
        elif kind == "circle":
            _keys(raw, {"type", "x", "y", "radius"}, "geometry")
            r = _number(raw["radius"], "radius", minimum=0.001)
            if conservative:
                r /= math.cos(math.pi / 128)
            g = Point(_number(raw["x"], "x"), _number(raw["y"], "y")).buffer(r, quad_segs=32)
        elif kind == "polygon":
            _keys(raw, {"type", "points", "holes"}, "geometry")
            holes = raw.get("holes", [])
            if not isinstance(holes, list):
                raise SpecError("holes must be a list of rings")
            g = Polygon(_points(raw["points"], "polygon"), [_points(h, "hole") for h in holes])
        elif kind == "multi_polygon":
            _keys(raw, {"type", "polygons"}, "geometry")
            polys = raw["polygons"]
            if not isinstance(polys, list) or not polys:
                raise SpecError("polygons must be a nonempty list")
            members = [geometry(p, conservative=conservative) for p in polys]
            if any(not isinstance(p, Polygon) for p in members):
                raise SpecError("multi_polygon members must be polygons")
            g = MultiPolygon(members)
        else:
            raise SpecError(f"Unsupported geometry type: {kind!r}")
    except KeyError as exc:
        raise SpecError(f"Missing geometry field: {exc.args[0]}") from exc
    _valid_geometry(g, "geometry")
    return g


def _valid_geometry(g: BaseGeometry, name: str) -> None:
    if not isinstance(g, (Polygon, MultiPolygon)) or g.is_empty:
        raise SpecError(f"{name} must contain positive-area polygons")
    if g.has_z:
        raise SpecError(f"{name} must be planar XY geometry, without Z coordinates")
    if not g.is_valid or not all(math.isfinite(v) for v in g.bounds):
        raise SpecError(f"Invalid {name}: {is_valid_reason(g)}")
    if g.area <= 0:
        raise SpecError(f"{name} must contain positive-area polygons")


def validate_spec(spec: InstallationSpec) -> None:
    _valid_geometry(spec.surface, "surface")
    if spec.board not in ("lite", "full"):
        raise SpecError("board must be lite or full")
    if spec.board == "full" and spec.joints.style == "under_desk":
        raise SpecError("board: full does not support the Lite-only under_desk spring clip; "
                        "use under_desk_puzzle, wall, puzzle or none")
    if not isinstance(spec.id, str) or not spec.id or spec.status not in {"draft", "ready"}:
        raise SpecError("Installation needs an id and status draft or ready")
    if spec.coordinate_system != {"origin": "rear_left", "x": "right", "y": "front"}:
        raise SpecError("coordinate_system must be rear_left with X right and Y front")
    if len(spec.grid_origin) != 2:
        raise SpecError("grid_origin must contain two coordinates")
    for v in spec.grid_origin:
        _number(v, "grid_origin")
    p = spec.printer
    for n in ("width", "depth", "max_panel_span"):
        _number(getattr(p, n), f"printer.{n}", minimum=1)
    _number(p.margin, "printer.margin", minimum=0)
    if 2 * p.margin >= min(p.width, p.depth):
        raise SpecError("Printer margin leaves no usable bed")
    for exclusion in p.exclusions:
        _valid_geometry(exclusion, "printer exclusion")
    if p.usable_bed.is_empty:
        raise SpecError("Printer exclusions leave no usable bed")
    j = spec.joints
    if j.style not in {"puzzle", "none", "wall", "under_desk", "under_desk_puzzle"}:
        raise SpecError("Joint style must be puzzle, none, wall, under_desk or under_desk_puzzle")
    _number(j.clearance, "joints.clearance", minimum=0)
    if j.style == "under_desk_puzzle" and j.clearance > 0.1:
        raise SpecError("Under-desk puzzle lip clearance must be between 0 and 0.1 mm per side")
    if j.style == "under_desk":
        from .underdesk import MAX_CLEARANCE

        if not 0 <= j.clearance <= MAX_CLEARANCE:
            raise SpecError(f"Under-desk clearance must be between 0 and {MAX_CLEARANCE:g} mm per side")
        if any(getattr(j, n) is not None for n in ("depth", "neck_width", "head_width", "fillet_radius")):
            raise SpecError("Under-desk dimensions are fixed; configure only style and clearance")
    else:
        for n in ("depth", "neck_width", "head_width"):
            _number(getattr(j, n), f"joints.{n}", minimum=0.1)
        _number(j.fillet_radius, "joints.fillet_radius", minimum=0)
        if j.head_width <= j.neck_width:
            raise SpecError("Puzzle head must be wider than its neck")
        if j.clearance >= min(j.depth, j.neck_width) / 2:
            raise SpecError("Joint clearance is too large for its neck/depth")
        if j.fillet_radius > min(j.depth, j.neck_width) / 4:
            raise SpecError("Joint fillet radius is too large for its neck/depth")
        if j.style == "puzzle" and j.fillet_radius:
            raise SpecError("Fillets require the wall style")
    ids = set()
    for k in spec.keepouts:
        if not isinstance(k.id, str) or not k.id or k.id in ids:
            raise SpecError("Keepout IDs must be nonempty and unique")
        ids.add(k.id)
        _valid_geometry(k.geometry, f"keepout {k.id}")
        _number(k.clearance, "clearance", minimum=0)
        _number(k.uncertainty, "uncertainty", minimum=0)
        if not 0 <= _number(k.confidence, "confidence") <= 1:
            raise SpecError("confidence must be between zero and one")
        if not isinstance(k.critical, bool):
            raise SpecError("critical must be boolean")


def ensure_ready(spec: InstallationSpec) -> None:
    validate_spec(spec)
    reasons = list(spec.unresolved)
    if spec.status != "ready":
        reasons.insert(0, "installation status is draft")
    reasons += [
        f"critical keepout {k.id} has confidence below 0.75"
        for k in spec.keepouts
        if k.critical and k.confidence < 0.75
    ]
    if reasons:
        raise SpecError("Printable export blocked: " + "; ".join(reasons))


def from_dict(document: Any) -> InstallationSpec:
    root = _mapping(document, "document")
    _keys(root, {"schema_version", "installation"}, "document")
    if isinstance(root.get("schema_version"), bool) or root.get("schema_version") != 1:
        raise SpecError("schema_version must be 1")
    raw = _mapping(root.get("installation"), "installation")
    _keys(
        raw,
        {
            "id",
            "status",
            "units",
            "surface",
            "grid_origin",
            "keepouts",
            "printer",
            "joints",
            "unresolved",
            "metadata",
            "coordinate_system",
            "source",
            "board",
        },
        "installation",
    )
    if raw.get("units") != "mm":
        raise SpecError("This generator requires units: mm")
    try:
        keepouts = []
        for item in raw.get("keepouts", []):
            k = _mapping(item, "keepout")
            _keys(
                k,
                {"id", "geometry", "clearance", "uncertainty", "confidence", "source", "critical"},
                "keepout",
            )
            keepouts.append(
                Keepout(
                    id=k["id"],
                    geometry=geometry(k["geometry"], conservative=True),
                    clearance=_number(k.get("clearance", 0), "clearance", minimum=0),
                    uncertainty=_number(k.get("uncertainty", 0), "uncertainty", minimum=0),
                    confidence=_number(k.get("confidence", 1), "confidence", minimum=0),
                    source=_mapping(k.get("source", {"kind": "explicit_geometry"}), "source"),
                    critical=k.get("critical", True),
                )
            )
        p = _mapping(raw.get("printer", {}), "printer")
        _keys(p, {"name", "width", "depth", "margin", "max_panel_span", "exclusions"}, "printer")
        printer = PrinterSpec(
            **{k: v for k, v in p.items() if k != "exclusions"},
            **(
                {"exclusions": tuple(geometry(g, conservative=True) for g in p["exclusions"])}
                if "exclusions" in p
                else {}
            ),
        )
        j = _mapping(raw.get("joints", {}), "joints")
        _keys(j, {"style", "depth", "neck_width", "head_width", "clearance", "fillet_radius"}, "joints")
        origin = raw.get("grid_origin", [0, 0])
        if not isinstance(origin, (list, tuple)) or len(origin) != 2:
            raise SpecError("grid_origin must contain two coordinates")
        unresolved = raw.get("unresolved", [])
        if not isinstance(unresolved, list) or any(not isinstance(s, str) for s in unresolved):
            raise SpecError("unresolved must be a list of descriptions")
        coordinates = {"origin": "rear_left", "x": "right", "y": "front"}
        coordinates.update(_mapping(raw.get("coordinate_system", {}), "coordinate_system"))
        spec = InstallationSpec(
            id=raw["id"],
            status=raw.get("status", "draft"),
            surface=geometry(raw["surface"]),
            grid_origin=tuple(_number(v, "grid_origin") for v in origin),
            keepouts=tuple(keepouts),
            printer=printer,
            joints=JointSpec(**j),
            unresolved=tuple(unresolved),
            metadata=_mapping(raw.get("metadata", {}), "metadata"),
            source=_mapping(raw.get("source", {"kind": "explicit_geometry"}), "source"),
            coordinate_system=coordinates,
            board=raw.get("board", "lite"),
        )
        json.dumps(
            {
                "metadata": spec.metadata,
                "source": spec.source,
                "keepout_sources": [k.source for k in spec.keepouts],
            },
            allow_nan=False,
        )
        validate_spec(spec)
        return spec
    except SpecError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError(f"Invalid installation input: {exc}") from exc


def load_spec(path: str | Path) -> InstallationSpec:
    try:
        with Path(path).open() as stream:
            return from_dict(yaml.safe_load(stream))
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"Cannot read specification: {exc}") from exc
