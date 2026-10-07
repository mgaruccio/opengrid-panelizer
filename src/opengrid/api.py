"""Expert geometry escape hatch; normal installations use an agent-prepared spec."""

from collections.abc import Iterable, Mapping
from typing import Any

from shapely.geometry.base import BaseGeometry

from .models import InstallationSpec, JointSpec, Keepout, Layout, PrinterSpec
from .spec import SpecError


def generate_installation(
    usable_surface: BaseGeometry,
    keepouts: Iterable[BaseGeometry | Keepout] = (),
    grid_origin: tuple[float, float] = (0, 0),
    printer: PrinterSpec | None = None,
    joint_strategy: JointSpec | None = None,
    mounting_constraints: Mapping[str, Any] | None = None,
) -> Layout:
    """Plan printable panels from explicit mm geometry. Use export_layout for files.

    Mounting is user-handled in this release; nonempty mounting constraints are
    explicitly rejected rather than silently ignored or interpreted by an LLM.
    """
    if mounting_constraints:
        raise SpecError("Automatic mounting constraints/design are outside this release")
    from .layout import plan_installation

    constraints = tuple(
        k if isinstance(k, Keepout) else Keepout(f"keepout-{i + 1}", k)
        for i, k in enumerate(keepouts)
    )
    return plan_installation(
        InstallationSpec(
            id="explicit-geometry",
            surface=usable_surface,
            status="ready",
            keepouts=constraints,
            grid_origin=grid_origin,
            printer=printer or PrinterSpec(),
            joints=joint_strategy or JointSpec(),
        )
    )
