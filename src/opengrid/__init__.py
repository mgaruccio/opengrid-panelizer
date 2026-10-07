"""Precise geometry in; deterministic openGrid panels out.

Product identification, image interpretation and reconstruction belong to the
calling agent workflow, not this package.
"""

__version__ = "0.1.0"

from .api import generate_installation
from .models import InstallationSpec, JointSpec, Keepout, PrinterSpec
from .spec import load_spec

__all__ = [
    "InstallationSpec",
    "JointSpec",
    "Keepout",
    "PrinterSpec",
    "generate_installation",
    "load_spec",
]
