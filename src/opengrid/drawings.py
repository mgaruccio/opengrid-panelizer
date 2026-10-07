"""Assembly drawings in installation coordinates, never a mirrored underside view."""

from pathlib import Path
from textwrap import wrap
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from shapely import union_all
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .models import PITCH, InstallationSpec, Layout

PALETTE = ("#d7e6f5", "#d9efdf", "#f4e4cd", "#e4def3", "#f1dce4", "#d8eceb")


def _notes(layout: Layout | None, provisional: str | None) -> list[tuple[str, bool]]:
    if provisional:
        return [("PROVISIONAL — NO PRINTABLE LAYOUT: " + provisional, True)]
    notes = [
        ("Panel STEP/STL use print placements; assembly.step uses installation coordinates.", False)
    ]
    if layout:
        notes.extend(("WARNING: " + warning, True) for warning in layout.warnings)
    return notes


def polygons(g: BaseGeometry):
    if isinstance(g, Polygon):
        yield g
    elif hasattr(g, "geoms"):
        for part in g.geoms:
            yield from polygons(part)


def svg_path(g: BaseGeometry) -> str:
    commands = []
    for p in polygons(g):
        for ring in (p.exterior, *p.interiors):
            points = list(ring.coords)
            commands.append(
                f"M {points[0][0]:.6f} {points[0][1]:.6f} "
                + " ".join(f"L {x:.6f} {y:.6f}" for x, y in points[1:])
                + " Z"
            )
    return " ".join(commands)


def _bounds(spec: InstallationSpec):
    x0, y0, x1, y1 = spec.surface.bounds
    pad = max(12, (x1 - x0) / 45)
    return x0, y0, x1, y1, pad


def write_svg(
    spec: InstallationSpec,
    path: Path,
    layout: Layout | None = None,
    *,
    provisional: str | None = None,
) -> None:
    x0, y0, x1, y1, pad = _bounds(spec)
    text_size = max(4, (x1 - x0) / 75)
    notes = [
        (line, caution)
        for message, caution in _notes(layout, provisional)
        for line in wrap(message, width=110)
    ]
    line_height = text_size * 1.05
    width = x1 - x0 + 2 * pad
    height = y1 - y0 + 4 * pad + line_height * max(0, len(notes) - 1)
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0 - pad} {y0 - 2 * pad} '
            f'{width} {height}" width="1200" role="img">'
        ),
        f"<title>{escape(spec.id)} — installation layout</title>",
        (
            "<style>path{stroke-width:.5;fill-rule:evenodd}text{font-family:system-ui,sans-serif}"
            "</style>"
        ),
        (
            f'<defs><pattern id="grid" patternUnits="userSpaceOnUse" x="{spec.grid_origin[0]}" '
            f'y="{spec.grid_origin[1]}" width="{PITCH}" height="{PITCH}">'
            f'<path d="M {PITCH} 0 L 0 0 0 {PITCH}" fill="none" stroke="#bcc6d0"/>'
            "</pattern></defs>"
        ),
        (
            f'<text x="{x0}" y="{y0 - pad}" font-size="{text_size}" fill="#182d42">'
            f"{escape(spec.id)} · rear at top · X right / Y toward front · mm</text>"
        ),
        f'<path d="{svg_path(spec.surface)}" fill="#f6f8fa" stroke="#333"/>',
    ]
    if layout:
        from .native import MOUNTING_HEAD_RADIUS, mounting_holes, socket_projection

        for i, panel in enumerate(layout.panels):
            lines.append(
                f'<g id="{panel.id}"><title>{panel.id}: {len(panel.cells)} intact cells'
                f'</title><path d="{svg_path(panel.footprint)}" '
                f'fill="{PALETTE[i % len(PALETTE)]}" stroke="#182d42"/>'
            )
            for cell in panel.cells:
                lines.append(
                    f'<path d="{svg_path(socket_projection(cell.x, cell.y))}" '
                    'fill="white" stroke="#738393"/>'
                )
            for x, y in mounting_holes(panel):
                lines.append(
                    f'<circle class="mounting-hole" cx="{x}" cy="{y}" '
                    f'r="{MOUNTING_HEAD_RADIUS}" fill="white" stroke="#182d42">'
                    "<title>Native Lite screw hole: 4.1 mm through / 7.2 mm head recess; "
                    "head face Z=4 (away from print bed)</title></circle>"
                )
            point = panel.footprint.representative_point()
            lines.append(
                f'<text x="{point.x}" y="{point.y}" font-size="{text_size}" '
                f'text-anchor="middle" fill="#182d42">{panel.id}</text></g>'
            )
        for joint in layout.joints:
            lines.append(
                f'<circle cx="{joint.x}" cy="{joint.y}" r="2" '
                f'fill="#d87600"><title>{joint.id}: {joint.male_panel} to '
                f"{joint.female_panel}</title></circle>"
            )
    else:
        lines.append(f'<path d="{svg_path(spec.surface)}" fill="url(#grid)" stroke="none"/>')
    for k in spec.keepouts:
        description = (
            f"{k.id}; confidence {k.confidence}; source {k.source}; "
            f"clearance {k.clearance} mm + uncertainty {k.uncertainty} mm"
        )
        lines.append(
            f'<path d="{svg_path(k.expanded)}" fill="#c9494933" stroke="#a72222">'
            f"<title>{escape(description)}</title></path>"
        )
    if layout and layout.warnings:
        lines.append(
            f'<text x="{x0}" y="{y0 - pad / 3}" font-size="{text_size * 0.8}" fill="#a72222">'
            f"WARNING: {len(layout.warnings)} conservative layout checks; review before printing.</text>"
        )
    for i, (line, caution) in enumerate(notes):
        lines.append(
            f'<text x="{x0}" y="{y1 + pad + i * line_height}" font-size="{text_size * 0.7}" '
            f'fill="{"#a72222" if caution else "#182d42"}">{escape(line)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_pdf(
    spec: InstallationSpec,
    path: Path,
    layout: Layout | None = None,
    *,
    provisional: str | None = None,
) -> None:
    """Vector PDF, fit-to-page and explicitly not a drilling template."""
    from reportlab.lib.colors import HexColor

    x0, y0, x1, y1, _ = _bounds(spec)
    page_w, page_h = 842, 595
    scale = min(760 / (x1 - x0), 415 / (y1 - y0))
    c = canvas.Canvas(str(path), pagesize=(page_w, page_h))
    c.setTitle(f"{spec.id} — openGrid assembly")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 555, spec.id)
    c.setFont("Helvetica", 10)
    c.drawString(
        40, 536, "Rear at top; X right / Y toward front. Millimetres. Fit-to-page, not 1:1."
    )

    def draw(g, fill, stroke):
        p = c.beginPath()
        for poly in polygons(g):
            for ring in (poly.exterior, *poly.interiors):
                points = list(ring.coords)
                p.moveTo(40 + (points[0][0] - x0) * scale, 505 - (points[0][1] - y0) * scale)
                for x, y in points[1:]:
                    p.lineTo(40 + (x - x0) * scale, 505 - (y - y0) * scale)
                p.close()
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(stroke))
        c.drawPath(p, fill=1, stroke=1, fillMode=0)

    draw(spec.surface, "#f6f8fa", "#333333")
    if layout:
        from .native import MOUNTING_HEAD_RADIUS, mounting_holes, socket_projection

        for i, panel in enumerate(layout.panels):
            draw(panel.footprint, PALETTE[i % len(PALETTE)], "#182d42")
            # Union before drawing avoids thousands of separate PDF objects.
            draw(
                union_all([socket_projection(cell.x, cell.y) for cell in panel.cells]),
                "#ffffff",
                "#738393",
            )
            from shapely.geometry import Point

            for x, y in mounting_holes(panel):
                draw(Point(x, y).buffer(MOUNTING_HEAD_RADIUS), "#ffffff", "#182d42")
            point = panel.footprint.representative_point()
            c.setFillColor(HexColor("#182d42"))
            c.drawCentredString(40 + (point.x - x0) * scale, 505 - (point.y - y0) * scale, panel.id)
    for k in spec.keepouts:
        draw(k.expanded.intersection(spec.surface), "#efbbbb", "#a72222")
    cautions = bool(provisional or (layout and layout.warnings))
    c.setFillColor(HexColor("#a72222" if cautions else "#182d42"))
    c.setFont("Helvetica", 9)
    if provisional:
        label = "PROVISIONAL — NO PRINTABLE LAYOUT. Unresolved constraints on following page."
    elif layout and layout.warnings:
        label = (
            f"WARNING: {len(layout.warnings)} layout checks. Read following page before printing."
        )
    else:
        label = (
            "Panel STEP/STL: print frame. assembly.step: installation frame. Fit needs a coupon."
        )
    c.drawString(40, 42, label)
    if cautions:
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(HexColor("#a72222"))
        c.drawString(40, 518, label)
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, 555, "Layout notes — " + spec.id)
        baseline = 520
        for message, caution in _notes(layout, provisional):
            for line in wrap(message, width=110):
                if baseline < 50:
                    c.showPage()
                    baseline = 540
                c.setFont("Helvetica", 11)
                c.setFillColor(HexColor("#a72222" if caution else "#182d42"))
                c.drawString(40, baseline, line)
                baseline -= 16
            baseline -= 8
    c.save()
