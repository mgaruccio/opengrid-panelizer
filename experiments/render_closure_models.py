"""Render the actual separate solids of a study STEP assembly to an orthographic PNG.

Run with `uv run --extra dev --with pillow python experiments/render_closure_models.py`.
Colors identify separate CAD solids; they are not a stress plot or filament colors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from build123d import import_step
from PIL import Image, ImageDraw

COLORS = ((49, 131, 175), (229, 158, 49), (65, 164, 130), (173, 114, 185))


def render(step: Path, output: Path, title: str, *, size=(1400, 1000)) -> None:
    shape = import_step(step)
    assert shape.is_valid and len(shape.solids()) > 0
    up = np.array((-0.34, 0.34, 0.877))
    up /= np.linalg.norm(up)
    horizontal = np.array((1, 1, 0)) / np.sqrt(2)
    view = np.cross(horizontal, up)
    light = np.array((-0.4, -0.4, 1.0))
    light /= np.linalg.norm(light)
    triangles, colors = [], []
    for index, solid in enumerate(shape.solids()):
        points, faces = solid.tessellate(0.08, 0.15)
        points = np.array([[p.X, p.Y, p.Z] for p in points])
        faces = np.array(faces)
        triangles.append(points[faces])
        colors.extend([COLORS[index % len(COLORS)]] * len(faces))
    triangles = np.concatenate(triangles)
    colors = np.array(colors, dtype=float)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    # Backface culling and painter ordering for an opaque orthographic CAD view.
    visible = normals @ view > -1e-7
    triangles, colors, normals = triangles[visible], colors[visible], normals[visible]
    screen = np.stack((triangles @ horizontal, -(triangles @ up)), axis=-1)
    low, high = screen.min(axis=(0, 1)), screen.max(axis=(0, 1))
    available = np.array((size[0] - 100, size[1] - 180))
    scale = float(min(available / np.maximum(high - low, 1e-9)))
    screen = (screen - (low + high) / 2) * scale + np.array((size[0] / 2, size[1] / 2 + 20))
    shade = 0.62 + 0.38 * np.maximum(0, normals @ light)
    colors = np.clip(colors * shade[:, None], 0, 255).astype(int)
    depth = triangles.mean(axis=1) @ view
    image = Image.new("RGB", size, (248, 247, 243))
    draw = ImageDraw.Draw(image)
    for index in np.argsort(depth):
        polygon = [(float(x), float(y)) for x, y in screen[index]]
        draw.polygon(polygon, fill=tuple(colors[index]))
    draw.text((35, 25), title.replace("—", "-"), fill=(32, 43, 55), font_size=29)
    draw.text((35, size[1] - 48), "Actual STEP geometry | colors = separate solids | not FEA",
              fill=(62, 72, 83), font_size=20)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Closure sample")
    args = parser.parse_args()
    render(args.step, args.output, args.title)


if __name__ == "__main__":
    main()
