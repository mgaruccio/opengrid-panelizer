"""Create sample/assembly pictures and a bed map from actual generated and sliced files.

uv run --extra dev --with pillow python experiments/illustrate_closure_study.py \
  --generated opengrid/verification/closure-study/generated \
  --sliced opengrid/verification/closure-study/sliced \
  --output opengrid/verification/closure-study/guide
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build123d import Axis, Compound, export_step, import_step
from PIL import Image, ImageDraw
from render_closure_models import render


def one_solid(path: Path):
    shape = import_step(path)
    assert shape.is_valid and len(shape.solids()) == 1
    return shape.solids()[0]


def plate_map(verification: dict, output: Path) -> None:
    size, margin, scale = 1200, 70, 4.0
    image = Image.new("RGB", (size, size), "#f8f7f3")
    draw = ImageDraw.Draw(image)

    def point(x, y):
        return margin + x * scale, size - margin - y * scale

    draw.rectangle((*point(0, 256), *point(256, 0)), fill="white", outline="#374151", width=2)
    draw.rectangle((*point(0, 28.5), *point(18.5, 0)), fill="#ffc3c3", outline="#b91c1c")
    colors = {"A": "#9acfe4", "B": "#efc584", "C": "#a8d7b9"}
    for name, (x0, y0, x1, y1) in verification["bounds"].items():
        base = Path(name).stem
        key = base[0] if base[0] in colors else "C" if base.startswith("key") else "tag"
        fill = colors.get(key, "#dededb")
        draw.rectangle((*point(x0, y1), *point(x1, y0)), fill=fill, outline="#374151", width=1)
        center = point((x0 + x1) / 2, (y0 + y1) / 2)
        label = base
        if base.startswith("key-spare-"):
            label = f"{round(100 * float(base.split('-')[2])):02d}"
        elif base.startswith("key-"):
            label = "05"
        draw.text(center, label, fill="#17212f", anchor="mm", font_size=19 if y1-y0 > 20 else 12)
    draw.text((margin, 15), "Actual Orca layout - front of printer is bottom", fill="#17212f", font_size=28)
    draw.text((margin, size - 38), "Boxes show occupied bounds, not part outlines. Red = exclusion.",
              fill="#374151", font_size=21)
    draw.text((margin, size - 64), "Key groups: 05 tight / 12 medium / 20 loose. Four of each.",
              fill="#374151", font_size=20)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--sliced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.generated / "manifest.json").read_text())
    verification = json.loads((args.sliced / "verification.json").read_text())
    plate_map(verification, args.output / "plate-map.png")
    for concept, data in manifest["concepts"].items():
        render(args.generated / data["assembly_step"], args.output / f"assembly-{concept}.png",
               f"{concept} — {data['name']}")
        fea = args.generated / "fea" / concept
        shapes = [one_solid(fea / f"{name}.step") for name in ("left", "right")]
        key_path = fea / "key.step"
        if key_path.is_file():
            shapes.append(one_solid(key_path))
        joint = args.output / f"joint-{concept}.step"
        export_step(Compound(children=shapes), joint)
        render(joint, args.output / f"joint-{concept}.png", f"{concept} — actual connector close-up")
        if concept == "C":
            travel = manifest["parameters"]["key_travel_mm"]
            stages = [("loading", (0, -travel, 7), "Lower key into the open loading bay"),
                      ("in-bay", (0, -travel, 0), f"Slide key {travel:g} mm toward the closed end")]
            for name, shift, title in stages:
                path = args.output / f"joint-C-{name}.step"
                export_step(Compound(children=shapes[:2] + [shapes[2].translate(shift)]), path)
                render(path, args.output / f"joint-C-{name}.png", title)
    # A full-array exploded view preserves actual source identities and seam directions.
    sites = {s["id"]: s for s in manifest["concepts"]["C"]["joints"]}
    exploded = []
    travel = manifest["parameters"]["key_travel_mm"]
    for part in manifest["parts"]:
        if part.get("concept") != "C" or part["kind"] not in ("panel", "key"):
            continue
        shape = one_solid(args.generated / part["step"])
        placement = part["print_placement"]
        shape = shape.translate(tuple(-v for v in placement["translation"]))
        shape = shape.rotate(Axis.Z, -placement["rotation"])
        if part["kind"] == "key":
            angle = math.radians(sites[part["joint_id"]]["angle"])
            shape = shape.translate((travel * math.sin(angle), -travel * math.cos(angle), 12))
        exploded.append(shape)
    path = args.output / "assembly-C-loading.step"
    export_step(Compound(children=exploded), path)
    render(path, args.output / "assembly-C-loading.png", "C — place all four tiles, then insert keys")
    print(args.output)


if __name__ == "__main__":
    main()
