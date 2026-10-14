#!/usr/bin/env python3
"""Generate the homepage model-results chart from docs/model-results.csv."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "model-results.csv"
OUTPUT_PATH = ROOT / "docs" / "images" / "model-results.png"
MODELS = ["PaliGemma-3B", "Qwen2-VL-2B", "Qwen2-VL-7B"]
DATASETS = ["COCO", "Visual Genome"]
METRICS = [
    ("pope_accuracy", "POPE accuracy", "higher is better"),
    ("rope_strict_accuracy", "ROPE strict accuracy", "higher is better"),
    (
        "chair_instance_hallucination_rate",
        "CHAIR-I hallucination rate",
        "lower is better",
    ),
    ("throne_f1", "THRONE F1", "higher is better"),
]
COLORS = {"COCO": "#355C9A", "Visual Genome": "#E58B3A"}


def main() -> None:
    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    image = Image.new("RGB", (1440, 1040), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=32)
    heading_font = ImageFont.load_default(size=22)
    body_font = ImageFont.load_default(size=17)
    small_font = ImageFont.load_default(size=15)
    draw.text((72, 36), "Reference model results", fill="#172B4D", font=title_font)

    panel_width, panel_height = 620, 405
    lefts, tops = (72, 748), (108, 548)
    plot_top_offset, plot_height = 68, 245
    plot_left_offset, plot_width = 60, 535
    for panel, (metric, label, direction) in enumerate(METRICS):
        x = lefts[panel % 2]
        y = tops[panel // 2]
        draw.text((x, y), label, fill="#172B4D", font=heading_font)
        draw.text((x, y + 30), direction, fill="#596579", font=small_font)
        plot_left = x + plot_left_offset
        plot_top = y + plot_top_offset
        plot_bottom = plot_top + plot_height
        for tick in range(0, 11, 2):
            tick_y = plot_bottom - round(plot_height * tick / 10)
            draw.line((plot_left, tick_y, plot_left + plot_width, tick_y), fill="#E5E9F0", width=1)
            draw.text((x + 12, tick_y - 8), f"{tick / 10:.1f}", fill="#596579", font=small_font)

        group_width = plot_width / len(MODELS)
        bar_width = 34
        for model_index, model in enumerate(MODELS):
            center = plot_left + group_width * (model_index + 0.5)
            for dataset_index, dataset in enumerate(DATASETS):
                score = float(
                    next(
                        row[metric]
                        for row in rows
                        if row["model"] == model and row["dataset"] == dataset
                    )
                )
                bar_x = center + (dataset_index - 0.5) * (bar_width + 5)
                bar_y = plot_bottom - round(plot_height * score)
                draw.rectangle(
                    (round(bar_x), bar_y, round(bar_x + bar_width), plot_bottom),
                    fill=COLORS[dataset],
                )
                draw.text(
                    (round(bar_x - 3), bar_y - 20),
                    f"{score:.3f}",
                    fill="#243247",
                    font=small_font,
                )
            model_label = model.replace("-", " ")
            box = draw.textbbox((0, 0), model_label, font=small_font)
            draw.text(
                (center - (box[2] - box[0]) / 2, plot_bottom + 10),
                model_label,
                fill="#344054",
                font=small_font,
            )

    legend_y = 974
    for index, dataset in enumerate(DATASETS):
        legend_x = 420 + index * 210
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=COLORS[dataset])
        draw.text((legend_x + 27, legend_y - 1), dataset, fill="#344054", font=body_font)
    draw.text(
        (72, 1008),
        "1,000 images per model and dataset. THRONE F1 normalized to 38 distractor categories per ground-truth object.",
        fill="#596579",
        font=small_font,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
