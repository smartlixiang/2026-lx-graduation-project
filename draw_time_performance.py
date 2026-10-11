#!/usr/bin/env python3
"""Draw standalone Figure 6 at its final manuscript size.

Run: python draw_time_performance.py
Dependency: matplotlib (pip install matplotlib)
Outputs are written to figures/ unless --output-dir is supplied.

Accuracy source: main.pdf, Table 1 and Table 4 for the original methods;
local result/DBCore/cifar100/resnet50/<seed>/result_*.json for DBCore.
Each plotted accuracy is the unweighted arithmetic mean of the
mean accuracies at retention ratios 60%, 70%, 80%, and 90%. Standard
deviations are not used to calculate this mean.
For DBCore, each retention-ratio accuracy is calculated by averaging the last
10 accuracy_samples per seed, then averaging across saved seeds. Its selection
time is estimated to equal Forgetting's, as specified by the user.

IMPORTANT NAMING DISTINCTION:
Table 4's "Ours-Reuse" row is the experiment calibrated with the 50% known
split. It is therefore plotted here as "Ours (50%-seen)".
"Ours (Reuse)" means the adapters and static-criterion weights have
already been obtained. Its accuracy equals Full, as specified by the user.
Its 9.5-minute runtime is an explicit estimate within the supplied <10 min
range, not an exact measurement. Set GROUP_ONLY_MINUTES to the actual
runtime when available. All runtime values are the estimates supplied
for this figure; they are not read from the paper's accuracy tables.

Default physical size: 114 x 85.5 mm (4:3), matching Figure 6's current
114-mm import width in main.pdf. All text is Times New Roman, 9.9626 PDF
points, matching the manuscript's 10-pt LaTeX body text at this import size.
Use --width-mm/--height-mm to match a different final layout, and --font-size
to match its body text. Import at the generated physical size; rescaling the
figure also rescales its lettering.

PDF embeds TrueType fonts; SVG preserves their exact glyphs as vector paths.
PNG and losslessly compressed TIFF are exported at 1000 dpi for line artwork.
No tight bounding-box crop is applied, preserving physical size and aspect ratio.
Elsevier artwork guidance:
https://www.elsevier.com/en-gb/subject/next/guide-for-authors
https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


FIGURE_WIDTH_MM = 114.0
ASPECT_RATIO = 4.0 / 3.0
BODY_FONT_PT = 9.9626
DPI = 1000
MARKER_AREA = 30.0
OURS_MARKER_AREA = 105.0
GROUP_ONLY_MINUTES = 9.5
RETENTION_RATIOS = (60, 70, 80, 90)
PROJECT_ROOT = Path(__file__).resolve().parent
FONT_PATH = PROJECT_ROOT / "fonts" / "TimesNewRoman-ManuscriptSubset.ttf"


@dataclass(frozen=True)
class Method:
    name: str
    minutes: float
    accuracies: tuple[str, str, str, str]
    source: str
    color: str
    marker: str
    offset: tuple[float, float]
    align: str = "center"

    @property
    def mean_accuracy(self) -> Decimal:
        return sum(map(Decimal, self.accuracies)) / Decimal("4")


FULL_ACCURACIES = ("71.78", "74.62", "75.54", "77.18")

# The four accuracies in every tuple are in 60%, 70%, 80%, 90% order.
# Annotation offsets are in points, independent of PNG resolution.
METHODS = (
    Method("Herding", 4 + 15.1 / 60,
           ("71.47", "74.04", "75.13", "76.77"), "Table 1", "#C74D38", "o", (14, 22), "left"),
    Method("EL2N", 5 + 0.1 / 60,
           ("70.66", "74.76", "75.50", "75.93"), "Table 1", "#277DA8", "o", (17, 12), "left"),
    Method("GraNd", 7 + 39.0 / 60,
           ("71.23", "73.75", "75.09", "76.41"), "Table 1", "#E58A25", "o", (18, -16), "left"),
    Method("Forgetting", 45 + 59.1 / 60,
           ("69.92", "73.09", "73.62", "75.92"), "Table 1", "#488C49", "o", (22, -5), "left"),
    Method("MDS", 19 + 1.3 / 60,
           ("70.12", "72.96", "74.68", "77.01"), "Table 1", "#92604E", "o", (-6, 24)),
    Method("YangCLIP", 20 + 57.3 / 60,
           ("69.62", "73.09", "74.77", "75.75"), "Table 1", "#B25C8C", "o", (-3, -29)),
    Method("RLSelector", 2 * 60 + 31,
           ("71.34", "73.83", "74.10", "76.54"), "Table 1", "#68717A", "o", (-16, 27), "right"),
    Method("MoSo", 4 * 60,
           ("71.73", "72.44", "75.49", "75.25"), "Table 1", "#8871AC", "o", (-15, -29), "right"),
    Method("Ours (Full)", 2 * 60 + 20,
           FULL_ACCURACIES, "Table 1; Table 4 Ours-Full", "#9A9700", "*", (0, 28)),
    Method("Ours (50%-seen)", 60 + 20,
           ("72.55", "74.15", "75.26", "77.02"), "Table 4 Ours-Reuse", "#009CAC", "*", (-6, -37)),
    Method("Ours (Reuse)", GROUP_ONLY_MINUTES,
           FULL_ACCURACIES, "Same as Full, as specified", "#0874B5", "*", (0, 28)),
)


def build_methods(result_dir: Path) -> tuple[Method, ...]:
    """Add DBCore using local CIFAR-100/ResNet-50 results and Forgetting's time."""
    root = result_dir / "DBCore" / "cifar100" / "resnet50"
    if not root.is_dir():
        raise FileNotFoundError(f"DBCore result directory not found: {root}")
    seed_dirs = sorted(path for path in root.iterdir()
                       if path.is_dir() and path.name.isdecimal())
    if not seed_dirs:
        raise FileNotFoundError(f"No saved DBCore seed results found: {root}")
    accuracies = []
    for ratio in RETENTION_RATIOS:
        values = []
        for seed_dir in seed_dirs:
            path = seed_dir / f"result_{ratio}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Required DBCore result is missing: {path}")
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            samples = payload.get("accuracy_samples")
            if samples:
                tail = [Decimal(str(value)) for value in samples[-10:]]
                accuracy = sum(tail) / Decimal(len(tail))
            elif "accuracy" in payload:
                accuracy = Decimal(str(payload["accuracy"]))
            else:
                raise ValueError(f"No accuracy_samples or accuracy in {path}")
            if not accuracy.is_finite() or not 0 <= accuracy <= 1:
                raise ValueError(f"Expected accuracy in [0, 1] in {path}: {accuracy}")
            values.append(accuracy)
        accuracies.append(str(sum(values) / Decimal(len(values)) * Decimal("100")))
    forgetting_index = next(i for i, method in enumerate(METHODS)
                            if method.name == "Forgetting")
    dbcore = Method(
        "DBCore", METHODS[forgetting_index].minutes, tuple(accuracies),
        f"{root}/<seed>/result_*.json; last-10 mean per seed, then mean across seeds",
        "#3157A0", "o", (22, 18), "left",
    )
    return METHODS[:forgetting_index + 1] + (dbcore,) + METHODS[forgetting_index + 1:]


def configure_style(font_size: float, font_file: Path) -> str:
    # Use the manuscript's actual glyphs even on servers without Times installed.
    # See fonts/README.md for the bundled subset's origin and character coverage.
    if not font_file.is_file():
        raise FileNotFoundError(f"Font file missing: {font_file}; use --font-file to supply it.")
    font_manager.fontManager.addfont(str(font_file))
    family = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.size": font_size,
        "font.weight": "normal",
        "axes.labelsize": font_size,
        "axes.labelweight": "normal",
        "axes.labelcolor": "#171717",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "axes.unicode_minus": False,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "path",
    })
    return family


def check_layout(fig, ax, annotations, methods: tuple[Method, ...]) -> None:
    """Check labels against each other, plot boundaries, and other points.

    This is a layout guard, not a replacement for visual inspection.
    It runs again after manual annotation or font-size changes.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ann.get_bbox_patch().get_window_extent(renderer)
             for ann in annotations]
    boundary = ax.get_window_extent(renderer)
    problems = []
    for i, box in enumerate(boxes):
        if not (boundary.x0 <= box.x0 and box.x1 <= boundary.x1
                and boundary.y0 <= box.y0 and box.y1 <= boundary.y1):
            problems.append(f"Label outside axes: {methods[i].name}")
        for j in range(i + 1, len(boxes)):
            if box.overlaps(boxes[j]):
                problems.append(f"Label overlap: {methods[i].name} / {methods[j].name}")
        for j, method in enumerate(methods):
            x, y = ax.transData.transform((method.minutes, float(method.mean_accuracy)))
            # Keep the label box clear of the entire visible marker.
            marker_area = OURS_MARKER_AREA if method.marker == "*" else MARKER_AREA
            radius = (marker_area ** 0.5 / 2 + 0.4) * fig.dpi / 72
            if (box.x0 - radius < x < box.x1 + radius
                    and box.y0 - radius < y < box.y1 + radius):
                problems.append(f"Label covers marker: {methods[i].name} / {method.name}")
    # Check the entire canvas, including axis labels and tick lettering.
    for text in [ax.xaxis.label, ax.yaxis.label,
                 *ax.get_xticklabels(), *ax.get_yticklabels()]:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        canvas = fig.bbox
        if not (canvas.x0 <= box.x0 and box.x1 <= canvas.x1
                and canvas.y0 <= box.y0 and box.y1 <= canvas.y1):
            problems.append(f"Text clipped by canvas: {text.get_text()}")
    if problems:
        raise RuntimeError("Adjust annotation offsets before export:\n" + "\n".join(problems))


def save_data(output_dir: Path, methods: tuple[Method, ...]) -> None:
    path = output_dir / "time_performance_values.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "Selection_time_minutes", *[
            f"Accuracy_{ratio}pct" for ratio in RETENTION_RATIOS
        ], "Mean_accuracy_pct", "Accuracy_source", "Time_source"])
        for method in methods:
            time_source = ("Illustrative estimate within supplied <10 min range"
                           if method.name == "Ours (Reuse)"
                           else "User-supplied runtime estimate")
            if method.name == "DBCore":
                time_source = "Estimated equal to Forgetting, as specified by user"
            writer.writerow([method.name, f"{method.minutes:.6f}",
                             *method.accuracies, f"{method.mean_accuracy:.4f}",
                             method.source, time_source])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default="figures/")
    parser.add_argument("--result-dir", type=Path, default=PROJECT_ROOT / "result",
                        help="Root directory for DBCore's CIFAR-100/ResNet-50 results.")
    parser.add_argument("--width-mm", type=float, default=FIGURE_WIDTH_MM,
                        help="Final import width in mm (default: 114, matching main.pdf).")
    parser.add_argument("--height-mm", type=float, default=None,
                        help="Final height in mm; default preserves a 4:3 aspect ratio.")
    parser.add_argument("--font-size", type=float, default=BODY_FONT_PT,
                        help="Final lettering size in PDF points (default: manuscript body size).")
    parser.add_argument("--font-file", type=Path, default=FONT_PATH,
                        help="TrueType/OpenType font matching the manuscript.")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help="Bitmap resolution; at least 1000 dpi for line artwork.")
    args = parser.parse_args()
    height_mm = args.height_mm if args.height_mm is not None else args.width_mm / ASPECT_RATIO
    if not all(0 < value < float("inf") for value in (args.width_mm, height_mm, args.font_size)):
        parser.error("Figure dimensions and font size must be finite and positive.")
    if args.dpi < 1000:
        parser.error("--dpi must be >= 1000 for bitmap line artwork; use the vector PDF for submission.")
    methods = build_methods(args.result_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < GROUP_ONLY_MINUTES < 10:
        raise ValueError("GROUP_ONLY_MINUTES must satisfy the supplied 0 < time < 10 min.")

    family = configure_style(args.font_size, args.font_file)
    fig, ax = plt.subplots(figsize=(args.width_mm / 25.4, height_mm / 25.4), dpi=args.dpi)
    fig.subplots_adjust(left=0.165, right=0.98, bottom=0.17, top=0.97)
    ax.set_xscale("log")
    ax.set_xlim(3.4, 310)
    ax.set_ylim(72.95, 75.18)
    ax.set_xlabel("Selection time (minutes)", labelpad=4)
    ax.set_ylabel("Mean accuracy (%)", labelpad=4)
    ax.set_xticks([5, 10, 30, 60, 120, 240])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yticks([73.00, 73.25, 73.50, 73.75, 74.00, 74.25, 74.50, 74.75, 75.00])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", which="major", length=3, width=0.6, pad=3)
    ax.grid(True, which="major", color="#DFE3E6", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)

    annotations = []
    for method in methods:
        ours = method.marker == "*"
        accuracy = float(method.mean_accuracy)
        ax.scatter(method.minutes, accuracy,
                   s=OURS_MARKER_AREA if ours else MARKER_AREA, marker=method.marker,
                   color=method.color, edgecolors="white", linewidths=0.4,
                   zorder=4)
        ann = ax.annotate(
            method.name, xy=(method.minutes, accuracy),
            xytext=tuple(value * args.font_size / 19.0 for value in method.offset),
            textcoords="offset points",
            ha=method.align, va="center", fontsize=args.font_size,
            fontweight="normal", color="#151515",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": "#C6CBCF", "linewidth": 0.4, "alpha": 1.0},
            arrowprops={"arrowstyle": "-", "color": "#8A9197",
                        "linewidth": 0.45, "shrinkA": 1, "shrinkB": 4 if ours else 3},
            annotation_clip=False, zorder=5,
        )
        annotations.append(ann)

    check_layout(fig, ax, annotations, methods)
    stem = args.output_dir / "time_performance_comparison"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(stem.with_suffix(".tiff"), dpi=args.dpi,
                pil_kwargs={"compression": "tiff_lzw"})
    pixel_width, pixel_height = fig.canvas.get_width_height()
    plt.close(fig)
    save_data(args.output_dir, methods)
    for method in methods:
        print(f"{method.name:22s} {method.minutes:10.6f} min  {method.mean_accuracy:.4f}%")
    print(f"Size: {args.width_mm:g} x {height_mm:g} mm; aspect={args.width_mm / height_mm:.4f}")
    print(f"Font: {family}, {args.font_size:g} pt (all labels and ticks)")
    print(f"PDF: {stem.with_suffix('.pdf')} (vector, embedded TrueType font)")
    print(f"SVG: {stem.with_suffix('.svg')}")
    print(f"PNG: {stem.with_suffix('.png')} ({pixel_width} x {pixel_height} pixels, {args.dpi} dpi)")
    print(f"TIFF: {stem.with_suffix('.tiff')} ({args.dpi} dpi, lossless LZW compression)")
    print("Layout checks passed: no overlapping labels, covered markers, or clipped axis text.")


if __name__ == "__main__":
    main()
