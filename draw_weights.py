#!/usr/bin/env python3
"""Draw standalone Figure 7 at its final manuscript size.

Run: python draw_weights.py
Dependency: matplotlib (pip install matplotlib)

Outputs: weights.pdf, weights.svg, weights.png, weights.tiff, and
static_criterion_weights.csv. By default, they are saved in figures/.
Use --output-dir to change
the destination. No original image, PDF, or experimental log is required.

Default physical size: 114 x 81.6 mm, matching Figure 7's current import
dimensions in main.pdf (aspect ratio approximately 1.40:1). Lettering is
Times New Roman, 9.9626 PDF points, matching the manuscript's 10-pt LaTeX
body text. Condition-label plus signs use LaTeX's Computer Modern math glyph.
Import at the generated physical size to preserve this lettering size.
Use --width-mm/--height-mm and --font-size for a different final layout.

PDF embeds fonts; SVG preserves exact glyphs as vector paths. PNG and
losslessly compressed TIFF are exported at 1000 dpi for line artwork.
No tight bounding-box crop is applied, preserving the physical canvas size.
Elsevier artwork guidance:
https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing

Equal weighting of three normalized criteria is exactly 1/3. The right
axis displays this reference as 0.33, rounded to two decimal places.
The twelve learned weights retain all three decimals from weights.png.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent
FONT_PATH = PROJECT_ROOT / "fonts" / "TimesNewRoman-ManuscriptSubset.ttf"
FIGURE_WIDTH_MM = 114.0
FIGURE_HEIGHT_MM = 81.6
ASPECT_RATIO = FIGURE_WIDTH_MM / FIGURE_HEIGHT_MM
BODY_FONT_PT = 9.9626
DPI = 1000
EQUAL_WEIGHT = 1.0 / 3.0

GROUP_NAMES = ("CIFAR-10", "Tiny-IN", "Tiny-IN + LN", "Tiny-IN + IC")
GROUP_TICK_LABELS = ("CIFAR-10", "Tiny-IN", "Tiny-IN\n$+$ LN", "Tiny-IN\n$+$ IC")
CRITERIA = ("SA", "Div", "SV")
WEIGHTS = (
    ("0.268", "0.478", "0.254"),
    ("0.346", "0.403", "0.251"),
    ("0.820", "0.007", "0.173"),
    ("0.391", "0.212", "0.397"),
)
COLORS = ("#087DB6", "#F58020", "#239A39")
REFERENCE_COLOR = "#1765A0"


def configure_style(font_size: float, font_file: Path) -> str:
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
        # The manuscript uses Computer Modern for mathematical symbols. The
        # bundled Times text subset omits '+', so render just this symbol as math.
        "mathtext.fontset": "cm",
    })
    return family


def check_layout(fig, ax, bars, annotations, legend, right_ax) -> None:
    """Verify values remain readable after font or geometry adjustments."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in annotations]
    bar_boxes = [bar.get_window_extent(renderer) for bar in bars]
    axes_box = ax.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    figure_box = fig.bbox
    problems = []
    for i, box in enumerate(boxes):
        if not (axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1
                and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1):
            problems.append(f"Value label {i} lies outside the axes.")
        if box.overlaps(legend_box):
            problems.append(f"Value label {i} overlaps the legend.")
        for j in range(i + 1, len(boxes)):
            if box.overlaps(boxes[j]):
                problems.append(f"Value labels {i} and {j} overlap.")
        for j, bar_box in enumerate(bar_boxes):
            if box.overlaps(bar_box):
                problems.append(f"Value label {i} overlaps bar {j}.")
        reference_y = ax.transData.transform((0, EQUAL_WEIGHT))[1]
        if box.y0 <= reference_y <= box.y1:
            problems.append(f"Value label {i} covers the equal-weight reference line.")
    all_text = [ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(),
                *right_ax.get_yticklabels()]
    for element in [legend, *all_text]:
        box = element.get_window_extent(renderer)
        if not (figure_box.x0 <= box.x0 and box.x1 <= figure_box.x1
                and figure_box.y0 <= box.y0 and box.y1 <= figure_box.y1):
            problems.append("Legend or axis text extends beyond the figure canvas.")
    if problems:
        raise RuntimeError("Adjust layout before exporting:\n" + "\n".join(problems))


def export_values(output_dir: Path) -> None:
    with (output_dir / "static_criterion_weights.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Condition", *CRITERIA])
        for name, row in zip(GROUP_NAMES, WEIGHTS):
            writer.writerow([name, *row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default="figures/")
    parser.add_argument("--width-mm", type=float, default=FIGURE_WIDTH_MM,
                        help="Final import width in mm (default: 114, matching main.pdf).")
    parser.add_argument("--height-mm", type=float, default=None,
                        help="Final height in mm; default preserves Figure 7's aspect ratio.")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, row in zip(GROUP_NAMES, WEIGHTS):
        if sum(map(Decimal, row)) != Decimal("1"):
            raise ValueError(f"Weights for {name} do not sum to 1.")

    family = configure_style(args.font_size, args.font_file)
    fig, ax = plt.subplots(figsize=(args.width_mm / 25.4, height_mm / 25.4), dpi=args.dpi)
    # Reserve a separate header for the legend and right margin for 0.33.
    fig.subplots_adjust(left=0.155, right=0.90, bottom=0.19, top=0.90)
    centers = [1.4 * i for i in range(len(GROUP_NAMES))]
    bar_offsets = (-0.40, 0.0, 0.40)
    bar_width = 0.19
    ax.set_xlim(-0.75, centers[-1] + 0.75)
    ax.set_ylim(0.0, 0.95)
    ax.set_ylabel("Normalized weight", labelpad=4)
    ax.set_xticks(centers, GROUP_TICK_LABELS)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=3, width=0.6, pad=3)
    ax.grid(axis="y", color="#DFE3E6", linewidth=0.45, linestyle="--")
    ax.set_axisbelow(True)

    bars, annotations = [], []
    for criterion_index, (criterion, offset, color) in enumerate(
            zip(CRITERIA, bar_offsets, COLORS)):
        x_positions = [center + offset for center in centers]
        values = [float(row[criterion_index]) for row in WEIGHTS]
        container = ax.bar(x_positions, values, width=bar_width,
                           color=color, linewidth=0, label=criterion, zorder=3)
        bars.extend(container)
        for group_index, (x, value, row) in enumerate(zip(x_positions, values, WEIGHTS)):
            tiny = value < 0.03
            # Leave clearance below the 1/3 line for the 0.268 label.
            vertical_offset = 0.5 if group_index == 0 and criterion_index == 0 else 3
            annotation = ax.annotate(
                row[criterion_index], xy=(x, value),
                xytext=(0, (9 if tiny else vertical_offset) * args.font_size / BODY_FONT_PT),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=args.font_size, fontweight="normal",
                color="#202020", zorder=6,
                bbox={"boxstyle": "square,pad=0.015", "facecolor": "white",
                      "edgecolor": "none"},
                arrowprops={"arrowstyle": "-", "color": "#6D747A",
                            "linewidth": 0.45, "shrinkA": 1, "shrinkB": 1}
                if tiny else None,
            )
            annotations.append(annotation)

    ax.axhline(EQUAL_WEIGHT, color=REFERENCE_COLOR, linewidth=0.8,
               linestyle=(0, (4, 2.5)), zorder=4)
    # Identity transform: both sides use the same normalized-weight scale.
    right_ax = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    right_ax.set_yticks([EQUAL_WEIGHT])
    right_ax.set_yticklabels(["0.33"], fontsize=args.font_size, fontweight="normal")
    right_ax.tick_params(axis="y", colors=REFERENCE_COLOR,
                         length=3, width=0.8, pad=3)

    handles = [Patch(facecolor=color, edgecolor="none", label=name)
               for color, name in zip(COLORS, CRITERIA)]
    handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linewidth=0.8,
                          linestyle=(0, (4, 2.5)), label="Equal weight"))
    legend = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 1.065),
                       ncol=4, fontsize=args.font_size, frameon=False,
                       handlelength=1.3, handletextpad=0.4, columnspacing=1.2,
                       borderaxespad=0.0)

    check_layout(fig, ax, bars, annotations, legend, right_ax)
    stem = args.output_dir / "weights"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(stem.with_suffix(".tiff"), dpi=args.dpi,
                pil_kwargs={"compression": "tiff_lzw"})
    pixel_width, pixel_height = fig.canvas.get_width_height()
    plt.close(fig)
    export_values(args.output_dir)
    print(f"Size: {args.width_mm:g} x {height_mm:g} mm; aspect={args.width_mm / height_mm:.4f}")
    print(f"Font: {family}, {args.font_size:g} pt (values, legend, axes and ticks)")
    print(f"PDF: {stem.with_suffix('.pdf')} (vector, embedded fonts)")
    print(f"SVG: {stem.with_suffix('.svg')}")
    print(f"PNG: {stem.with_suffix('.png')} ({pixel_width} x {pixel_height} pixels, {args.dpi} dpi)")
    print(f"TIFF: {stem.with_suffix('.tiff')} ({args.dpi} dpi, lossless LZW compression)")
    print("Equal-weight reference: 1/3, displayed as 0.33 on the right axis.")
    print("Layout checks passed: no overlapping values, bars, or clipped text.")


if __name__ == "__main__":
    main()
