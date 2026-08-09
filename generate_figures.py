"""Generate revised, print-ready manuscript figures from verified run artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ijies-20265081-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import PercentFormatter


TEXT_RED = "#C00000"
EDGE = "#263238"
COLORS = {
    "Global": "#455A64",
    "Global + MI": "#4C78A8",
    "Quantile": "#E08B6D",
    "Quantile + MI": "#2A9D8F",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "text.color": TEXT_RED,
            "axes.labelcolor": TEXT_RED,
            "axes.titlecolor": TEXT_RED,
            "xtick.color": TEXT_RED,
            "ytick.color": TEXT_RED,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 1.0,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_exact(fig: plt.Figure, path: Path, width_px: int, height_px: int, dpi: int = 200) -> None:
    fig.set_size_inches(width_px / dpi, height_px / dpi)
    fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Creator": "IJIES reviewer-revision pipeline"})
    plt.close(fig)


def workflow_figure(path: Path) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, face, size=10.5, weight="normal"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=EDGE, linewidth=1.5))
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=TEXT_RED,
            fontweight=weight,
            linespacing=1.0,
        )

    def arrow(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.3,
                color=EDGE,
                shrinkA=2,
                shrinkB=2,
            )
        )

    ax.text(
        0.5,
        0.985,
        "Availability-audited, fold-isolated workflow",
        ha="center",
        va="top",
        fontsize=12.5,
        color=TEXT_RED,
    )
    box(0.01, 0.65, 0.20, 0.22, "OUTER\ntrain | test", "#ECEFF1", 10.5, "bold")
    box(
        0.24,
        0.65,
        0.34,
        0.22,
        "TRAIN ONLY\nimpute • split • select\ninner tune • fit",
        "#E8F5E9",
        10.5,
        "bold",
    )
    box(0.62, 0.65, 0.14, 0.22, "FREEZE\nobjects", "#E3F2FD", 10.5, "bold")
    box(
        0.80,
        0.65,
        0.18,
        0.22,
        "HELD-OUT\ntransform\nroute/predict",
        "#FFF3E0",
        10.5,
        "bold",
    )
    box(
        0.04,
        0.33,
        0.28,
        0.20,
        "PRIMARY\nstrict-6 scenarios\nnested Extra Trees",
        "#E0F2F1",
        10.5,
        "bold",
    )
    box(
        0.36,
        0.33,
        0.28,
        0.20,
        "BASELINES\nforest • boosting\nstacking",
        "#F3E5F5",
        10.5,
        "bold",
    )
    box(
        0.68,
        0.33,
        0.28,
        0.20,
        "SENSITIVITY\nuncertain vs leaky\nfixed estimator",
        "#FCE4EC",
        10.5,
        "bold",
    )
    box(
        0.17,
        0.04,
        0.66,
        0.20,
        "SHARED INFERENCE\nproject bootstrap • paired tests\ncorrected repeated-CV test",
        "#ECEFF1",
        10.5,
        "bold",
    )
    arrow((0.21, 0.76), (0.24, 0.76))
    arrow((0.58, 0.76), (0.62, 0.76))
    arrow((0.76, 0.76), (0.80, 0.76))
    arrow((0.89, 0.65), (0.18, 0.53))
    arrow((0.89, 0.65), (0.50, 0.53))
    arrow((0.89, 0.65), (0.82, 0.53))
    arrow((0.18, 0.33), (0.35, 0.24))
    arrow((0.50, 0.33), (0.50, 0.24))
    arrow((0.82, 0.33), (0.65, 0.24))
    save_exact(fig, path, 2550, 1150, dpi=425)


def metric_lookup(metrics: pd.DataFrame, family: str, feature_set: str, method: str, metric: str) -> pd.Series:
    row = metrics[
        (metrics["family"] == family)
        & (metrics["feature_set"] == feature_set)
        & (metrics["method"] == method)
        & (metrics["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError((family, feature_set, method, metric, len(row)))
    return row.iloc[0]


def provenance_figure(metrics: pd.DataFrame, path: Path) -> None:
    protocols = ["leaky14", "uncertain9", "strict6"]
    labels = ["Leaky 14", "Uncertain 9", "Strict 6"]
    bar_colors = ["#C62828", "#D98E04", "#2A9D8F"]
    fig, axes = plt.subplots(1, 2)
    for ax, metric, title in zip(axes, ["MAE", "Pred25"], ["MAE (person-hours)", "Pred(25), %"]):
        rows = [metric_lookup(metrics, "scale_fixed_et_sensitivity", protocol, "Global", metric) for protocol in protocols]
        values = np.asarray([row["estimate_full_precision"] for row in rows])
        lower = values - np.asarray([row["ci_2_5"] for row in rows])
        upper = np.asarray([row["ci_97_5"] for row in rows]) - values
        positions = np.arange(len(protocols))
        ax.bar(positions, values, color=bar_colors, width=0.62)
        ax.errorbar(positions, values, yerr=np.vstack([lower, upper]), fmt="none", color=EDGE, capsize=4, linewidth=1.3)
        ax.set_xticks(positions, labels)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.tick_params(axis="y", labelsize=11.5)
        ax.grid(axis="y", alpha=0.8)
        ax.set_axisbelow(True)
        if metric == "Pred25":
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    fig.suptitle("Feature provenance changes apparent accuracy", fontsize=13.5, color=TEXT_RED)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.78, wspace=0.22)
    save_exact(fig, path, 2850, 900, dpi=425)


def performance_figure(metrics: pd.DataFrame, path: Path) -> None:
    methods = list(COLORS)
    metric_titles = [("MAE", "MAE"), ("RMSE", "RMSE"), ("Pred25", "Pred(25), %"), ("R2", r"$R^2$")]
    fig, axes = plt.subplots(2, 2)
    for axis_index, (ax, (metric, title)) in enumerate(zip(axes.flat, metric_titles)):
        rows = [metric_lookup(metrics, "scale_nested_et", "strict6", method, metric) for method in methods]
        estimates = np.asarray([row["estimate_full_precision"] for row in rows])
        low = np.asarray([row["ci_2_5"] for row in rows])
        high = np.asarray([row["ci_97_5"] for row in rows])
        y = np.arange(len(methods))
        for position, method in enumerate(methods):
            ax.errorbar(
                estimates[position],
                position,
                xerr=[[estimates[position] - low[position]], [high[position] - estimates[position]]],
                fmt="o",
                color=COLORS[method],
                ecolor=EDGE,
                capsize=3,
                markersize=5.5,
                linewidth=1.1,
            )
        ax.set_yticks(y, methods if axis_index % 2 == 0 else [""] * len(methods))
        ax.invert_yaxis()
        # Keep the metric label inside the panel. Upper-row numeric ticks are
        # placed above the axes to avoid collisions in the compact 2x2 layout.
        ax.text(
            0.98,
            0.92,
            title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11.5,
            color=TEXT_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        )
        if axis_index < 2:
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", labeltop=True, labelbottom=False)
        ax.grid(axis="x", alpha=0.8)
        ax.tick_params(labelsize=11.5)
        if metric == "Pred25":
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    fig.suptitle("Strict-6 nested Extra Trees: shared 95% confidence intervals", fontsize=13.5, color=TEXT_RED)
    # Keep the four panels legible at the final 6.1-inch manuscript width,
    # while avoiding a tall canvas that forces an otherwise unnecessary page.
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.78, hspace=0.24, wspace=0.26)
    save_exact(fig, path, 2850, 1120, dpi=425)


def paired_delta_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    subset = predictions[
        (predictions["family"] == "scale_nested_et") & (predictions["feature_set"] == "strict6")
    ]
    global_errors = subset[subset["method"] == "Global"][
        ["project_id", "repetition", "absolute_error"]
    ].rename(columns={"absolute_error": "global_error"})
    records = []
    for method in ["Global + MI", "Quantile", "Quantile + MI"]:
        comparator = subset[subset["method"] == method][
            ["project_id", "repetition", "absolute_error"]
        ].rename(columns={"absolute_error": "comparator_error"})
        merged = global_errors.merge(comparator, on=["project_id", "repetition"], validate="one_to_one")
        delta = (
            merged.assign(delta=merged["global_error"] - merged["comparator_error"])
            .groupby("project_id")["delta"]
            .mean()
        )
        records.extend({"method": method, "project_id": project_id, "delta": value} for project_id, value in delta.items())
    return pd.DataFrame.from_records(records)


def paired_figure(predictions: pd.DataFrame, path: Path) -> None:
    deltas = paired_delta_frame(predictions)
    methods = ["Global + MI", "Quantile", "Quantile + MI"]
    fig, ax = plt.subplots()
    arrays = [deltas[deltas["method"] == method]["delta"].to_numpy() for method in methods]
    parts = ax.violinplot(arrays, positions=np.arange(1, 4), showmeans=True, showmedians=True, showextrema=True, widths=0.82)
    for body, method in zip(parts["bodies"], methods):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor(EDGE)
        body.set_alpha(0.70)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(EDGE if key != "cmeans" else TEXT_RED)
        parts[key].set_linewidth(1.2)
    ax.axhline(0, color=TEXT_RED, linestyle="--", linewidth=1.4)
    ax.set_xticks(np.arange(1, 4), methods)
    ax.set_title("Per-project error difference (five-repeat mean)", fontsize=13)
    ax.text(
        0.01,
        0.95,
        "Error difference (hours); positive favors comparator",
        transform=ax.transAxes,
        va="top",
        fontsize=11.5,
        color=TEXT_RED,
    )
    ax.grid(axis="y", alpha=0.8)
    ax.tick_params(labelsize=11.5)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.82)
    save_exact(fig, path, 2850, 900, dpi=425)


def category_figure(summary: pd.DataFrame, path: Path) -> None:
    rows = summary[summary["scenario"].isin(["Adequacy diagnostic only", "Quantile"])].copy()
    desired = ["Training-fold AFP quantiles", "Multivariate K-Means", "AFP K-Means"]
    rows = rows.drop_duplicates(["strategy"]).set_index("strategy").reindex(desired)
    medians = rows["median_minimum"].to_numpy(dtype=float)
    low = medians - rows["minimum_observed"].to_numpy(dtype=float)
    high = rows["maximum_minimum"].to_numpy(dtype=float) - medians
    fig, ax = plt.subplots()
    y = np.arange(3)
    ax.errorbar(medians, y, xerr=np.vstack([low, high]), fmt="o", color="#2A9D8F", ecolor=EDGE, capsize=5, markersize=8)
    ax.axvline(30, color=TEXT_RED, linestyle="--", linewidth=1.4, label=r"Pre-specified minimum $n$ = 30")
    ax.set_yticks(y, ["AFP quantiles", "Multivariate K-Means", "AFP K-Means"])
    ax.invert_yaxis()
    ax.set_xlabel("Smallest training category (projects)")
    ax.set_title("Category adequacy over 50 outer training folds", fontsize=13)
    ax.grid(axis="x", alpha=0.8)
    legend = ax.legend(loc="lower right", frameon=True, fontsize=11.5)
    for text in legend.get_texts():
        text.set_color(TEXT_RED)
    ax.tick_params(labelsize=11.5)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.32, top=0.74)
    save_exact(fig, path, 2850, 740, dpi=425)


def mi_figure(mi_records: pd.DataFrame, path: Path) -> None:
    nested = mi_records[mi_records["context"].str.startswith("scale_nested_et|strict6|")].copy()
    parts = nested["context"].str.split("|", expand=True)
    nested["scenario"] = parts[2]
    nested["category"] = parts[5]
    nested["row_label"] = nested["scenario"] + " — " + nested["category"]
    row_order = ["Global + MI — All", "Quantile + MI — Large", "Quantile + MI — Medium", "Quantile + MI — Small"]
    feature_order = ["AFP", "Input", "Output", "Enquiry", "File", "Interface"]
    matrix = (
        nested.groupby(["row_label", "feature"])["selected"]
        .mean()
        .unstack("feature")
        .reindex(index=row_order, columns=feature_order)
    )
    fig, ax = plt.subplots()
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(feature_order)), feature_order, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(row_order)), row_order)
    ax.set_title("Mutual-information selection stability", fontsize=13)
    ax.tick_params(labelsize=11.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.ax.set_title("Frequency", color=TEXT_RED, fontsize=11.5, pad=4)
    colorbar.ax.tick_params(colors=TEXT_RED, labelsize=11.5)
    for spine in colorbar.ax.spines.values():
        spine.set_edgecolor(EDGE)
    fig.subplots_adjust(left=0.32, right=0.88, bottom=0.31, top=0.78)
    save_exact(fig, path, 2850, 850, dpi=425)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    metrics_frame = pd.read_csv(run_dir / "metrics_full_precision.csv")
    prediction_frame = pd.read_csv(run_dir / "oof_predictions_full_precision.csv.gz")
    category_frame = pd.read_csv(run_dir / "category_diagnostics_summary.csv")
    mi_frame = pd.read_csv(run_dir / "mutual_information_by_outer_training_set.csv.gz")
    workflow_figure(output_dir / "image2.png")
    provenance_figure(metrics_frame, output_dir / "image3.png")
    performance_figure(metrics_frame, output_dir / "image4.png")
    paired_figure(prediction_frame, output_dir / "image5.png")
    category_figure(category_frame, output_dir / "image6.png")
    mi_figure(mi_frame, output_dir / "image7.png")
    for path in sorted(output_dir.glob("image*.png")):
        print(path)
