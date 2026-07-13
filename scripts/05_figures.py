from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

FIGURES_DIR = config.FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR = config.STAGING_RESULTS_DIR 
LONG_DIR = config.LONGITUDINAL_RESULTS_DIR
DPI = config.FIGURE_DPI   
FONT_SIZE  = 9
TITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 8
LINE_WIDTH = 0.8
COL_MRI = "#2166AC"   
COL_AMYLOID = "#D6604D"   
COL_MMSE = "#4393C3"   
COL_CDRSUM  = "#F4A582"
CI_ALPHA = 0.7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "axes.linewidth": LINE_WIDTH,
    "xtick.major.width": LINE_WIDTH,
    "ytick.major.width": LINE_WIDTH,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": LINE_WIDTH,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

_BIOMARKER_LABELS: dict[str, str] = {
    "hippocampus_vol": "Hippocampal\nVolume",
    "entorhinal_thickness": "Entorhinal\nThickness",
    "fusiform_vol": "Fusiform\nVolume",
    "inferiortemporal_vol": "Inf. Temporal\nVolume",
    "ventricular_vol": "Ventricular\nVolume",
    "whole_brain_vol": "Whole Brain\nVolume",
    "Centiloid_fSUVR_TOT_CORTMEAN": "Amyloid\n(Centiloid)",
}
 
_PANEL_LABELS: dict[str, str] = {
    "mri_only": "MRI-only",
    "mri_amyloid": "MRI + Amyloid",
}
 
 
def _bio_label(raw: str) -> str:
    """Return the formatted biomarker label, falling back to the raw name."""
    return _BIOMARKER_LABELS.get(raw, raw.replace("_", " ").title())

def _load_pvd(panel_name: str) -> np.ndarray:
    """Load the positional variance diagram matrix for one panel.
 
    Returns shape (N_positions, N_biomarkers).  Each entry pvd[p, b] is the
    proportion of bootstrap resamples placing biomarker b at position p.
    """
    path = STAGING_DIR / panel_name / "pvd_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"PVD matrix not found: {path}\n"
            "Run 03_ebm_staging.py with bootstrap enabled first."
        )
    return np.load(str(path))
 
def _load_event_sequence(panel_name: str) -> pd.DataFrame:
    """Load the ML event sequence CSV for one panel.
 
    Expected columns: position, biomarker, biomarker_col_index,
                      positional_confidence.
    """
    path = STAGING_DIR / panel_name / "event_sequence.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Event sequence not found: {path}\n"
            "Run 03_ebm_staging.py first."
        )
    df = pd.read_csv(path)
    df = df.sort_values("position").reset_index(drop=True)
    return df
  
def _load_lme_results() -> pd.DataFrame:
    """Load the mixed-effects model results CSV (read-only)."""
    path = LONG_DIR / "mixed_effects_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"LME results not found: {path}\n"
            "Run 04_longitudinal_validation.py first."
        )
    return pd.read_csv(path)

def _pvd_display_matrix(
    pvd: np.ndarray,
    seq: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Reorder the PVD columns so biomarkers appear in ML sequence order.
 
    The raw pvd[position_0idx, biomarker_col_idx] matrix has columns in the
    arbitrary column-index order of the original feature matrix.  For a
    readable heatmap we reorder columns so that the highest-probability cell
    falls near the diagonal.
 
    Returns
    -------
    display     (N, N) float — display[row=ml_pos, col=event_pos]
    ylabels     list[str]   — biomarker labels in ML sequence order (top→bottom)
    col_indices list[int]   — original biomarker_col_index values in ML order
    """
    n = len(pvd)
    col_order   = [int(seq.loc[i, "biomarker_col_index"]) for i in range(n)]
    bio_names   = [str(seq.loc[i, "biomarker"]) for i in range(n)]
    display = np.zeros((n, n))
    for ml_pos, bio_col in enumerate(col_order):
        display[ml_pos, :] = pvd[:, bio_col]
 
    ylabels = [_bio_label(name) for name in bio_names]
    return display, ylabels, col_order

def fig01_pvd_heatmap(panel_name: str) -> Path:
    """Generate a publication-quality PVD heatmap.
 
    x-axis = Event Position (1..N)
    y-axis = Biomarker (in ML sequence order, most likely position at top)
    colour = bootstrap probability (0 → 1)
 
    The ML sequence diagonal is highlighted with a subtle border to guide
    the reader to the most likely ordering.
 
    Saves to  results/figures/fig01_pvd_heatmap_{panel_name}.png
    """
    pvd   = _load_pvd(panel_name)
    seq   = _load_event_sequence(panel_name)
    display, ylabels, _ = _pvd_display_matrix(pvd, seq)
    n     = len(display)
 
    panel_label = _PANEL_LABELS.get(panel_name, panel_name)
    cell_size = 0.75
    fig_w     = max(5.5, n * cell_size + 2.2)
    fig_h     = max(4.0, n * cell_size + 1.6)
    fig, ax   = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        display,
        cmap="Blues",
        vmin=0.0,
        vmax=min(1.0, display.max() * 1.05),
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label("Probability", fontsize=LABEL_SIZE, labelpad=6)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.outline.set_linewidth(LINE_WIDTH)

    for row in range(n):
        for col in range(n):
            val = display[row, col]
            if val >= 0.10:
                ax.text(
                    col, row, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if val > 0.45 else "black",
                    fontweight="bold" if val > 0.45 else "normal",
                )

    for ml_pos in range(n):
        rect = plt.Rectangle(
            (ml_pos - 0.5, ml_pos - 0.5), 1, 1,
            fill=False, edgecolor="#333333", linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=TICK_SIZE)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ylabels, fontsize=TICK_SIZE)
        ax.set_xlabel("Event Position", fontsize=LABEL_SIZE, labelpad=6)
        ax.set_ylabel("Biomarker", fontsize=LABEL_SIZE, labelpad=6)
        ax.set_title(
        f"Positional Variance Diagram\n{panel_label}",
        fontsize=TITLE_SIZE, fontweight="bold", pad=8,
        )
        ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        ax.spines["left"].set_linewidth(LINE_WIDTH)
        ax.spines["bottom"].set_linewidth(LINE_WIDTH)
        plt.tight_layout()
        out_path = FIGURES_DIR / f"fig01_pvd_heatmap_{panel_name}.png"
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved: {out_path}")
        return out_path

def fig02_event_sequence(panel_name: str) -> Path:
    """Generate a horizontal event sequence summary diagram.
 
    Shows biomarkers ranked by their ML sequence position, with the
    bootstrap positional confidence displayed alongside each biomarker.
 
    Layout (top → bottom = position 1 → N):
 
        Position 1 ── [██████████░░░░] Biomarker Name   conf = 0.60
        Position 2 ── [████████░░░░░░] Biomarker Name   conf = 0.30
        ...
 
    Saves to  results/figures/fig02_event_sequence_{panel_name}.png
    """
    seq          = _load_event_sequence(panel_name)
    n            = len(seq)
    panel_label  = _PANEL_LABELS.get(panel_name, panel_name)
 
    fig_h = max(3.5, n * 0.7 + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.set_xlim(-0.05, 1.30)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")
    fig.suptitle(
        f"Biomarker Event Sequence  —  {panel_label}",
        fontsize=TITLE_SIZE, fontweight="bold", y=0.97,
    )
    ax.text(0.35, n - 0.05, "Biomarker",
            fontsize=LABEL_SIZE, fontweight="bold", ha="center", va="bottom")
    ax.text(0.80, n - 0.05, "Position",
            fontsize=LABEL_SIZE, fontweight="bold", ha="center", va="bottom")
    ax.text(1.12, n - 0.05, "Confidence",
            fontsize=LABEL_SIZE, fontweight="bold", ha="center", va="bottom")
    ax.axhline(n - 0.15, xmin=0.0, xmax=1.0,
               color="#444444", linewidth=0.8, linestyle="-")
    
    for i, row in seq.iterrows():
        y_pos   = n - 1 - i 
        pos_num = int(row["position"])
        bio     = str(row["biomarker"])
        conf    = float(row["positional_confidence"])

        if i % 2 == 0:
            bg = plt.Rectangle((-0.05, y_pos - 0.45), 1.40, 0.90,
                               color="#F5F5F5", zorder=0)
            ax.add_patch(bg)
        badge = plt.Circle((0.04, y_pos), 0.28, color="#2166AC",
                            zorder=2, transform=ax.transData)
        ax.add_patch(badge)
        ax.text(0.04, y_pos, str(pos_num),
                fontsize=8, fontweight="bold", color="white",
                ha="center", va="center", zorder=3)
        ax.annotate(
            "", xy=(0.14, y_pos), xytext=(0.08, y_pos),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
            zorder=3,
        )
        bio_short = _BIOMARKER_LABELS.get(bio, bio.replace("_", " ").title())
        bio_short = bio_short.replace("\n", " ")
        ax.text(0.54, y_pos, bio_short,
                fontsize=FONT_SIZE, ha="center", va="center", zorder=3)
        bar_x0 = 0.72
        bar_w = 0.30
        bar_h = 0.22
        bar_y = y_pos - bar_h / 2
        ax.add_patch(plt.Rectangle(
            (bar_x0, bar_y), bar_w, bar_h,
            color="#DDEEFF", zorder=2,
        ))
        ax.add_patch(plt.Rectangle(
            (bar_x0, bar_y), bar_w * conf, bar_h,
            color="#2166AC", zorder=3, alpha=0.85,
        ))
        ax.add_patch(plt.Rectangle(
            (bar_x0, bar_y), bar_w, bar_h,
            fill=False, edgecolor="#888888", linewidth=0.6, zorder=4,
        ))
        ax.text(1.12, y_pos, f"{conf:.2f}",
                fontsize=FONT_SIZE, ha="center", va="center",
                color="#333333", fontweight="bold" if conf >= 0.50 else "normal",
                zorder=3)
        ax.text(0.04, -0.12, "Pos.",
            fontsize=7, ha="center", va="center", color="#555555")
        ax.text(0.54, -0.12, "Biomarker",
            fontsize=7, ha="center", va="center", color="#555555")
        ax.text(0.87, -0.12, "Bootstrap confidence",
            fontsize=7, ha="center", va="center", color="#555555")
        ax.text(0.54, -0.40,
            "Confidence = proportion of bootstrap resamples placing biomarker at this position",
            fontsize=6.5, ha="center", va="center", color="#777777",
            style="italic")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = FIGURES_DIR / f"fig02_event_sequence_{panel_name}.png"
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved: {out_path}")
        return out_path

def fig05_combined_forest_plot() -> Path:
    """Generate a combined forest plot for the stage×time LME interaction.
 
    Displays only the stage:years_since_baseline coefficient — the primary
    effect of interest — for both panels and both outcomes.
 
    A positive β for CDR-SB means faster CDR increase per unit stage.
    A negative β for MMSE means faster MMSE decline per unit stage.
 
    Layout: four rows, two panel groups (MRI-only top, MRI+Amyloid bottom),
    separated by a subtle horizontal rule.
 
    Saves to  results/figures/fig05_combined_forest_plot.png
    """
    lme_df = _load_lme_results()
    interaction_param = "stage:years_since_baseline"
    df = lme_df[lme_df["parameter"] == interaction_param].copy()
 
    if df.empty:
        raise ValueError(
            f"No rows with parameter == '{interaction_param}' found in "
            f"{LONG_DIR / 'mixed_effects_results.csv'}.\n"
            "Verify that 04_longitudinal_validation.py ran successfully."
        )

    plot_rows = [
        ("mri_only", "MMSE", 3, "MRI-only  |  MMSE", COL_MRI),
        ("mri_only",    "CDRSUM", 2, "MRI-only  |  CDR-SB", COL_MRI),
        ("mri_amyloid", "MMSE",   1, "MRI + Amyloid  |  MMSE", COL_AMYLOID),
        ("mri_amyloid", "CDRSUM", 0, "MRI + Amyloid  |  CDR-SB", COL_AMYLOID),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
 
    y_positions = [r[2] for r in plot_rows]
    y_labels = [r[3] for r in plot_rows]
    y_colours = [r[4] for r in plot_rows]
    all_lo, all_hi = [], []
 
    for panel_name, outcome, y_pos, label, colour in plot_rows:
        subset = df[(df["panel"] == panel_name) & (df["outcome"] == outcome)]
        if subset.empty:
            print(f"  WARNING: no LME row for panel={panel_name}, outcome={outcome}. Skipping.")
            continue
 
        row = subset.iloc[0]
        beta = float(row["beta"])
        lo = float(row["ci_95_low"])
        hi = float(row["ci_95_high"])
        p_str = str(row["p_value"])
        n_subj = int(row["n_subjects"])
        n_obs = int(row["n_observations"])
        sig = bool(row["significant"])
 
        all_lo.append(lo)
        all_hi.append(hi)
        ax.plot([lo, hi], [y_pos, y_pos],
                color=colour, lw=2.0, solid_capstyle="round",
                alpha=CI_ALPHA, zorder=2)
        marker   = "D" if sig else "o"
        msize    = 8 if sig else 6
        ax.scatter(beta, y_pos,
                   color=colour, s=msize ** 2,
                   marker=marker, zorder=4, edgecolors="white",
                   linewidths=0.8)
        p_display = "p<0.001" if p_str == "<0.001" else f"p={p_str}"
        ax.text(
            hi + 0.005, y_pos,
            f"β = {beta:+.3f} [{lo:.3f}, {hi:.3f}]  {p_display}  n={n_subj}",
            va="center", ha="left", fontsize=6.8, color="#333333",
        )

    ax.axvline(0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)
    ax.axhline(1.5, color="#AAAAAA", linewidth=0.6, linestyle="-", xmin=0, xmax=0.5)
    padding = 0.04
    x_min   = min(all_lo) - padding if all_lo else -0.35
    x_max   = max(all_hi) + padding if all_hi else  0.35
    ax.set_xlim(x_min, x_max + 0.28) 
    ax.set_ylim(-0.7, len(plot_rows) - 0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE)
    ax.set_xlabel(
        "Regression coefficient β  (stage × years since baseline)",
        fontsize=LABEL_SIZE, labelpad=6,
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune="both"))
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.text(
        x_min - 0.01, 2.5, "MRI-only",
        fontsize=7.5, fontweight="bold", ha="right", va="center",
        color=COL_MRI, rotation=90,
    )
    ax.text(
        x_min - 0.01, 0.5, "MRI+Amyloid",
        fontsize=7.5, fontweight="bold", ha="right", va="center",
        color=COL_AMYLOID, rotation=90,
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color=COL_MRI,     linestyle="None",
               markersize=6, label="MRI-only"),
        Line2D([0], [0], marker="D", color=COL_AMYLOID, linestyle="None",
               markersize=6, label="MRI + Amyloid"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC", handlelength=1.2)
    ax.text(
        x_min + 0.002, -0.62,
        "← better cognitive performance",
        fontsize=6.5, color="#777777", style="italic", va="bottom",
    )
    ax.text(
        x_max + 0.27, -0.62,
        "worse cognitive performance →",
        fontsize=6.5, color="#777777", style="italic", va="bottom", ha="right",
    )
 
    ax.set_title(
        "Mixed-Effects Model: Baseline EBM Stage × Years Since Baseline",
        fontsize=TITLE_SIZE, fontweight="bold", pad=8,
    )
 
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("none")
 
    plt.tight_layout()
 
    out_path = FIGURES_DIR / "fig05_combined_forest_plot.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path

def main() -> None:
    """Generate all publication figures from existing pipeline outputs."""
    print("=" * 60)
    print("05_make_publication_figures.py")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)
 
    panels = [
        ("mri_only",    "MRI-only"),
        ("mri_amyloid", "MRI + Amyloid"),
    ]
 
    errors: list[str] = []

    print("\nFigure 1 — Positional Variance Diagram heatmaps")
    for panel_name, panel_label in panels:
        try:
            fig01_pvd_heatmap(panel_name)
        except FileNotFoundError as exc:
            msg = f"  SKIP [{panel_label}] PVD heatmap — {exc}"
            print(msg)
            errors.append(msg)
        except Exception as exc:
            msg = f"  ERROR [{panel_label}] PVD heatmap — {exc}"
            print(msg)
            errors.append(msg)
            raise
    
    print("\nFigure 2 — Event sequence diagrams")
    for panel_name, panel_label in panels:
        try:
            fig02_event_sequence(panel_name)
        except FileNotFoundError as exc:
            msg = f"  SKIP [{panel_label}] event sequence — {exc}"
            print(msg)
            errors.append(msg)
        except Exception as exc:
            msg = f"  ERROR [{panel_label}] event sequence — {exc}"
            print(msg)
            errors.append(msg)
            raise
    
    print("\nFigure 5 — Combined forest plot")
    try:
        fig05_combined_forest_plot()
    except FileNotFoundError as exc:
        msg = f"  SKIP combined forest plot — {exc}"
        print(msg)
        errors.append(msg)
    except Exception as exc:
        msg = f"  ERROR combined forest plot — {exc}"
        print(msg)
        errors.append(msg)
        raise
    
    print()
    print("=" * 60)
    all_expected = [
        FIGURES_DIR / "fig01_pvd_heatmap_mri_only.png",
        FIGURES_DIR / "fig01_pvd_heatmap_mri_amyloid.png",
        FIGURES_DIR / "fig02_event_sequence_mri_only.png",
        FIGURES_DIR / "fig02_event_sequence_mri_amyloid.png",
        FIGURES_DIR / "fig05_combined_forest_plot.png",
    ]
    n_present = sum(1 for p in all_expected if p.exists())
    print(f"Figures generated: {n_present} / {len(all_expected)}")
    for p in all_expected:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")
 
    if errors:
        print(f"\n{len(errors)} figure(s) skipped (missing upstream files):")
        for e in errors:
            print(f"  {e}")
 
    print("=" * 60)
 
if __name__ == "__main__":
    main()
    

    
    

    




        
 

        



    

 




