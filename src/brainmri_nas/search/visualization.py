"""NSGA-II Pareto front plots, marking the TOPSIS-selected architecture.

Both plots use a *consistent* "lower is better" sign convention on every
axis -- the same `-log_synflow`, `-zico`, `flops_billion` space NSGA-II
actually minimized (`topsis.nsga2_objective_matrix`), not the natural
"higher is better" proxy values. That choice isn't cosmetic: for two
objectives that are genuinely in tension, a real Pareto front only forms a
monotonic downward-sloping curve/staircase when sorted by one *minimize*
axis and plotted against another *minimize* axis -- mixing a "lower is
better" axis (FLOPs) with a "higher is better" one (raw SynFlow/ZiCO) makes
a genuine front slope *upward* instead, and isn't what people mean by "a
Pareto curve." An earlier version of this plot did exactly that (plus drew
4 overlapping translucent trisurf meshes and a connector line only sorted
by one of three axes), which produced a chaotic crumpled-looking mesh
instead of a readable trade-off -- see git history if you want to compare.

`save_pareto_front_3d_plot` shows the top `max_fronts_to_plot` nondominated
fronts (not just front 0) as translucent trisurf sheets, one color per
front rank -- reinstating the legacy repo's layered-sheet look, but now
with the consistent sign convention above so the sheets actually stack in
a coherent direction instead of fighting a mixed-direction axis. A 3D
Pareto front is still fundamentally a *surface*, not a curve, so even with
consistent axes this won't look like a single clean line -- that's
inherent to having 3 objectives, not a rendering bug.
`save_pareto_front_2d_plot` complements it with two genuine 2-objective
Pareto staircases (FLOPs vs. each proxy), which *are* mathematically
guaranteed to be monotonic, computed directly rather than by projecting
the 3-objective front (a 3D front's 2D projection is not itself
guaranteed monotonic, since a point can dominate on the hidden third
axis).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from brainmri_nas.search.topsis import DEFAULT_CRITERIA, finite_valid_records, nsga2_objective_matrix


def _setup_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_pareto_front_3d_plot(
    archive: list[dict],
    selected_architecture: dict,
    path: str | Path,
    *,
    max_fronts_to_plot: int = 4,
) -> None:
    plt = _setup_figure()

    valid = finite_valid_records(archive, DEFAULT_CRITERIA)
    if not valid:
        return  # nothing scoreable to plot -- callers can check the file wasn't written

    objectives = nsga2_objective_matrix(valid)  # columns: -log_synflow, -zico, flops_billion (all "lower is better")
    fronts = NonDominatedSorting().do(objectives, only_non_dominated_front=False)
    fronts_to_plot = fronts[: min(max_fronts_to_plot, len(fronts))]
    plotted_mask = np.zeros(len(valid), dtype=bool)
    for idx in fronts_to_plot:
        plotted_mask[idx] = True

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    # Candidates beyond max_fronts_to_plot: small, faint, for scale/context only.
    beyond = objectives[~plotted_mask]
    if len(beyond) > 0:
        ax.scatter(
            beyond[:, 2], beyond[:, 0], beyond[:, 1],
            s=10, alpha=0.15, c="gray", label="other candidates", zorder=1,
        )

    cmap = plt.get_cmap("viridis_r")
    denom = max(1, len(fronts_to_plot) - 1)

    for front_rank, front_idx in enumerate(fronts_to_plot):
        front = objectives[front_idx]
        order = np.argsort(front[:, 2])  # sort by FLOPs -- gives trisurf's triangulation a sane starting order
        front = front[order]
        x, y, z = front[:, 2], front[:, 0], front[:, 1]

        is_pareto = front_rank == 0
        color = cmap(front_rank / denom)
        label = "Front 0 / Pareto front" if is_pareto else f"Front {front_rank}"

        ax.scatter(
            x, y, z,
            s=55 if is_pareto else 30,
            alpha=0.95 if is_pareto else 0.7,
            color=color,
            label=label,
            zorder=10 if is_pareto else 5,
        )

        if len(front) >= 3:
            try:
                ax.plot_trisurf(
                    x, y, z,
                    color=color,
                    alpha=0.45 if is_pareto else 0.22,
                    edgecolor="black",
                    linewidth=0.25,
                    antialiased=True,
                )
            except Exception:
                pass  # a degenerate (e.g. collinear) front can't always form a surface -- the scatter still shows it

    sel_x = selected_architecture["flops_billion"]
    sel_y = -selected_architecture["log_synflow"]
    sel_z = -selected_architecture["zico"]
    ax.scatter(
        [sel_x], [sel_y], [sel_z],
        marker="*", s=340, c="red", edgecolors="black", linewidths=1.2,
        label="TOPSIS selected", zorder=20,
    )
    ax.text(sel_x, sel_y, sel_z, "  selected", fontsize=9)

    ax.set_xlabel("FLOPs (B) -- lower is better")
    ax.set_ylabel("-log10(SynFlow) -- lower is better")
    ax.set_zlabel("-ZiCO -- lower is better")
    ax.set_title("NSGA-II nondominated fronts (all axes: lower is better)")
    ax.view_init(elev=22, azim=-60)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pareto_staircase(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The genuine 2-objective (both minimize) non-dominated front, sorted --
    guaranteed monotonic non-increasing in y as x increases."""
    order = np.argsort(x)
    x, y = x[order], y[order]

    front_x, front_y = [], []
    best_y = np.inf
    for xi, yi in zip(x, y):
        if yi < best_y:
            front_x.append(xi)
            front_y.append(yi)
            best_y = yi
    return np.array(front_x), np.array(front_y)


def save_pareto_front_2d_plot(
    archive: list[dict],
    selected_architecture: dict,
    path: str | Path,
) -> None:
    plt = _setup_figure()

    valid = finite_valid_records(archive, DEFAULT_CRITERIA)
    if not valid:
        return

    flops = np.array([r["flops_billion"] for r in valid], dtype=float)
    neg_synflow = np.array([-r["log_synflow"] for r in valid], dtype=float)
    neg_zico = np.array([-r["zico"] for r in valid], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], neg_synflow, "-log10(SynFlow) -- lower is better", -selected_architecture["log_synflow"]),
        (axes[1], neg_zico, "-ZiCO -- lower is better", -selected_architecture["zico"]),
    ]

    for ax, y_values, y_label, selected_y in panels:
        ax.scatter(flops, y_values, s=16, alpha=0.3, c="gray", label="dominated candidates")

        step_x, step_y = _pareto_staircase(flops, y_values)
        ax.step(step_x, step_y, where="post", color="tab:blue", linewidth=2.0, label="Pareto front")
        ax.scatter(step_x, step_y, s=30, color="tab:blue", zorder=5)

        ax.scatter(
            [selected_architecture["flops_billion"]], [selected_y],
            marker="*", s=280, c="red", edgecolors="black", linewidths=1.0,
            label="TOPSIS selected", zorder=20,
        )

        ax.set_xlabel("FLOPs (B) -- lower is better")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("2-objective Pareto fronts (each pair, lower-left is better)")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
