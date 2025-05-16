import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SARCOS", help="Dataset name")
parser.add_argument("--N_seeds", type=int, default=10, help="Number of seeds")

args = parser.parse_args()

colors = ["#FE7F2D", "#780116","#FCCA46",  "#A1C181", "#619B8A", "#1D4EFC", "#C1007E","#0B3D91",  "#00D4FF", "#8C00FF"]
SIZE_TINY = 10
SIZE_SMALL = 12
SIZE_DEFAULT = 14
SIZE_LARGE = 16
plt.rc("font", weight="normal")
plt.rc("font", size=SIZE_DEFAULT)
plt.rc("axes", titlesize=SIZE_LARGE)
plt.rc("axes", labelsize=SIZE_LARGE)
plt.rc("xtick", labelsize=SIZE_DEFAULT)
plt.rc("ytick", labelsize=SIZE_DEFAULT)


def make_plot(
    x, y, labels, colors, ax, linestyle="solid", offsets=None, ylim=None, ticks_y=True
):
    if offsets is None:
        offsets = [
            0.0,
        ] * len(labels)

    # Plot each of the main lines
    for i, label in enumerate(labels):
        y_mean = np.mean(y[i], axis=0)
        y_std = np.std(y[i], axis=0)

        ax.plot(
            x,
            np.quantile(y[i], 0.5, axis=0),
            label=label,
            color=colors[i],
            linewidth=2,
            linestyle=linestyle,
        )

        ax.fill_between(
            x,
            np.quantile(y[i], 0.1, axis=0),
            np.quantile(y[i], 0.9, axis=0),
            color=colors[i],
            linewidth=2,
            alpha=0.3,
        )

        ax.text(
            x[-1] * 1.01,
            np.median(y[i], axis=0)[-1] + offsets[i],
            label,
            color=colors[i],
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    if ticks_y:
        ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(x), max(x))
    if ylim:
        ax.set_ylim(ylim)

dataset = args.dataset
N_SEEDS = args.N_seeds

import yaml

with open(f"config_doegp_{dataset}.yaml", "r") as f:
    config = yaml.safe_load(f)

offsets = config["offsets"]

if "ax_label_size" in config:
    ax_label_size = config["ax_label_size"]
else:
    ax_label_size = SIZE_DEFAULT

plt.rc("xtick", labelsize=ax_label_size)


logws_egs = []
reward_t_eg_s = []
logws_bmas = []
reward_t_bmas = []
static_weights_s = []
reward_t_statics = []
logws_softbayes_s = []
reward_t_softbayes_s = []
logws_dmas = []
reward_t_dmas = []
ws_ons = []
reward_t_ons = []
ws_ons_with_gamma = []
reward_t_ons_with_gamma = []

for seed in range(N_SEEDS):
    results = np.load(f"results_doegp_dataset_{dataset}_seed_{seed}.npz")
    logws_egs.append(results["arr_0"])
    reward_t_eg_s.append(results["arr_1"])
    logws_bmas.append(results["arr_2"])
    reward_t_bmas.append(results["arr_3"])
    static_weights_s.append(results["arr_4"])
    reward_t_statics.append(results["arr_5"])

    ws_ons.append(results["arr_6"])
    reward_t_ons.append(results["arr_7"])
    logws_softbayes_s.append(results["arr_8"])
    reward_t_softbayes_s.append(results["arr_9"])

    logws_dmas.append(results["arr_10"])
    reward_t_dmas.append(results["arr_11"])

    ws_ons_with_gamma.append(results["arr_12"])
    reward_t_ons_with_gamma.append(results["arr_13"])

log_ws_eg = np.stack(logws_egs)
reward_t_eg = np.stack(reward_t_eg_s)
logws_bma = np.stack(logws_bmas)
reward_t_bma = np.stack(reward_t_bmas)
static_weights = np.stack(static_weights_s)
reward_t_static = np.stack(reward_t_statics)
logws_softbayes = np.stack(logws_softbayes_s)
reward_t_softbayes = np.stack(reward_t_softbayes_s)
logws_dma = np.stack(logws_dmas)
reward_t_dma = np.stack(reward_t_dmas)

ws_ons_with_gamma = np.stack(ws_ons_with_gamma)
reward_t_ons_with_gamma = np.stack(reward_t_ons_with_gamma)

ws_ons = np.stack(ws_ons)
reward_t_ons = np.stack(reward_t_ons)

N = ws_ons.shape[1] - 1

# Get median values
values = {
    "reward_t_eg": np.median(reward_t_eg, axis=0)[-1],
    "reward_t_bma": np.median(reward_t_bma, axis=0)[-1],
    "reward_t_static": np.median(reward_t_static, axis=0)[-1],
    "reward_t_ons": np.median(reward_t_ons, axis=0)[-1],
    "reward_t_softbayes": np.median(reward_t_softbayes, axis=0)[-1],
    "reward_t_dma": np.median(reward_t_dma, axis=0)[-1],
    "reward_t_ons_with_gamma": np.median(reward_t_ons_with_gamma, axis=0)[-1],
}

# Sort by value and print variable names in order
sorted_vars = sorted(values.items(), key=lambda x: x[1])
for var_name, _ in sorted_vars:
    print(var_name)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Suppose these are your data arrays (replace with your actual data)
import numpy as np

N = reward_t_bma.shape[1]
labels = ["EG", "O-BMA", "ONS", "Soft-Bayes", "BCRP", "DMA", "D-ONS"]
N_start = config["N_start"]
# Main figure and axes
fig, ax = plt.subplots(figsize=(4.25, 3))

# Main plot
# reward_t_bma
# reward_t_eg
# reward_t_softbayes
# reward_t_static
# reward_t_ons


make_plot(
    np.arange(N_start, N),
    [
        reward_t_eg[:, N_start:],
        reward_t_bma[:, N_start:],
        reward_t_ons[:, N_start:],
        reward_t_softbayes[:, N_start:],
        reward_t_static[:, N_start:],
        reward_t_dma[:, N_start:],
        reward_t_ons_with_gamma[:, N_start:],
    ],
    labels,
    colors,
    ax,
    offsets=offsets,
)


ax.set_xlabel("t")
ax.set_ylabel("Average PLL")

if "make_inset_plot" in config and config["make_inset_plot"]:
    # Create the inset axes in the lower-right corner
    # width="40%" and height="40%" is just an example sizing
    ax_inset = inset_axes(
        ax,
        width="40%",
        height="40%",
        loc="lower right",
        bbox_to_anchor=config["ax_inset_bbox"],
        bbox_transform=ax.transAxes,  # interpret bbox in ax's coordinate system
        borderpad=0,
    )


    def make_inset_plot(x, y, colors, ax, linestyle="solid"):
        for i, label in enumerate(labels):
            y_mean = np.mean(y[i], axis=0)
            y_std = np.std(y[i], axis=0)

            ax.plot(
                x,
                np.quantile(y[i], 0.5, axis=0),
                color=colors[i],
                linewidth=2,
                linestyle=linestyle,
            )

            # for q, alpha in zip([0.10, 0.25, 0.45], [0.5, 0.4, 0.3]):
            #     ax.fill_between(
            #         x,
            #         np.quantile(y[i], 0.5 - q, axis=0),
            #         np.quantile(y[i], 0.5 + q, axis=0),
            #         color=colors[i],
            #         linewidth=2,
            #         alpha=alpha,
            #     )

            ax.fill_between(
                x,
                np.quantile(y[i], 0.1, axis=0),
                np.quantile(y[i], 0.9, axis=0),
                color=colors[i],
                linewidth=2,
                alpha=0.3,
            )


    # Replot (or partially re-plot) only the region of interest in the inset
    make_inset_plot(
        np.arange(int(0.75 * N), N),
        [
            reward_t_eg[:, int(0.75 * N):],
            reward_t_bma[:, int(0.75 * N):],
            reward_t_ons[:, int(0.75 * N):],
            reward_t_softbayes[:, int(0.75 * N):],
            reward_t_static[:, int(0.75 * N):],
            reward_t_dma[:, int(0.75 * N):],
            reward_t_ons_with_gamma[:, int(0.75 * N):],
        ],
        colors,
        ax_inset,
    )
    # Zoom in on t > 1500
    ax_inset.set_xlim(int(0.75 * N), N)
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")


if config["ax_sci_notation"]:
    from matplotlib.ticker import FuncFormatter

    def b_times_10k(x, pos):
        # Special case for zero
        if x == 0:
            return "0"
        # Determine the exponent
        exponent = int(np.floor(np.log10(abs(x))))
        # Compute the coefficient
        coeff = x / 10**exponent
        # Format using math text (adjust precision as desired)
        return r"${0:.1f}\cdot10^{{{1}}}$".format(coeff, exponent)

    ax.xaxis.set_major_formatter(FuncFormatter(b_times_10k))
    
    if "make_inset_plot" in config and config["make_inset_plot"]:
        ax_inset.xaxis.set_major_formatter(FuncFormatter(b_times_10k))
        ax_inset.tick_params(axis="x", labelsize=SIZE_TINY)
        ax_inset.tick_params(axis="y", labelsize=SIZE_TINY)


plt.savefig(f"{dataset}_plls_with_zoom.png", dpi=400, bbox_inches="tight")

simu = 0
fig, axs = plt.subplots(1, 5, figsize=(8, 2), sharey=True)
M = log_ws_eg.shape[2]

axs[1].bar(range(M), np.exp(log_ws_eg)[simu, -1, :], color="black")
axs[1].set_xticks(range(M), [""] * M)
axs[1].set_title("EG")
axs[0].set_ylabel("$w_k$")

axs[0].bar(range(M), np.exp(logws_bma)[simu, -1, :], color="black")
axs[0].set_xticks(range(M), [""] * M)
axs[0].set_title("BMA")

axs[2].bar(range(M), ws_ons[simu, -1, :], color="black")
axs[2].set_xticks(range(M), [""] * M)
axs[2].set_title("ONS")

axs[3].bar(range(M), static_weights[simu, :], color="black")
axs[3].set_xticks(range(M), [""] * M)
axs[3].set_title("BCRP")

axs[4].bar(range(M), np.exp(logws_softbayes[simu, -1, :]), color="black")
axs[4].set_xticks(range(M), [""] * M)
axs[4].set_title("Soft-Bayes")

plt.savefig(f"weights_{dataset}_final.png", dpi=400, bbox_inches="tight")

colors_hist = [
    "#008000",
    "#808080",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#000000",
    "#FFA500",
]  # , '#FFA500', '#800080']#, '#FF0000', '#00FF00',  '#FFC0CB', '#800000', '#000080']

simu = 0
fig, axs = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

J = M

make_plot(
    np.arange(N - 100),
    [np.exp(logws_bma)[simu:simu+1, 100:, i] for i in range(M)],
    [""] * M,
    colors_hist,
    axs[0],
    offsets=[0] * M,
    ylim=None,
    ticks_y=[-0.05, 1],
)
axs[0].set_title("BMA")
axs[0].set_xticks(range(M), [""] * M)

make_plot(
    np.arange(N - 100),
    [np.exp(log_ws_eg)[simu:simu+1, 100:, i] for i in range(M)],
    [""] * M,
    colors_hist,
    axs[1],
    offsets=[0] * M,
    ylim=None,
    ticks_y=None,
)
axs[1].set_title("EG")
axs[1].scatter([N * 1.25] * M, static_weights[-1], color=colors_hist[:J])
axs[1].set_xticks(range(M), [""] * M)


make_plot(
    np.arange(N - 100),
    [ws_ons[simu:simu+1, 100:-1, i] for i in range(M)],
    [""] * M,
    colors_hist,
    axs[2],
    offsets=[0] * M,
    ylim=None,
    ticks_y=None,
)
axs[2].scatter([N * 1.25] * M, static_weights[-1], color=colors_hist[:J])
axs[2].set_title("ONS")
axs[2].set_xticks(range(M), [""] * M)


make_plot(
    np.arange(N - 100),
    [np.exp(logws_softbayes)[simu:simu+1, 100:, i] for i in range(M)],
    [""] * M,
    colors_hist,
    axs[3],
    offsets=[0] * M,
    ylim=None,
    ticks_y=None,
)
axs[3].scatter([N * 1.25] * M, static_weights[-1], color=colors_hist[:J])
axs[3].set_title("Soft-Bayes")
axs[3].set_xticks(range(M), [""] * M)


plt.savefig(f"weights_{dataset}_evolutions.png", dpi=400, bbox_inches="tight")