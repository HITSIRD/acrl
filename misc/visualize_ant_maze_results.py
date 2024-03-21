import os
import pickle
import matplotlib
import numpy as np
from matplotlib import cm

from deep_sprl.experiments.ant_maze_experiment import AntMazeExperiment
from deep_sprl.experiments.swimmer_maze_experiment import SwimmerMazeExperiment
from misc.util import add_plot
import matplotlib.pyplot as plt

plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\newcommand{\currot}{\textsc{currot}}'
                              r'\newcommand{\sprl}{\textsc{sprl}}'
                              r'\newcommand{\alpgmm}{\textsc{alp-gmm}}'
                              r'\newcommand{\goalgan}{\textsc{goalgan}}'
                              r'\newcommand{\acl}{\textsc{acl}}'
                              r'\newcommand{\vds}{\textsc{vds}}'
                              r'\newcommand{\plr}{\textsc{plr}}')
plt.rcParams.update({
    "text.usetex": True,
    "font.serif": "Helvetica",
    "font.family": "serif"
})


FONT_SIZE = 12
TICK_SIZE = 6
def performance_plot(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(4.5, 3))
        ax = plt.Axes(f, [0.16, 0.19, 0.82, 0.71])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = []
    for method, color in zip(["self_paced", "random", "default", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds", "acrl"],
                             ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]):
        exp = SwimmerMazeExperiment(base_log_dir, method, "ppo", {}, seed=1)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    ax.set_title('Swimmer-Maze', fontsize=FONT_SIZE)
    ax.set_ylabel(r"Episodic Return", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_xlabel(r"Train Steps ($\times 10^6$)", fontsize=FONT_SIZE, labelpad=2.)

    # if show:
    #     f.legend(lines,
    #              [r"\sprl", "Random", "Oracle", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds", "acrl"],
    #              fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.02, 1.01), ncol=4, columnspacing=0.4,
    #              handlelength=0.9, handletextpad=0.25)
    if show:
        ax.legend(lines,
                 [r"\sprl", "Random", "Default", r"CURROT", r"Goal GAN", r"ALP-GMM", r"ACL", r"PLR", r"VDS", "ACRL (ours)"],
                 fontsize=10, loc='upper left')

    ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_xticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"1.0"])
    # ax.set_xlim([0, 200])

    ax.set_yticks([-100, -80, -60, -40, -20])
    ax.set_yticklabels([r"$-100$", r"$-80$", r"$-60$", r"$-40$", r"$-20$"])
    # ax.set_ylim([-105, -10])
    ax.grid()

    # ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    # ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    plt.xticks(size=FONT_SIZE)
    plt.yticks(size=FONT_SIZE)
    plt.tight_layout()
    if show:
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
    else:
        return lines


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    # base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
    base_log_dir = "/home/wenyongyan/文档/currot-icml_副本/logs"
    performance_plot(path="figures/swimmer_maze_performance.pdf", base_log_dir=base_log_dir)
