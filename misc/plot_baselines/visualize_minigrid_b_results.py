import os
import pickle
import matplotlib
import numpy as np
from matplotlib import cm
from misc.util import add_plot
import matplotlib.pyplot as plt
from acrl.experiments import MinigridBExperiment

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

FONT_SIZE = 15
TICK_SIZE = 6


def performance_plot(ax=None, path=None, base_log_dir="logs", acrl_lambda=0.5):
    if ax is None:
        f = plt.figure(figsize=(4.5, 3))
        ax = plt.Axes(f, [0.13, 0.19, 0.86, 0.71])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = []
    for method, color in zip(["self_paced", "random", "default", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds", "acrl"],
                             ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]):
        exp = MinigridBExperiment(base_log_dir, method, "ppo", {'ACRL_LAMBDA': acrl_lambda}, seed=1)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    ax.set_title('MiniGrid-B', fontsize=FONT_SIZE)
    ax.set_ylabel(r"Episodic Return", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_xlabel(r"Train Steps ($\times 10^3$)", fontsize=FONT_SIZE, labelpad=2.)

    if show:
        ax.legend(lines,
                  [r"\sprl", "Random", "Default", "CURROT", "Goal GAN", "ALP-GMM", r"\acl", "PLR", "VDS",
                   "ACRL (ours)"],
                  fontsize=11, loc='upper left')

    ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_xticklabels([r"$0$", r"$100$", r"$200$", r"$30$", r"$400$", r"500"])
    # ax.set_xlim([0, 150])

    # ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
    # ax.set_yticklabels([r"$-1$", r"$-0.5$", r"$0$", r"$0.5$", r"$1$"])
    # ax.set_ylim([-0.1, 0.8])
    ax.grid()

    # ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    # ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    plt.xticks(size=FONT_SIZE)
    plt.yticks(size=FONT_SIZE)

    if show:
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
    else:
        return lines


if __name__ == "__main__":
    os.makedirs("./figures/minigrid_b", exist_ok=True)
    # base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
    base_log_dir = "./logs"
    # acrl_lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acrl_lambda = 0.25
    performance_plot(path='./figures/minigrid_b/' + str(acrl_lambda) + '.pdf', base_log_dir=base_log_dir,
                     acrl_lambda=acrl_lambda)
