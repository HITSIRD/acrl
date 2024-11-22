import os
import pickle
import matplotlib
import numpy as np
from matplotlib import cm

from acrl.experiments.ant_u_maze_experiment import AntUMazeExperiment
from acrl.experiments.u_maze_experiment import UMazeExperiment
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


def performance_plot(ax=None, path=None, base_log_dir="logs", acrl_lambda=0.5):
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
        exp = UMazeExperiment(base_log_dir, method, "sac", {'ACRL_LAMBDA': acrl_lambda}, seed=1)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color, iter_steps=5000))

    ax.set_title('U-Maze', fontsize=FONT_SIZE)
    ax.set_ylabel(r"Success Rate", fontsize=FONT_SIZE)
    ax.set_xlabel(r"Train Steps", fontsize=FONT_SIZE)

    if show:
        ax.legend(lines,
                  [r"\sprl", "Random", "Default", r"CURROT", r"Goal GAN", r"ALP-GMM", r"ACL", r"PLR", r"VDS",
                   "ACRL (ours)"],
                  fontsize=10, loc='upper left')

    # ax.set_xticks([0, 40, 80, 120, 160, 200])
    # ax.set_xticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"1.0"])
    # ax.set_xlim([0, 200])

    ax.ticklabel_format(style='sci', axis='x')
    ax.ticklabel_format(style='plain', axis='y')

    # ax.set_yticks([-100, -80, -60, -40, -20])
    # ax.set_yticklabels([r"$-100$", r"$-80$", r"$-60$", r"$-40$", r"$-20$"])
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
    exp_name = 'u_maze'
    os.makedirs("./figures/" + exp_name, exist_ok=True)
    # base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
    base_log_dir = "./logs"
    acrl_lambda = 0.9
    performance_plot(path='./figures/' + exp_name + '/sac_' + str(acrl_lambda) + '.pdf', base_log_dir=base_log_dir,
                     acrl_lambda=acrl_lambda)
