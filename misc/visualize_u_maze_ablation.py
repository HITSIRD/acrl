import os
import pickle
import matplotlib
import numpy as np
from matplotlib import cm

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

FONT_SIZE = 15
TICK_SIZE = 6


def plot_performance(log_dir):
    fig = plt.figure(figsize=(7, 4))
    ax = plt.Axes(fig, [0.13, 0.2, 0.62, 0.72])
    fig.add_axes(ax)

    # mean = y.mean(axis=0)
    # std = y.std(axis=0)
    #
    # plt.plot(x, mean, color='darkorange', label='Random')
    # plt.fill_between(x, mean - std, mean + std, color='r', alpha=0.1)

    ax = plt.gca()
    lines = []
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0.01_ACRL_LAMBDA=0', ax, 'C0'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0.25_ACRL_LAMBDA=0', ax, 'C1'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0.5_ACRL_LAMBDA=0', ax, 'C2'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0.75_ACRL_LAMBDA=0', ax, 'C3'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=1.0_ACRL_LAMBDA=0', ax, 'C4'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0_ACRL_LAMBDA=0.25', ax, 'C5'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0_ACRL_LAMBDA=0.5', ax, 'C6'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0_ACRL_LAMBDA=0.75', ax, 'C7'))
    lines.append(add_plot(log_dir + 'ppo_ACRL_EBU_RATIO=0_ACRL_LAMBDA=1.0', ax, 'C8'))

    # ax.tickabel_format(style='sci', scilimits=(-1, 2), axis='x')
    # ax.get_xaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.xaxis.get_offet_text.set_fontsize(15)
    fig.legend(lines,
               ['$\lambda=0$', 'EBU $\lambda=0.25$', 'EBU $\lambda=0.5$', 'EBU $\lambda=0.75$', 'EBU $\lambda=1.0$',
                'LSP $\lambda=0.25$', 'LSP $\lambda=0.5$', 'LSP $\lambda=0.75$', 'LSP $\lambda=1.0$'],
               fontsize=12, loc='right')

    plt.xlabel('Train Steps ($\\times 10^6$)', size=FONT_SIZE)
    plt.ylabel('Episodic Return', size=FONT_SIZE)
    # plt.title('u-maze', size=FONT_SIZE)

    ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_xticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"1.0"])
    ax.set_yticks([-100, -80, -60, -40, -20])
    ax.set_yticklabels([r"$-100$", r"$-80$", r"$-60$", r"$-40$", r"$-20$"])
    plt.xticks(size=FONT_SIZE)
    plt.yticks(size=FONT_SIZE)
    plt.grid()

    plt.tight_layout()
    plt.savefig('figures/u_maze_ablation.pdf')
    # plt.show()


if __name__ == "__main__":
    os.makedirs("../figures", exist_ok=True)
    # base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
    base_log_dir = "/home/wenyongyan/文档/currot-icml_副本/logs/u_maze/acrl/"
    plot_performance(base_log_dir)
