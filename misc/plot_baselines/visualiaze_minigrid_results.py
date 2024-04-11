import os
import matplotlib
from matplotlib import cm, ticker
from misc.util import add_plot
import matplotlib.pyplot as plt
from acrl.experiments import MinigridAExperiment, MinigridBExperiment, MinigridCExperiment, MinigridDExperiment, \
    MinigridEExperiment, MinigridFExperiment, MinigridGExperiment, MinigridHExperiment

# plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
#                               r'\newcommand{\currot}{\textsc{currot}}'
#                               r'\newcommand{\sprl}{\textsc{sprl}}'
#                               r'\newcommand{\alpgmm}{\textsc{alp-gmm}}'
#                               r'\newcommand{\goalgan}{\textsc{goalgan}}'
#                               r'\newcommand{\acl}{\textsc{acl}}'
#                               r'\newcommand{\vds}{\textsc{vds}}'
#                               r'\newcommand{\plr}{\textsc{plr}}')
plt.rcParams.update({
    "text.usetex": True,
    "font.serif": "Helvetica",
    "font.family": "serif"
})

FONT_SIZE = 8
TICK_SIZE = 6

plt.rcParams['axes.formatter.limits'] = [-1, 1]
plt.rcParams['font.size'] = TICK_SIZE


def full_plot(path=None, base_log_dir="logs"):
    fig = plt.figure(figsize=(8, 4))

    ax = fig.subplots(2, 4)
    # fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.8, wspace=0.3, hspace=0.3)

    lines = performance_plot(ax[0, 0], base_log_dir=base_log_dir, title='Easy-A', exp_cls=MinigridHExperiment,
                     parameters={'ACRL_LAMBDA': 0.25}, x_label=False, iter_steps=1000)
    performance_plot(ax[0, 1], base_log_dir=base_log_dir, title='Easy-B', exp_cls=MinigridAExperiment,
                     parameters={'ACRL_LAMBDA': 0.25}, x_label=False, y_label=False, iter_steps=1000)
    performance_plot(ax[0, 2], base_log_dir=base_log_dir, title='Easy-C', exp_cls=MinigridDExperiment,
                     parameters={'ACRL_LAMBDA': 0.25}, x_label=False, y_label=False, iter_steps=1000)
    performance_plot(ax[0, 3], base_log_dir=base_log_dir, title='Medium-A', exp_cls=MinigridBExperiment,
                     parameters={'ACRL_LAMBDA': 0.25}, x_label=False, y_label=False, iter_steps=2500)
    performance_plot(ax[1, 0], base_log_dir=base_log_dir, title='Medium-B', exp_cls=MinigridEExperiment,
                             parameters={'ACRL_LAMBDA': 0.25}, iter_steps=2500)
    performance_plot(ax[1, 1], base_log_dir=base_log_dir, title='Medium-C', exp_cls=MinigridFExperiment,
                     parameters={'ACRL_LAMBDA': 0.25}, y_label=False, iter_steps=2500)
    performance_plot(ax[1, 2], base_log_dir=base_log_dir, title='Hard-A', exp_cls=MinigridCExperiment,
                     parameters={'ACRL_LAMBDA': 0.5}, y_label=False, iter_steps=10000)
    performance_plot(ax[1, 3], base_log_dir=base_log_dir, title='Hard-B', exp_cls=MinigridGExperiment,
                     parameters={'ACRL_LAMBDA': 0.25}, y_label=False, iter_steps=10000)

    fig.legend(lines,
               [r"\sprl", "Random", "Default", "CURROT", "Goal GAN", "ALP-GMM", r"\acl", "PLR", "VDS",
                "ACRL (ours)"],
               fontsize=FONT_SIZE, loc='upper center', ncol=8)

    fig.tight_layout(rect=[0, 0, 1.0, 0.92])

    if path is None:
        fig.show()
    else:
        fig.savefig(path)


def performance_plot(ax=None, base_log_dir="logs", title=None, exp_cls=None, parameters=None, x_label=True,
                     y_label=True, iter_steps=1000):
    assert exp_cls is not None
    assert title is not None
    lines = []
    for method, color in zip(["self_paced", "random", "default", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds", "acrl"],
                             ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]):
        exp = exp_cls(base_log_dir, method, "ppo", parameters=parameters, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color, iter_steps=iter_steps))

    if y_label:
        ax.set_ylabel(r"Episodic Return", fontsize=FONT_SIZE)
    if x_label:
        ax.set_xlabel(r"Train Steps", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)

    ax.ticklabel_format(style='sci', axis='x')
    ax.ticklabel_format(style='plain', axis='y')

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_yticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])
    # ax.set_ylim([0, 1])
    ax.grid()

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    plt.grid()
    # plt.tight_layout(pad=1.0)
    return lines


if __name__ == "__main__":
    os.makedirs("./figures/minigrid", exist_ok=True)
    # base_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
    base_log_dir = "./logs"
    # acrl_lambda = 0.25
    full_plot(path='./figures/minigrid/minigrid.pdf', base_log_dir=base_log_dir)
