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


FONT_SIZE = 8
TICK_SIZE = 6

def add_precision_plot(log_dir, ax, color):
    xs = []
    ys = []

    if os.path.exists(log_dir):
        seed_dirs = [f for f in os.listdir(log_dir) if f.startswith("seed")]
        for seed_dir in seed_dirs:
            seed_path = os.path.join(log_dir, seed_dir)
            iteration_dirs = [d for d in os.listdir(seed_path) if d.startswith("iteration-")]
            unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
            idxs = np.argsort(unsorted_iterations)
            iterations = unsorted_iterations[idxs]

            avg_precs = []
            for iteration in iterations:
                with open(os.path.join(seed_path, "iteration-%d" % iteration, "context_trace.pkl"), "rb") as f:
                    trace = pickle.load(f)
                if len(trace[0]) != 0:
                    avg_precs.append(np.median(np.array(trace[-1])[:, -1]))

            if len(avg_precs) < len(iterations):
                avg_precs = [avg_precs[0]] * (len(iterations) - len(avg_precs)) + avg_precs

            xs.append(iterations)
            ys.append(avg_precs)

    if len(ys) > 0:
        print("Found %d completed seeds" % len(ys))
        min_length = np.min([len(y) for y in ys])
        iterations = iterations[0: min_length]
        ys = [y[0: min_length] for y in ys]

        mid = np.mean(ys, axis=0)
        # sem = np.std(ys, axis=0) / np.sqrt(len(ys))
        std = np.std(ys, axis=0)
        low = mid - std
        high = mid + std

        l, = ax.plot(iterations, mid, color=color, linewidth=1)
        ax.fill_between(iterations, low, high, color=color, alpha=0.5)
        return l
    else:
        return None


def precision_comparison(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(2.3, 1.4))
        ax = plt.Axes(f, [0.2, 0.23, 0.76, 0.52])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = []
    count = 0
    for method, color in zip(["self_paced", "random", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds"],
                             ["C0", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]):
        exp = MinigridExperiment(base_log_dir, method, "ppo", {}, seed=0)
        log_dir = os.path.dirname(exp.get_log_dir())

        try:
            lines.append(add_precision_plot(log_dir, ax, color))
        except Exception:
            lines.append(None)
        count += 1

    lines.append(ax.hlines(0.05, 0., 400, color="black", linestyle="--"))

    if show:
        f.legend(lines,
                 [r"\sprl", "Random", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds", "Min. Tol."],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.05, 1.01), ncol=4, columnspacing=0.4,
                 handlelength=0.6, handletextpad=0.25)

    ax.set_ylim([0.04, 18])
    ax.set_yscale("log")

    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_xticklabels([r"$0$", r"$30$", r"$60$", r"$90$", r"$120$", r"150"])
    ax.set_xlim([0, 150])
    ax.set_xlabel(r"Epoch", fontsize=FONT_SIZE, labelpad=2)
    ax.set_ylabel(r"Tolerance", fontsize=FONT_SIZE, labelpad=1)

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    if show:
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


def full_plot(path=None, base_log_dir="logs"):
    f = plt.figure(figsize=(4.83, 1.25))
    ax1 = f.add_axes([0.088, 0.24, 0.38, 0.56])
    ax2 = f.add_axes([0.6, 0.24, 0.38, 0.56])

    lines = performance_plot(ax1, base_log_dir=base_log_dir)
    precision_comparison(ax2, base_log_dir=base_log_dir)

    f.legend(lines,
             [r"\sprl", "Random", "Oracle", r"\currot \tiny{(Ours)}", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds"],
             fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(-0.01, 1.03), ncol=9, columnspacing=0.4,
             handlelength=0.9, handletextpad=0.25)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def performance_plot(ax=None, path=None, base_log_dir="logs"):
    if ax is None:
        f = plt.figure(figsize=(3, 2))
        ax = plt.Axes(f, [0.19, 0.23, 0.77, 0.52])
        f.add_axes(ax)
        show = True
    else:
        f = plt.gcf()
        show = False

    lines = []
    for method, color in zip(["self_paced", "random", "default", "wasserstein", "goal_gan", "alp_gmm",
                              "acl", "plr", "vds", "acrl"],
                             ["C0", "C2", (0.2, 0.2, 0.2), "C4", "C5", "C6", "C7", "C8", "C9", "C10"]):
        exp = MinigridBExperiment(base_log_dir, method, "ppo", {}, seed=1)
        log_dir = os.path.dirname(exp.get_log_dir())
        lines.append(add_plot(log_dir, ax, color))

    ax.set_ylabel(r"Episodic Return", fontsize=FONT_SIZE, labelpad=2.)
    ax.set_xlabel(r"Train Steps ($\times 10^3$)", fontsize=FONT_SIZE, labelpad=2.)

    if show:
        f.legend(lines,
                 [r"\sprl", "Random", "Default", r"\currot", r"\goalgan", r"\alpgmm", r"\acl", r"\plr", r"\vds", "acrl"],
                 fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(0.02, 1.01), ncol=4, columnspacing=0.4,
                 handlelength=0.9, handletextpad=0.25)
    ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_xticklabels([r"$0$", r"$100$", r"$200$", r"$300$", r"$400$", r"500"])
    ax.set_xlim([0, 200])

    # ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
    # ax.set_yticklabels([r"$-1$", r"$-0.5$", r"$0$", r"$0.5$", r"$1$"])
    ax.set_ylim([-0.1, 1.25])
    ax.grid()

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    plt.grid()
    # plt.tight_layout()
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
    performance_plot(path="figures/minigrid_b_performance.pdf", base_log_dir=base_log_dir)
