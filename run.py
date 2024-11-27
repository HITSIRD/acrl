import argparse

import torch

from acrl.util.parameter_parser import parse_parameters


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="wasserstein",
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds", "acrl"])
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac", "td3", "ddpg"])
    parser.add_argument("--env", type=str, default="minigrid",
                        choices=["point_mass_2d", "maze", "minigrid-a", "minigrid-b", "minigrid-c", "minigrid-d",
                                 "minigrid-e", "minigrid-f", "minigrid-g", "minigrid-h", "u-maze", "n-maze",
                                 "ant-u-maze", "swimmer-maze", "fetchpush"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_cores", type=int, default=8)

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)
    torch.set_num_threads(args.n_cores)

    if args.env == "point_mass_2d":
        from acrl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "maze":
        from acrl.experiments import MazeExperiment
        exp = MazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-a':
        from acrl.experiments.minigrid_a_experiment import MinigridAExperiment
        exp = MinigridAExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-b':
        from acrl.experiments.minigrid_b_experiment import MinigridBExperiment
        exp = MinigridBExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-c':
        from acrl.experiments.minigrid_c_experiment import MinigridCExperiment
        exp = MinigridCExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-d':
        from acrl.experiments.minigrid_d_experiment import MinigridDExperiment
        exp = MinigridDExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-e':
        from acrl.experiments.minigrid_e_experiment import MinigridEExperiment
        exp = MinigridEExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-f':
        from acrl.experiments.minigrid_f_experiment import MinigridFExperiment
        exp = MinigridFExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-g':
        from acrl.experiments.minigrid_g_experiment import MinigridGExperiment
        exp = MinigridGExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'minigrid-h':
        from acrl.experiments.minigrid_h_experiment import MinigridHExperiment
        exp = MinigridHExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'u-maze':
        from acrl.experiments.u_maze_experiment import UMazeExperiment
        exp = UMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'n-maze':
        from acrl.experiments.n_maze_experiment import NMazeExperiment
        exp = NMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'ant-u-maze':
        from acrl.experiments.ant_u_maze_experiment import AntUMazeExperiment
        exp = AntUMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'fetchpush':
        from acrl.experiments.fetchpush_experiment import FetchPushExperiment
        exp = FetchPushExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
