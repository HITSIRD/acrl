import argparse

import torch

from acrl.util.parameter_parser import parse_parameters


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="wasserstein",
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds", "acrl"])
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac", "td3"])
    parser.add_argument("--env", type=str, default="minigrid",
                        choices=["point_mass_2d", "maze", "minigrid-a", "minigrid-b", "minigrid-c", "u-maze",
                                 "ant-maze", "swimmer-maze"])
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
    elif args.env == 'u-maze':
        from acrl.experiments.u_maze_experiment import UMazeExperiment
        exp = UMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'ant-maze':
        from acrl.experiments.ant_maze_experiment import AntMazeExperiment
        exp = AntMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == 'swimmer-maze':
        from acrl.experiments.swimmer_maze_experiment import SwimmerMazeExperiment
        exp = SwimmerMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
