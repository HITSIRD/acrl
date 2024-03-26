from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, ListedColormap

# import utils.helpers
from acrl.environments.minigrid.core.grid import Grid
from acrl.environments.minigrid.core.mission import MissionSpace
from acrl.environments.minigrid.core.world_object import Goal
from acrl.environments.minigrid.minigrid_env import MiniGridEnv

if torch.cuda.is_available():
    device = torch.device('cuda:0')
# elif torch.has_mps:
#    device = torch.device('mps')
else:
    device = torch.device('cpu')

domain = [[1, 1], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7]]
# [3, 4], [3, 5], [4, 4], [4, 5], [5, 4], [5, 5], [7, 4], [7, 5], [3, 7], [3, 8], [4, 7], [4, 8]

plt.rc('text.latex')
plt.rcParams.update({'text.usetex': True})


class AEnv(MiniGridEnv):
    """
    ## Description

    This environment has a key that the agent must pick up in order to unlock a
    goal and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning.

    ## Mission Space

    "use the key to open the door and then get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-DoorKey-5x5-v0`
    - `MiniGrid-DoorKey-6x6-v0`
    - `MiniGrid-DoorKey-8x8-v0`
    - `MiniGrid-DoorKey-16x16-v0`

    """

    def __init__(self, size=8, max_steps: int | None = None, context=np.array([7, 1]), **kwargs):
        if max_steps is None:
            max_steps = 50

        self.task_dim = 2
        self.context = context
        self.num_possible_goals = 0
        self.possible_goals = None

        self.domain = domain

        self.step_count = 0
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, render_mode='human', **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def get_info(self):
        return {'pos': self.agent_pos, 'dir': self.agent_dir, 'task': self.goal}

    def get_task(self):
        return self.task

    @staticmethod
    def _is_feasible(context):
        # Check that the context is not in or beyond the outer wall
        # if context[0] > 5.5 and context[0] < 6.5:
        #     if context[1] < 7.5:
        #         return False
        # if context[0] > 8 or context[0] < 1 or context[1] > 8 or context[1] < 1:
        #     return False
        # return True

        goal_x = np.rint(context[0])
        goal_y = np.rint(context[1])
        if goal_x < 1 or goal_x > 8:
            return False
        if goal_y < 1 or goal_y > 8:
            return False
        if domain.count([goal_x, goal_y]) == 0:
            return True
        else:
            return False

    def sample_initial_state(self, contexts=None):
        if contexts is None:
            return self.reset()
        else:
            return np.array([self.reset(task=contexts[i]) for i in range(len(contexts))])

    def _gen_grid(self, width, height):
        task = self.context
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        init_pos_x = 1
        init_pos_y = 1
        agent_dir = 1
        goal_x = int(np.rint(task[0]))
        goal_y = int(np.rint(task[1]))

        # Create a vertical splitting wall
        # splitIdx = self._rand_int(2, width - 2)
        splitIdx = width // 2 + 1
        self.grid.vert_wall(splitIdx, 1, self.height - 3)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        # self.place_agent(top=(1, 1), size=(1, 1), rand_dir=False, agent_dir=task[4])
        self.place_agent(top=(init_pos_x, init_pos_y), size=(1, 1), rand_dir=False, agent_dir=agent_dir)

        # Place a door in the wall
        # doorIdx = self._rand_int(1, width - 2)
        doorIdx = width - 2
        # self.put_obj(Door("yellow", is_locked=False, is_open=False), splitIdx, doorIdx)
        # self.put_obj(Lava(), 3, 7)
        # self.put_obj(Lava(), 3, 8)
        # self.put_obj(Lava(), 4, 7)
        # self.put_obj(Lava(), 4, 8)

        # self.put_obj(Lava(), 3, 4)
        # self.put_obj(Lava(), 3, 5)
        # self.put_obj(Lava(), 4, 4)
        # self.put_obj(Lava(), 4, 5)
        # self.put_obj(Lava(), 5, 4)
        # self.put_obj(Lava(), 5, 5)
        # self.put_obj(Lava(), 6, 4)
        # self.put_obj(Lava(), 6, 5)
        # self.put_obj(Lava(), 7, 4)
        # self.put_obj(Lava(), 7, 5)

        # Place the goal at last to cover other object
        self.put_obj(Goal(), goal_x, goal_y)
        self.goal = (goal_x, goal_y)  # set goal position
        self.task = (goal_x, goal_y)

        # Place a yellow key on the left side
        # self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        if self.possible_goals is None:
            self.possible_goals = []
            for i in range(1, self.width - 1):
                for j in range(1, self.height - 1):
                    self.possible_goals.append([i, j])

            for goal in self.domain:
                self.possible_goals.remove(goal)

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0
        self.door_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs

    def plot_latent_cluster(self,
                            env,
                            policy,
                            iter_idx,
                            teacher,
                            encoder=None,
                            task_decoder=None,
                            image_folder=None):
        # fig, (ax1) = plt.subplots(1, 1, figsize=(12, 12))
        from deep_sprl.teachers.acrl.util import sample_trajectory, trajectory_embedding

        cmap = plt.colormaps['spring']

        tasks = [np.rint(teacher.sample()) for i in range(64)]
        # tasks = np.unique(np.array(tasks), axis=0).tolist()

        # # ax1.set_xlim(0, 10)
        # # ax1.set_ylim(0, 10)
        # ax1.tick_params(labelsize=20)
        #
        # # tasks = torch.from_numpy(tasks)
        # for task in tasks:
        #     latent_means, latent_logvars, _, _, _, _, _ = get_test_rollout(env, args, policy, transition_encoder, task)
        #     latent_means = torch.stack(latent_means).squeeze(1).cpu().numpy()
        #
        #     # ax2.scatter(pos[:, 0], pos[:, 1], s=500, c=rand_colors, cmap=cmap)
        #     # latent_std = torch.exp(0.5 * torch.stack(latent_logvars)).mean(dim=-1).cpu().numpy()
        #     latent_std = torch.exp(0.5 * torch.stack(latent_logvars)).squeeze(1).cpu().numpy()
        #
        #     goal = f'[{task[0]}, {task[1]}]'
        #
        #     # s = 200 * latent_std
        #     # num_sample = 4096
        #     # sample = []
        #     # for i, mean in enumerate(latent_means):
        #     #     sample.append(utl.sample_gaussian(mean, latent_logvars[i], num=num_sample).cpu().numpy())
        #     # sample = np.concatenate(sample, axis=-1)
        #
        #     ax1.scatter(latent_means[:, 0], latent_means[:, 1], s=200, label=goal)
        #     for i, std in enumerate(latent_std):
        #         v_x = (latent_means[i][0], latent_means[i][0])
        #         v_y = (latent_means[i][1] - std[1] * 0.01, latent_means[i][1] + std[1] * 0.01)
        #         h_x = (latent_means[i][0] - std[0] * 0.01, latent_means[i][0] + std[0] * 0.01)
        #         h_y = (latent_means[i][1], latent_means[i][1])
        #         ax1.plot(v_x, v_y, color='k', linewidth=1, alpha=0.1)
        #         ax1.plot(h_x, h_y, color='k', linewidth=1, alpha=0.1)
        #
        #     center = latent_means.mean(axis=0)
        #     ax1.text(center[0], center[1], goal)
        #
        # ax1.legend()
        # ax1.set_title('latent cluster', size=20)

        # position = fig.add_axes([0.93, 0.11, 0.02, 0.77])
        # colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=position)
        # colorbar.ax.set_yticks([])
        # plt.tight_layout()

        # if image_folder is not None:
        #     plt.savefig('{}/{}_latent_cluster'.format(image_folder, iter_idx))
        #     plt.close()
        # else:
        #     plt.show()

        #  plot the context latent output
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 5))
        ax1.tick_params(labelsize=14)

        episode_returns = []
        scale = 0.01
        means = []
        selected_tasks = []

        # tasks = torch.from_numpy(tasks)
        for task in tasks:
            latent_means, latent_logvars, _, _, _, _, episode_return = sample_trajectory(env, policy, encoder, task)
            if episode_return < 0.3:
                continue
            mean, logvars = trajectory_embedding(latent_means, latent_logvars)
            mean = mean.cpu().detach().numpy()
            means.append(mean)
            selected_tasks.append(task)
            episode_returns.append(episode_return)
            # v_x = (mean[0], mean[0])
            # v_y = (mean[1] - std[1] * scale, mean[1] + std[1] * scale)
            # h_x = (mean[0] - std[0] * scale, mean[0] + std[0] * scale)
            # h_y = (mean[1], mean[1])
            # print(f'mean: {mean}')
            # print(f'std: {std}')
            # ax1.plot(v_x, v_y, color='k', linewidth=1, alpha=0.1)
            # ax1.plot(h_x, h_y, color='k', linewidth=1, alpha=0.1)
            # ax1.text(mean[0], mean[1], goal)
            # goal = f'[{round(task[0], 1)}, {round(task[1], 1)}]'
            # ax1.text(mean[0], mean[1], goal)

        latent_means, latent_logvars, _, _, _, _, episode_return = sample_trajectory(env, policy, encoder, [7, 1])
        mean, logvars = trajectory_embedding(latent_means, latent_logvars)
        mean = mean.cpu().detach().numpy()
        means.append(mean)
        episode_returns.append(episode_return)

        # ax1.legend()
        means = np.array(means)
        episode_returns = np.array(episode_returns)
        ax1.scatter(means[:, 0], means[:, 1], s=200, c=episode_returns, cmap=cmap)
        # ax1.set_title('latent space', size=16)
        # ax1.axes.xaxis.set_ticks([])
        # ax1.axes.yaxis.set_ticks([])
        # colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=position)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap))
        cb.ax.tick_params(labelsize=14)
        # colorbar.ax.set_yticks([])
        plt.tight_layout()

        if image_folder is not None:
            plt.savefig('{}/{}_latent_label.pdf'.format(image_folder, iter_idx))
        else:
            plt.show()

        for i, task in enumerate(selected_tasks):
            goal = f'[{round(task[0], 1)}, {round(task[1], 1)}]'
            ax1.text(means[i, 0], means[i, 1], goal)

        goal = 'TARGET [7, 1]'
        ax1.text(mean[0], mean[1], goal)

        if image_folder is not None:
            plt.savefig('{}/{}_latent.pdf'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()
        # with label
        # fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
        # ax1.tick_params(labelsize=16)
        #
        # # tasks = torch.from_numpy(tasks)
        # for task in tasks:
        #     latent_means, latent_logvars, _, _, _, _, _ = sample_trajectory(env, policy, encoder, task)
        #     mean, logvars = trajectory_embedding(latent_means, latent_logvars)
        #     mean = mean.detach().cpu().numpy()
        #     task = np.rint(task)
        #     goal = f'[{task[0]}, {task[1]}]'
        #
        #     ax1.scatter(mean[0], mean[1], s=200, label=goal)
        #     ax1.text(mean[0], mean[1], goal)
        #
        # # ax1.legend()
        # ax1.set_title('latent space', size=16)
        # # ax1.axes.xaxis.set_ticks([])
        # # ax1.axes.yaxis.set_ticks([])
        # plt.tight_layout()
        #
        # if image_folder is not None:
        #     plt.savefig('{}/{}_latent_with_label.pdf'.format(image_folder, iter_idx))
        #     plt.close()
        # else:
        #     plt.show()

        # #  last latent
        # fig, (ax1) = plt.subplots(1, 1, figsize=(12, 12))
        # ax1.tick_params(labelsize=20)
        # last_means = torch.stack(last_means).cpu().numpy()
        # ax1.scatter(last_means[:, 0], last_means[:, 1], s=200)
        # for i, task in enumerate(tasks):
        #     goal = f'[{task[0]}, {task[1]}]'
        #     ax1.text(last_means[i][0], last_means[i][1], goal)
        #
        # # ax1.legend()
        # ax1.set_title('last latent', size=20)
        #
        # # position = fig.add_axes([0.93, 0.11, 0.02, 0.77])
        # # colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=position)
        # # colorbar.ax.set_yticks([])
        # # plt.tight_layout()
        # if image_folder is not None:
        #     plt.savefig('{}/{}_last_latent'.format(image_folder, iter_idx))
        #     plt.close()
        # else:
        #     plt.show()

        # if task_decoder is not None:
        #     #  visualize task decoder
        #     fig, (ax1) = plt.subplots(1, 1, figsize=(20, 20))
        #     ax1.tick_params(labelsize=40)
        #
        #     points = []
        #     colors = []
        #     for i in range(35):
        #         for j in range(35):
        #             x = 0.1 * i - 1.75
        #             y = 0.1 * j - 0.5
        #             task = task_decoder(torch.from_numpy(np.array([x, y])).float().to(device)).detach().cpu().numpy()
        #             goal = f'[{task[0]:.2f}, {task[1]:.2f}]'
        #             points.append([x, y])
        #             if task[0] < 0:
        #                 task[0] = 0
        #             if task[0] > 8:
        #                 task[0] = 8
        #             if task[1] < 0:
        #                 task[1] = 0
        #             if task[1] > 8:
        #                 task[1] = 8
        #             colors.append((task[0] / 8.0, task[1] / 8.0, 0))
        #             # ax1.text(x, y, goal, size=5)
        #
        #     colors = np.array(colors)
        #     points = np.array(points)
        #     ax1.scatter(points[:, 0], points[:, 1], marker='s', s=850, c=colors)
        #     # ax1.set_title('task decode output', size=40)
        #     # plt.axis([-3, 0.5, -1.75, 1.75])
        #     plt.tight_layout()
        #
        #     if image_folder is not None:
        #         plt.savefig('{}/{}_task_decoder_output.pdf'.format(image_folder, iter_idx))
        #         plt.close()
        #     else:
        #         plt.show()
        #
        #     #  plot legend
        #     fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
        #     ax1.tick_params(labelsize=14)

        # points = []
        # colors = []
        # for i in range(8):
        #     for j in range(8):
        #         x = i + 1
        #         y = j + 1
        #         points.append([x, y])
        #         colors.append((x / 8.0, y / 8.0, 0))
        #         # ax1.text(x, y, goal, size=5)
        #
        # colors = np.array(colors)
        # points = np.array(points)
        # ax1.scatter(points[:, 0], points[:, 1], marker='s', s=1200, c=colors)
        # plt.axis([0.5, 8.5, 0.5, 8.5])
        # if image_folder is not None:
        #     plt.savefig('{}/{}_reference.pdf'.format(image_folder, iter_idx))
        #     plt.close()
        # else:
        #     plt.show()

    def plot_dist(self,
                  iter_idx,
                  teacher,
                  image_folder=None):
        self.context = [7, 1]
        self.reset()
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(0, 1, N)
        vals[:, 1] = np.linspace(0, 0, N)
        vals[:, 2] = np.linspace(0, 0, N)
        cmap = ListedColormap(vals)

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
        for i in range(256):
            task = np.rint(teacher.sample()).astype(int).tolist()
            if self.domain.count(task) == 0:
                offset_x = self.tile_size * task[1] + 1
                offset_y = self.tile_size * task[0] + 1
                img[offset_x:offset_x + self.tile_size - 1, offset_y:offset_y + self.tile_size - 1, 0] += 10
                #
                # key_x = self.tile_size * task[3] + 1
                # key_y = self.tile_size * task[2] + 1
                # img[key_x:key_x + self.tile_size - 1, key_y:key_y + self.tile_size - 1, 2] += 8

        plt.imshow(img)
        color_bar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 0.1), cmap=cmap))
        plt.axis('off')

        if image_folder is not None:
            plt.savefig('{}/{}_dist.pdf'.format(image_folder, iter_idx), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    def plot_evaluate_task(self,
                           env,
                           policy,
                           encoder,
                           iter_idx,
                           image_folder=None):
        from deep_sprl.teachers.acrl.util import sample_trajectory
        env.reset()

        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(0, 0, N)
        vals[:, 1] = np.linspace(0, 0.75, N)
        vals[:, 2] = np.linspace(0, 0, N)
        cmap = ListedColormap(vals)

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        episode_returns = []
        for task in self.possible_goals:
            episode_latent_means, episode_latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, \
                episode_rewards, episode_return = sample_trajectory(env, policy, encoder, task)
            offset_x = self.tile_size * task[1]
            offset_y = self.tile_size * task[0]
            img[offset_x:offset_x + self.tile_size, offset_y:offset_y + self.tile_size] *= 0
            # episode_return = 0 if episode_return < 0 else episode_return
            # img[offset_x:offset_x + self.tile_size, offset_y:offset_y + self.tile_size, 0] = int(
            #     255 - episode_return * 255)
            img[offset_x:offset_x + self.tile_size, offset_y:offset_y + self.tile_size, 1] = int(
                (episode_return + 0.55) / 1.55 * 255)

        plt.imshow(img)
        color_bar = plt.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(-0.55, 1), cmap=cmap))
        plt.axis('off')
        plt.tight_layout()

        if image_folder is not None:
            plt.savefig('{}/{}_task_evaluation.pdf'.format(image_folder, iter_idx), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
