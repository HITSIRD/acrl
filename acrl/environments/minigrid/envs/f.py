from __future__ import annotations

import random

import torch
from matplotlib.colors import Normalize, ListedColormap

from acrl.environments.minigrid.core.grid import Grid
from acrl.environments.minigrid.core.mission import MissionSpace
from acrl.environments.minigrid.core.world_object import Door, Goal, Wall, Lava, Key
from acrl.environments.minigrid.minigrid_env import MiniGridEnv
from acrl.environments.minigrid.utils.util import get_area

import matplotlib.pyplot as plt
import numpy as np

# wall = [[3, 7], [3, 8], [3, 9], [3, 10], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [6, 7], [6, 8], [6, 9], [7, 7], [8, 7],
#         [5, 4], [6, 4], [7, 4], [8, 4], [9, 4]]
wall = [[3, 7], [3, 8], [3, 9], [3, 10], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [6, 7], [6, 8], [6, 9], [7, 7], [8, 7], [5, 4], [6, 4], [7, 4], [8, 4]]
# lava = [[1, 3], [2, 3], [3, 3], [3, 4], [3, 5], [3, 6]]
# lava = [[1, 3], [2, 3], [7, 7], [8, 7], [9, 7], [10, 7]]
lava = [[1, 3], [2, 3], [9, 7], [10, 7]]
# door = [6, 4]
obstacle = wall + lava
# key_ban_domain = get_area([1, 3], height=5, weight=5)
# key = [8, 8]
start = [1, 10]


class FEnv(MiniGridEnv):
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

    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 100

        self.task_dim = 2
        self.step_count = 0
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def get_info(self):
        return {'pos': self.agent_pos, 'dir': self.agent_dir, 'task': self.goal}

    def get_task(self):
        return self.task

    @staticmethod
    def is_feasible(context):
        # Check that the context is not in or beyond the outer wall
        goal = [np.rint(context[0]), np.rint(context[1])]
        if goal[0] < 1 or goal[0] > 10 or goal[1] < 1 or goal[1] > 10:
            return False

        if goal == start:
            return False

        if obstacle.count(goal) == 0:
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

        init_pos_x = start[0]
        init_pos_y = start[1]
        agent_dir = 1

        goal_x = int(np.rint(task[0]))
        goal_y = int(np.rint(task[1]))

        for pos in wall:
            self.grid.set(pos[0], pos[1], Wall())

        for pos in lava:
            self.grid.set(pos[0], pos[1], Lava())

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(top=(init_pos_x, init_pos_y), size=(1, 1), rand_dir=False, agent_dir=agent_dir)

        # Place the goal at last to cover other object
        self.put_obj(Goal(), goal_x, goal_y)
        self.goal = (goal_x, goal_y)  # set goal position
        self.task = (goal_x, goal_y)

        self.mission = "use the key to open the door and then get to the goal"

    def reset(self, *, seed=None, options=None, context=None):
        super().reset(seed=seed, context=context)
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
