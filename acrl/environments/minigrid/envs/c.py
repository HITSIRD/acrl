from __future__ import annotations

import numpy as np

# import utils.helpers
from acrl.environments.minigrid.core.grid import Grid
from acrl.environments.minigrid.core.mission import MissionSpace
from acrl.environments.minigrid.core.world_object import Door, Goal, Wall, Key, Lava
from acrl.environments.minigrid.minigrid_env import MiniGridEnv

domain = [[1, 1], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [8, 5], [9, 5],
          [10, 5]]
key_domain = [[1, 6], [2, 6], [3, 6], [4, 6], [1, 7], [2, 7], [3, 7], [4, 7], [1, 8], [2, 8], [3, 8], [4, 8],
              [1, 9], [2, 9], [3, 9], [4, 9], [1, 10], [2, 10], [3, 10], [4, 10]]


class CEnv(MiniGridEnv):
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

    def __init__(self, size=12, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 200

        self.task_dim = 4
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
        goal_x = int(np.rint(context[0]))
        goal_y = int(np.rint(context[1]))
        key_x = int(np.rint(context[2]))
        key_y = int(np.rint(context[3]))

        if goal_x < 1 or goal_x > 10:
            return False
        if goal_y < 1 or goal_y > 10:
            return False
        if key_x < 1 or key_x > 10:
            return False
        if key_y < 1 or key_y > 10:
            return False

        if domain.count([goal_x, goal_y]) == 0 and domain.count([key_x, key_y]) == 0 and key_domain.count(
                [key_x, key_y]) == 0:
            if [goal_x, goal_y] != [key_x, key_y]:
                return True
            else:
                return False
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
        door_x = 5
        door_y = 10

        goal_x = int(np.rint(task[0]))
        goal_y = int(np.rint(task[1]))
        key_x = int(np.rint(task[2]))
        key_y = int(np.rint(task[3]))

        # Create a vertical splitting wall
        self.grid.set(1, 5, Wall())
        self.grid.set(2, 5, Wall())
        self.grid.set(3, 5, Wall())
        self.grid.set(4, 5, Wall())
        self.grid.set(5, 5, Wall())
        self.grid.set(5, 6, Wall())
        self.grid.set(5, 7, Wall())
        self.grid.set(5, 8, Wall())
        self.grid.set(5, 9, Wall())

        # self.grid.set(7, 5, Lava())
        self.grid.set(8, 5, Lava())
        self.grid.set(9, 5, Lava())
        self.grid.set(10, 5, Lava())
        #
        # self.grid.set(5, 1, Lava())
        # self.grid.set(6, 1, Lava())
        # self.grid.set(7, 1, Lava())
        # self.grid.set(5, 2, Lava())
        # self.grid.set(6, 2, Lava())
        # self.grid.set(7, 2, Lava())

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        # self.place_agent(top=(1, 1), size=(1, 1), rand_dir=False, agent_dir=task[4])
        self.place_agent(top=(init_pos_x, init_pos_y), size=(1, 1), rand_dir=False, agent_dir=agent_dir)

        # Place a door in the wall
        self.put_obj(Door("yellow", is_locked=True, is_open=False), door_x, door_y)

        # Place the goal at last to cover other object
        self.put_obj(Goal(), goal_x, goal_y)
        self.goal = (goal_x, goal_y)  # set goal position
        self.task = (goal_x, goal_y, key_x, key_y)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("yellow"), top=(key_x, key_y), size=(1, 1))

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

        # if self.render_mode == "human":
        # self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs
