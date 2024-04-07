"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""


import gym

from acrl.environments.mujoco_maze.ant import AntEnv
from acrl.environments.mujoco_maze.maze_task import TaskRegistry
from acrl.environments.mujoco_maze.point import PointEnv
from acrl.environments.mujoco_maze.reacher import ReacherEnv
from acrl.environments.mujoco_maze.swimmer import SwimmerEnv

for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        point_scale = task_cls.MAZE_SIZE_SCALING.point
        if point_scale is not None:
            # Point
            gym.envs.register(
                id=f"Point{maze_id}-v{i}",
                entry_point="acrl.environments.mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=PointEnv,
                    maze_task=task_cls,
                    maze_size_scaling=point_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

        ant_scale = task_cls.MAZE_SIZE_SCALING.ant
        if ant_scale is not None:
            # Ant
            gym.envs.register(
                id=f"Ant{maze_id}-v{i}",
                entry_point="acrl.environments.mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=AntEnv,
                    maze_task=task_cls,
                    maze_size_scaling=ant_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )
            # print(f'Ant{maze_id}-v{i}')

        swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
        if swimmer_scale is not None:
            # Reacher
            gym.envs.register(
                id=f"Reacher{maze_id}-v{i}",
                entry_point="acrl.environments.mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=ReacherEnv,
                    maze_task=task_cls,
                    maze_size_scaling=task_cls.MAZE_SIZE_SCALING.swimmer,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )
            # Swimmer
            gym.envs.register(
                id=f"Swimmer{maze_id}-v{i}",
                entry_point="acrl.environments.mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=SwimmerEnv,
                    maze_task=task_cls,
                    maze_size_scaling=task_cls.MAZE_SIZE_SCALING.swimmer,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )


__version__ = "0.2.0"
