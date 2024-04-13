from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=100,
    entry_point='acrl.environments.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='Maze-v1',
    max_episode_steps=200,
    entry_point='acrl.environments.maze:MazeEnv'
)

register(
    id="MiniGrid-A-v1",
    entry_point="acrl.environments.minigrid.envs:AEnv",
    kwargs={"size": 10},
)

register(
    id="MiniGrid-B-v1",
    entry_point="acrl.environments.minigrid.envs:BEnv",
    kwargs={"size": 10},
)

register(
    id="MiniGrid-C-v1",
    entry_point="acrl.environments.minigrid.envs:CEnv",
    kwargs={"size": 12},
)

register(
    id="MiniGrid-D-v1",
    entry_point="acrl.environments.minigrid.envs:DEnv",
    kwargs={"size": 10},
)

register(
    id="MiniGrid-E-v1",
    entry_point="acrl.environments.minigrid.envs:EEnv",
    kwargs={"size": 10},
)

register(
    id="MiniGrid-F-v1",
    entry_point="acrl.environments.minigrid.envs:FEnv",
    kwargs={"size": 12},
)

register(
    id="MiniGrid-G-v1",
    entry_point="acrl.environments.minigrid.envs:GEnv",
    kwargs={"size": 16},
)

robots = ['Point', 'Ant', 'Swimmer']
# task_types = ['Maze', 'Maze1', 'Maze2', 'Push', 'Fall', 'Block', 'BlockMaze']
task_types = ['Maze', 'Maze1', 'Maze2']
all_name = [x + y for x in robots for y in task_types]
random_start = False

# if args.image:
#     top_down = True
# else:
#     top_down = False

top_down = True

for name_t in all_name:
    # episode length
    if name_t == "AntMaze" or name_t == "AntMaze1" or name_t == "AntMaze2":
        max_timestep = 1000
        top_down = False
        goal_obs = True
    if name_t == "PointMaze2":
        max_timestep = 250
    else:
        max_timestep = 100
    for Test in ['', 'Test', 'Test1', 'Test2']:

        if Test in ['Test', 'Test1', 'Test2']:
            fix_goal = True
        else:
            if name_t == "AntBlock":
                fix_goal = True
            else:
                fix_goal = False
        goal_args = [[-5, -5], [5, 5]]

        register(
            id=name_t + Test + '-v0',
            entry_point='acrl.environments.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 8, 'random_start': random_start,
                    "fix_goal": fix_goal, "top_down_view": top_down, 'test': Test},
            max_episode_steps=max_timestep,
        )

        # v1 is the one we use in the main paper
        register(
            id=name_t + Test + '-v1',
            entry_point='acrl.environments.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 4, 'random_start': random_start,
                    "fix_goal": fix_goal, "top_down_view": top_down, 'test': Test},
            max_episode_steps=max_timestep,
        )

        register(
            id=name_t + Test + '-v2',
            entry_point='acrl.environments.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 2, 'random_start': random_start,
                    "fix_goal": fix_goal, "top_down_view": top_down, 'test': Test},
            max_episode_steps=max_timestep,
        )
