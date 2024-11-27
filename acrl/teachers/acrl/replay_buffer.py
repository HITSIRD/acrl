from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np


class HERReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_space, action_space, goal_dim, device, her_ratio=0.8, strategy="future"):
        super(HERReplayBuffer, self).__init__(buffer_size, obs_space, action_space, device, n_envs=1)
        self.goal_dim = goal_dim
        self.her_ratio = her_ratio
        self.strategy = strategy
        self.episode_storage = []

    def add_transition(self, obs, action, reward, next_obs, done, goal, infos):
        """
        Store the transition and the goal in the replay buffer.
        """
        self.episode_storage.append((obs, action, reward, next_obs, done, goal, infos))
        super().add(obs, next_obs, action, reward, done, infos)

        if done:
            self._store_her_transitions()

    def _store_her_transitions(self):
        """
        Generate HER transitions after an episode is finished.
        """
        episode = self.episode_storage
        self.episode_storage = []  # Clear after processing

        episode_length = len(episode)
        for t, (obs, action, reward, next_obs, done, goal, infos) in enumerate(episode):
            if np.random.rand() < self.her_ratio:
                # Apply goal relabeling
                if self.strategy == "future":
                    future_time = np.random.randint(t, episode_length)
                    new_goal = episode[future_time][3]  # Use future next_obs as goal
                else:
                    raise NotImplementedError("Only 'future' strategy is supported for now.")

                # Compute new reward
                new_reward = self.compute_reward(next_obs, new_goal)
                # Add HER transition to the buffer
                super().add(obs, next_obs, action, new_reward, done, infos)

    def compute_reward(self, obs, goal):
        """
        Custom reward computation for HER, can be based on distance or other metrics.
        """
        return 0