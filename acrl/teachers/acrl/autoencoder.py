import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from acrl.teachers.acrl.decoder import TaskDecoder
from acrl.teachers.acrl.encoder import Encoder
from acrl.teachers.acrl.util import FeatureExtractor
from acrl.util.device import device


class AutoEncoder(nn.Module):
    def __init__(self,  # network size
                 config):
        super(AutoEncoder, self).__init__()

        self.config = config
        self.encoder = self.initialise_encoder()
        self.decoder = self.initialise_decoder()

        self.rollout_storage = RolloutStorage(max_traj_len=self.config['max_episode_len'],
                                              max_traj_size=self.config['task_buffer_size'],
                                              state_dim=self.config['state_dim'],
                                              action_dim=self.config['action_dim'],
                                              task_dim=self.config['context_dim'])

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.001)

    def initialise_encoder(self):
        """ Initialises and returns an RNN transition_encoder """
        encoder = Encoder(
            encoder_layers=self.config['encoder_layers'],
            latent_dim=self.config['latent_dim'],
            action_dim=self.config['action_dim'],
            action_embed_dim=self.config['action_embedding_size'],
            state_dim=self.config['state_dim'],
            state_embed_dim=self.config['state_embedding_size'],
            reward_size=1,
            reward_embed_dim=self.config['reward_embedding_size'],
            task_dim=self.config['context_dim'],
            task_embed_dim=self.config['task_embedding_size']
        ).to(device)

        return encoder

    def initialise_decoder(self):
        # initialise task decoder for VAE
        task_decoder = TaskDecoder(
            layers=self.config['task_decoder_layers'],
            latent_dim=self.config['latent_dim'],
            task_dim=self.config['context_dim'] if self.config['task_pred_type'] == 'param' else self.config[
                'num_possible_goals']
        ).to(device)

        return task_decoder

    def forward(self, x):
        z = self.encoder(x)
        return z

    def update(self):
        x_pair, h_pair = self.rollout_storage.get_batch()

        # batch_size x 2 x x_dim
        z_1 = self.encoder(x_pair[:, 0:1, :])
        z_2 = self.encoder(x_pair[:, 1:2, :])

        loss_func = nn.MSELoss()
        target = torch.abs(h_pair[:, 0] - h_pair[:, 1])
        dis_loss = loss_func(torch.norm(z_1 - z_2, dim=-1), target)
        # loss = recon_loss + dis_loss
        # loss = dis_loss

        self.encoder_optimizer.zero_grad()
        dis_loss.backward()
        self.encoder_optimizer.step()

        rec_x_1 = self.decoder(z_1.detach())
        rec_x_2 = self.decoder(z_2.detach())
        recon_loss = loss_func(x_pair[:, 0:1, :], rec_x_1) + loss_func(x_pair[:, 1:2, :], rec_x_2)

        self.decoder_optimizer.zero_grad()
        recon_loss.backward()
        self.decoder_optimizer.step()

        loss_dir = {}
        loss_dir['recon_loss'] = recon_loss.mean()
        loss_dir['dis_loss'] = dis_loss.mean()
        # print(dis_loss.mean())
        # loss_dir['loss'] = loss.mean()
        return loss_dir

    def eval(self, x):
        pass


class RolloutStorage(object):
    def __init__(self, max_traj_len, max_traj_size, state_dim, action_dim, task_dim):
        """
        Store everything that is needed for the VAE update
        """

        self.action_dim = action_dim
        self.task_dim = task_dim
        if isinstance(state_dim, tuple) and len(state_dim) > 0:
            self.obs_dim = np.array(state_dim).prod()
        else:
            self.obs_dim = state_dim

        # self.obs_dim += self.task_dim  # cat context

        self.max_traj_len = max_traj_len
        self.max_buffer_size = max_traj_size  # maximum buffer len (number of trajectories)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.step_idx = 0  # at which index we're currently inserting the new step
        self.buffer_len = 0  # how much of the buffer has been fill

        if self.max_buffer_size > 0:
            # self.prev_state = torch.zeros((self.max_buffer_size, self.max_traj_len, self.obs_dim))
            # self.action = torch.zeros((self.max_buffer_size, self.max_traj_len, action_dim))
            self.reward = torch.zeros((self.max_buffer_size, self.max_traj_len, 1))
            # self.next_state = torch.zeros((self.max_buffer_size, self.max_traj_len, self.obs_dim))
            self.task = torch.zeros((self.max_buffer_size, task_dim))
            self.trajectory_lens = np.zeros(self.max_buffer_size, dtype=int)
            self.episode_return = np.zeros(self.max_buffer_size, dtype=float)

    def clear(self):
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.step_idx = 0
        self.buffer_len = 0  # how much of the buffer has been fill

    def insert(self, step=None, trajectory=None, done=False):
        if step is not None:
            prev_state, action, reward, next_state, task = step
            # self.prev_state[self.insert_idx, self.step_idx] = torch.from_numpy(prev_state)
            # self.action[self.insert_idx, self.step_idx] = torch.tensor(action)
            self.reward[self.insert_idx, self.step_idx] = torch.tensor([reward])
            # self.next_state[self.insert_idx, self.step_idx] = torch.from_numpy(next_state)
            self.trajectory_lens[self.insert_idx] = self.step_idx + 1

            self.step_idx += 1
            if done:
                self.task[self.insert_idx] = torch.from_numpy(np.array(task))
                self.episode_return[self.insert_idx] = self.reward[self.insert_idx, :self.step_idx].sum().numpy()
                self.insert_idx = (self.insert_idx + 1) % self.max_buffer_size
                self.step_idx = 0
            self.buffer_len = max(self.buffer_len, self.insert_idx)

            # if not done and self.step_idx == self.max_traj_len:
            #     self.step_idx = 0
        elif trajectory is not None:
            prev_state, action, reward, next_state, task = trajectory
            trajectory_len = len(action)
            # self.prev_state[self.insert_idx, :trajectory_len] = torch.stack(prev_state).squeeze(1)
            # self.action[self.insert_idx, :trajectory_len] = torch.stack(action).squeeze(1)
            self.reward[self.insert_idx, :trajectory_len] = torch.stack(reward).unsqueeze(1)
            # self.next_state[self.insert_idx, :trajectory_len] = torch.stack(next_state).squeeze(1)
            self.task[self.insert_idx] = torch.from_numpy(np.array(task))
            self.trajectory_lens[self.insert_idx] = trajectory_len
            self.episode_return[self.insert_idx] = self.reward[self.insert_idx].sum().numpy()

            self.insert_idx = self.insert_idx + 1 % self.max_buffer_size
            self.buffer_len = max(self.buffer_len, self.insert_idx)
        else:
            raise NotImplementedError

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False, return_delta=-np.inf):
        batchsize = min(self.buffer_len, batchsize)
        index = np.argwhere(self.episode_return[:self.buffer_len] > return_delta).flatten()
        # assert index.size > 0
        size = min(len(index), batchsize)
        if index.size > 0:
            rollout_indices = index[np.random.choice(len(index), size, replace=replace)]
        else:
            rollout_indices = np.random.choice(self.buffer_len, batchsize, replace=replace)

        # trajectory_lens = self.trajectory_lens[rollout_indices]
        # max_lens = np.max(trajectory_lens)
        # assert np.min(trajectory_lens) > 0

        # # prev_obs = self.prev_state[rollout_indices]
        # # actions = self.action[rollout_indices]
        # rewards = self.reward[rollout_indices]
        # # next_obs = self.next_state[rollout_indices]
        # tasks = self.task[rollout_indices]
        #
        # #  select 0:max_lens to save memory
        # # prev_obs = prev_obs[:, :max_lens]
        # # actions = actions[:, :max_lens]
        # rewards = rewards[:, :max_lens]
        # # next_obs = next_obs[:, :max_lens]
        # tasks = tasks
        #
        # return rewards.to(device), tasks.to(device), trajectory_lens

        task_pairs = []
        horizon_pairs = []
        for i in range(size):
            index_1, index_2 = np.random.choice(index, size=2, replace=False)
            task_pairs.append([self.task[index_1], self.task[index_2]])
            horizon_pairs.append([self.trajectory_lens[index_1], self.trajectory_lens[index_2]])

        task_pairs = torch.from_numpy(np.array(task_pairs)).float().to(device)
        horizon_pairs = torch.from_numpy(np.array(horizon_pairs)).float().to(device)

        return task_pairs, horizon_pairs
