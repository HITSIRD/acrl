import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from acrl.util.device import device

from acrl.teachers.acrl.util import FeatureExtractor


class TransitionEncoder(nn.Module):
    def __init__(self,
                 # network size
                 encoder_layers=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=147,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_dim=5,
                 task_size=2,
                 task_embed_dim=8
                 ):
        super(TransitionEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        # self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        # self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)
        # self.reward_encoder = FeatureExtractor(reward_size, reward_embed_dim, F.relu)
        # self.task_encoder = FeatureExtractor(task_size, task_embed_dim, F.relu)

        # curr_input_dim = 1 + state_embed_dim * 2 + 2
        # curr_input_dim = state_embed_dim * 2 + task_size

        curr_input_dim = state_dim * 2 + 1
        # if self.config['task_embedding_size'] > 0:
        #     curr_input_dim += task_embed_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(encoder_layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, encoder_layers[i]))
            curr_input_dim = encoder_layers[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            std = torch.exp(0.5 * logvar).repeat(num, 1, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1, 1)
            return eps.mul(std).add_(mu)

    def forward(self, prev_states, actions, rewards, next_states=None, tasks=None, sample=True):

        # we do the action-normalisation (the env bounds) here
        # actions = utl.squash_action(actions, self.args)

        # shape should be: batch_size x trajectory_len x hidden_size
        # prev_states = prev_states.reshape((-1, prev_states.shape[1], self.state_dim))
        # if next_states is not None:
        #     next_states = next_states.reshape((-1, next_states.shape[1], self.state_dim))

        # hps = self.state_encoder(prev_states.float())
        hps = prev_states.float()

        # ha = self.action_encoder(actions.float())
        # ha = actions.float()
        # hr = self.reward_encoder(rewards.float())
        hr = rewards.float()
        h = torch.cat((hps, hr), dim=-1)

        if next_states is not None:
            # hns = self.state_encoder(next_states.float())
            hns = next_states.float()
            h = torch.cat((h, hns), dim=-1)

        # if tasks is not None:
        #     ht = self.task_encoder(tasks)
        #     h = torch.cat((h, ht), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        # outputs
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        return latent_sample, latent_mean, latent_logvar

# class TrajectoryEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(TrajectoryEncoder, self).__init__()
#
#         self.latent_dim = latent_dim
#         self.attention = nn.MultiheadAttention(embed_dim=latent_dim * 2, num_heads=1, batch_first=True)
#
#     def forward(self, inputs):
#         # batch_size, trajectory_length, dim = input.shape
#         q, k, v = torch.clone(inputs), torch.clone(inputs), torch.clone(inputs)
#         attn_output, attn_output_weights = self.attention(q, k, v)
#         outputs = torch.mean(attn_output, dim=1)
#
#         latent_mean = outputs[:, :self.latent_dim]
#         latent_logvar = outputs[:, self.latent_dim:]
#         return latent_mean, latent_logvar

class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(TrajectoryEncoder, self).__init__()

        self.input_dim = latent_dim
        self.score_func = nn.Linear(latent_dim, 1)

    def forward(self, input, sample=True):
        scores = self.score_func(input)
        scores = F.softmax(scores, dim=-1)
        outputs = scores.expand_as(input).mul(input).sum(1)

        latent_mean = outputs[:, :self.input_dim]
        latent_logvar = outputs[:, self.input_dim:]
        if sample:
            latent_sample = _sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_sample = None

        return latent_sample, latent_mean, latent_logvar

def _sample_gaussian(mu, logvar, num=None):
    if num is None:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        std = torch.exp(0.5 * logvar).repeat(num, 1, 1)
        eps = torch.randn_like(std)
        mu = mu.repeat(num, 1, 1)
        return eps.mul(std).add_(mu)
