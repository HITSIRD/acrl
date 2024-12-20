import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from acrl.teachers.acrl.util import FeatureExtractor
from acrl.util.device import device


class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 input_prev_state=False,
                 input_action=False,
                 pred_type='deterministic'
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if isinstance(state_dim, tuple) and len(state_dim) > 0:
            state_dim = np.array(state_dim).prod()
        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim
        if self.input_prev_state:
            curr_input_dim += state_embed_dim
        if self.input_action:
            curr_input_dim += action_embed_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state=None, actions=None):
        # we do the action-normalisation (the the env bounds) here
        h = latent_state.clone()
        if self.input_prev_state:
            hs = self.state_encoder(state)
            h = torch.cat((h, hs), dim=-1)
        if self.input_action:
            # actions = squash_action(actions, self.args)
            ha = self.action_encoder(actions)
            h = torch.cat((h, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.tanh(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 multi_head=False,
                 input_prev_state=False,
                 input_action=False,
                 input_next_state=False,
                 pred_type='deterministic'
                 ):
        super(RewardDecoder, self).__init__()

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action
        self.input_next_state = input_next_state

        # if self.multi_head:
        #     # one output head per state to predict rewards
        #     curr_input_dim = latent_dim
        #     self.fc_layers = nn.ModuleList([])
        #     for i in range(len(layers)):
        #         self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
        #         curr_input_dim = layers[i]
        #     self.fc_out = nn.Linear(curr_input_dim, num_states)
        # else:
        # get state as input and predict reward prob
        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        if self.input_action:
            self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)
        else:
            self.action_encoder = None
        curr_input_dim = latent_dim
        if self.input_prev_state:
            curr_input_dim += state_embed_dim
        if self.input_action:
            curr_input_dim += action_embed_dim
        if self.input_next_state:
            curr_input_dim += state_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2)
        else:
            self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, prev_state=None, actions=None, next_state=None):
        # we do the action-normalisation (the the env bounds) here
        # if actions is not None:
        #     actions = utl.squash_action(actions, self.args)

        if self.multi_head:
            h = latent_state.clone()
        else:
            h = latent_state.clone()
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                h = torch.cat((h, ha), dim=-1)
            if self.input_next_state:
                hns = self.state_encoder(next_state)
                h = torch.cat((h, hns), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.tanh(self.fc_layers[i](h))

        return self.fc_out(h)


class StateDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 state_dim):
        super(StateDecoder, self).__init__()

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = state_dim
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent):
        h = latent
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 task_dim):
        super(TaskDecoder, self).__init__()

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):
        h = latent_state
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
            # dropout = torch.nn.Dropout(p=0.05)
            # h = dropout(h)

        return self.fc_out(h)
