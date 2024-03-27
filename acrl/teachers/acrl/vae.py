import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from acrl.teachers.acrl.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from acrl.teachers.acrl.util import trajectory_embedding
from acrl.teachers.acrl.encoder import TransitionEncoder, TrajectoryEncoder
from acrl.util.device import device


class VAE:
    """
    VAE
    - has an transition_encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (transition_encoder+decoder)
    """

    def __init__(self, config):
        self.config = config
        # self.get_iter_idx = get_iter_idx
        self.task_decoder_loss_threshold = self.config['task_decoder_loss_threshold']
        self.max_task_decoder_update = self.config['max_task_decoder_update']

        # initialise the transition_encoder
        self.transition_encoder, self.trajectory_encoder = self.initialise_encoder()

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder, self.reward_decoder, self.task_decoder = self.initialise_decoder()

        # initialise rollout storage for the VAE update
        self.rollout_storage = RolloutStorageVAE(max_traj_len=self.config['max_episode_len'],
                                                 max_traj_size=self.config['task_buffer_size'],
                                                 state_dim=self.config['state_dim'],
                                                 action_dim=self.config['action_dim'],
                                                 task_dim=self.config['context_dim'])

        decoder_params = []
        if not self.config['disable_decoder']:
            if self.config['decode_reward']:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.config['decode_state']:
                decoder_params.extend(self.state_decoder.parameters())
            if self.config['decode_task']:
                decoder_params.extend(self.task_decoder.parameters())
        if self.transition_encoder is not None:
            self.optimiser_vae = torch.optim.Adam([*self.transition_encoder.parameters(), *decoder_params],
                                                  lr=self.config['lr_vae'])
        if self.task_decoder is not None:
            self.optimiser_task = torch.optim.Adam([*self.task_decoder.parameters()], lr=self.config['lr_task_decoder'])

    def initialise_encoder(self):
        """ Initialises and returns an RNN transition_encoder """
        transition_encoder = TransitionEncoder(
            encoder_layers=self.config['encoder_layers'],
            latent_dim=self.config['latent_dim'],
            action_dim=self.config['action_dim'],
            action_embed_dim=self.config['action_embedding_size'],
            state_dim=self.config['state_dim'],
            state_embed_dim=self.config['state_embedding_size'],
            reward_size=1,
            reward_embed_dim=self.config['reward_embedding_size'],
            task_size=self.config['context_dim'],
            task_embed_dim=self.config['task_embedding_size']
        ).to(device)

        trajectory_encoder = TrajectoryEncoder(latent_dim=self.config['latent_dim']).to(device)
        return transition_encoder, trajectory_encoder

    def initialise_decoder(self):
        if self.config['disable_decoder']:
            return None, None

        latent_dim = self.config['latent_dim']
        # if we don't sample embeddings for the decoder, we feed in mean & variance
        if self.config['disable_stochasticity_in_latent']:
            latent_dim *= 2

        # initialise state decoder for VAE
        if self.config['decode_state']:
            state_decoder = StateTransitionDecoder(
                layers=self.config['state_decoder_layers'],
                latent_dim=latent_dim,
                action_dim=self.config['action_dim'],
                action_embed_dim=self.config['action_embedding_size'],
                state_dim=self.config['state_dim'],
                state_embed_dim=self.config['state_embedding_size'],
                input_prev_state=self.config['state_input_prev_state'],
                input_action=self.config['state_input_action'],
                pred_type=self.config['state_pred_type'],
            ).to(device)
        else:
            state_decoder = None

        # initialise reward decoder for VAE
        if self.config['decode_reward']:
            reward_decoder = RewardDecoder(
                layers=self.config['reward_decoder_layers'],
                latent_dim=latent_dim,
                state_dim=self.config['state_dim'],
                state_embed_dim=self.config['state_embedding_size'],
                action_dim=self.config['action_dim'],
                action_embed_dim=self.config['action_embedding_size'],
                multi_head=self.config['multihead_for_reward'],
                input_prev_state=self.config['reward_input_prev_state'],
                input_action=self.config['reward_input_action'],
                input_next_state=self.config['reward_input_next_state'],
                pred_type=self.config['rew_pred_type']
            ).to(device)
        else:
            reward_decoder = None

        # initialise task decoder for VAE
        task_decoder = TaskDecoder(
            layers=self.config['task_decoder_layers'],
            latent_dim=self.config['latent_dim'],
            task_dim=self.config['context_dim'] if self.config['task_pred_type'] == 'param' else self.config[
                'num_possible_goals']
        ).to(device)

        return state_decoder, reward_decoder, task_decoder

    def compute_state_reconstruction_loss(self, latent, prev_obs, next_obs, action, return_predictions=False):
        """ Compute state reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        state_pred = self.state_decoder(latent, prev_obs, action)

        if self.config['state_pred_type'] == 'deterministic':
            loss = torch.nn.MSELoss(reduction='none')
            # loss_state = torch.norm(state_pred - next_obs, dim=1)
            loss_state = loss(state_pred, next_obs)
        elif self.config['state_pred_type'] == 'gaussian':  # TODO: untested!
            state_pred_mean = state_pred[:, :state_pred.shape[1] // 2]
            state_pred_std = torch.exp(0.5 * state_pred[:, state_pred.shape[1] // 2:])
            m = torch.distributions.normal.Normal(state_pred_mean, state_pred_std)
            loss_state = -m.log_prob(next_obs).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_state, state_pred
        else:
            return loss_state

    def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, action, reward, return_predictions=False):
        """ Compute reward reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        rew_pred = self.reward_decoder(latent, prev_obs, action.float(), next_obs)
        if self.config['rew_pred_type'] == 'bernoulli':
            rew_pred = torch.sigmoid(rew_pred)
            rew_target = (reward > 0.01).float()
            loss = torch.nn.BCELoss(reduction='none')
            loss_rew = loss(rew_pred, rew_target)
        elif self.config['rew_pred_type'] == 'deterministic':
            loss = torch.nn.MSELoss(reduction='none')
            loss_rew = loss(rew_pred, reward)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew

    def compute_task_reconstruction_loss(self, latent_means, latent_logvars, trajectory_lens, tasks, num_sample=256,
                                         return_predictions=False):
        """ Compute task reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """
        task_means = torch.zeros((latent_means.shape[0], latent_means.shape[-1])).to(device)
        task_logvars = torch.zeros((latent_means.shape[0], latent_means.shape[-1])).to(device)
        for i, latent in enumerate(latent_means):
            means = latent_means[i]
            means = means[:trajectory_lens[i]]
            logvars = latent_logvars[i]
            logvars = logvars[:trajectory_lens[i]]

            mean, logvars = trajectory_embedding(means, logvars)
            task_means[i] = mean
            # task_means[i] = vars * (means * vars_rec).sum(dim=0)
            # task_logvars[i] = vars.log()

        # if not self.args.disable_stochasticity_in_latent:
        #     task_latent = self.transition_encoder._sample_gaussian(task_means, task_logvars)
        # else:
        # task_latent = torch.cat((task_means, task_logvars), dim=-1)

        if self.config['task_pred_type'] == 'param':
            # print(torch.cat((task_means, task_logvars), dim=-1))
            # task_latent = self.transition_encoder._sample_gaussian(task_means, task_logvars)
            task_latent = task_means
            # tasks = tasks.repeat(num_sample, 1, 1)
            task_pred = self.task_decoder(task_latent)
            # print(torch.cat((tasks, task_pred), dim=-1))
            loss = torch.nn.MSELoss(reduction='none')
            loss_task = loss(task_pred, tasks)
        # cross entropy
        elif self.config['task_pred_type'] == 'categ':
            task_latent = self.transition_encoder._sample_gaussian(task_means, task_logvars)
            # tasks = tasks.repeat(num_sample, 1, 1)
            tasks = ((tasks[:, :, 0] - 1) * 8 + tasks[:, :, 1] - 1).long().view(-1)
            task_pred = self.task_decoder(task_latent)
            task_pred = task_pred.view(-1, task_pred.shape[-1])

            loss = torch.nn.CrossEntropyLoss(reduction='none')
            loss_task = loss(task_pred, tasks)

            #  one hot MSE
            # task_latent = self.transition_encoder._sample_gaussian(task_means, task_logvars, num_sample)
            # tasks = tasks.repeat(num_sample, 1, 1)
            # tasks = ((tasks[:, :, 0] - 1) * 8 + tasks[:, :, 1] - 1).unsqueeze(2)
            # num_categ = 64
            # index = tasks.long()
            # tasks_one_hot = torch.zeros((*tasks.shape[:2], num_categ)).to(device)
            # tasks_one_hot = tasks_one_hot.scatter_(-1, index, 1)
            # task_pred = self.task_decoder(task_latent)
            #
            # loss = torch.nn.MSELoss(reduction='none')
            # loss_task = loss(task_pred, tasks_one_hot)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_task, task_pred
        else:
            return loss_task

    def compute_kl_loss(self, latent_mean, latent_logvar):
        # -- KL divergence
        if self.config['kl_to_gauss_prior']:
            kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()))
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior
            all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
            all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

        return kl_divergences

    def compute_loss(self, latent_mean, latent_logvar, prev_obs, next_obs, actions, rewards, tasks, trajectory_lens):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """

        # cut down the batch to the longest trajectory length
        # this way we can preserve the structure
        # but we will waste some computation on zero-padded trajectories that are shorter than max_traj_len

        # take one sample for each ELBO term
        if not self.config['disable_stochasticity_in_latent']:
            latent_samples = self.transition_encoder._sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        tasks = tasks[:, 0]
        embedding = latent_samples

        if self.config['decode_reward']:
            rew_reconstruction_loss = self.compute_rew_reconstruction_loss(embedding, prev_obs, next_obs, actions,
                                                                           rewards)
            # average across tasks
            rew_reconstruction_loss = rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = 0

        if self.config['decode_state']:
            state_reconstruction_loss = self.compute_state_reconstruction_loss(embedding, prev_obs, next_obs, actions)
            # average across tasks
            # state_reconstruction_loss = state_reconstruction_loss.sum(dim=-1)
            state_reconstruction_loss = state_reconstruction_loss.mean()
        else:
            state_reconstruction_loss = 0

        if self.config['decode_task']:
            task_reconstruction_loss = self.compute_task_reconstruction_loss(latent_mean.detach(),
                                                                             latent_logvar.detach(),
                                                                             trajectory_lens,
                                                                             tasks)
            task_reconstruction_loss = task_reconstruction_loss.mean()
        else:
            task_reconstruction_loss = 0

        if not self.config['disable_kl_term']:
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar)
            # avg/sum the elbos
            kl_loss = kl_loss.mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=True):
        """ Returns the VAE loss """

        if not self.rollout_storage.ready_for_update():
            return 0

        if self.config['disable_decoder'] and self.config['disable_kl_term']:
            return 0

        # get a mini-batch
        vae_prev_obs, vae_actions, vae_rewards, vae_next_obs, vae_tasks, trajectory_lens = self.rollout_storage.get_batch(
            batchsize=self.config['vae_batch_num_trajectories'])

        _, latent_mean, latent_logvar = self.transition_encoder(prev_states=vae_prev_obs,
                                                                actions=vae_actions,
                                                                rewards=vae_rewards,
                                                                next_states=vae_next_obs,
                                                                tasks=vae_tasks)

        losses = self.compute_loss(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions, vae_rewards,
                                   vae_tasks, trajectory_lens)
        rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss = losses

        # not include task reconstruction loss
        loss = self.config['rew_loss_coeff'] * rew_reconstruction_loss + \
               self.config['state_loss_coeff'] * state_reconstruction_loss + \
               self.config['task_loss_coeff'] * task_reconstruction_loss + \
               self.config['kl_weight'] * kl_loss

        # make sure we can compute gradients
        if not self.config['disable_kl_term']:
            assert kl_loss.requires_grad
        if self.config['decode_reward']:
            assert rew_reconstruction_loss.requires_grad
        if self.config['decode_state']:
            assert state_reconstruction_loss.requires_grad
        if self.config['decode_task']:
            assert task_reconstruction_loss.requires_grad
        # overall loss
        elbo_loss = loss.mean()
        # print(f'elbo loss {elbo_loss}')
        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.config['encoder_max_grad_norm'] is not None:
                nn.utils.clip_grad_norm_(self.transition_encoder.parameters(), self.config['encoder_max_grad_norm'])
            if self.config['decoder_max_grad_norm'] is not None:
                if self.config['decode_reward']:
                    nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), self.config['decoder_max_grad_norm'])
                if self.config['decode_state']:
                    nn.utils.clip_grad_norm_(self.state_decoder.parameters(), self.config['decoder_max_grad_norm'])
                if self.config['decode_task']:
                    nn.utils.clip_grad_norm_(self.task_decoder.parameters(), self.config['decoder_max_grad_norm'])
            self.optimiser_vae.step()

        # self.log(elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss,
        #          pretrain_index)

    def update_task_decoder(self):
        threshold = self.task_decoder_loss_threshold + 1
        step = 0
        while threshold > self.task_decoder_loss_threshold and step < self.max_task_decoder_update:
            vae_prev_obs, vae_actions, vae_rewards, vae_next_obs, vae_tasks, trajectory_lens = self.rollout_storage.get_batch(
                batchsize=self.config['task_batch_num_trajectories'])
            _, latent_mean, latent_logvar = self.transition_encoder(prev_states=vae_prev_obs,
                                                                    actions=vae_actions,
                                                                    rewards=vae_rewards,
                                                                    next_states=vae_next_obs,
                                                                    tasks=vae_tasks)

            task_reconstruction_loss = self.compute_task_reconstruction_loss(latent_mean.detach(),
                                                                             latent_logvar.detach(),
                                                                             trajectory_lens,
                                                                             vae_tasks)
            task_reconstruction_loss = task_reconstruction_loss.mean()
            assert task_reconstruction_loss.requires_grad
            # print(f'task reconstruction loss {task_reconstruction_loss}')

            #  update task decoder
            self.optimiser_task.zero_grad()
            task_reconstruction_loss.backward()
            if self.config['decoder_max_grad_norm'] is not None:
                if self.config['decode_task']:
                    nn.utils.clip_grad_norm_(self.task_decoder.parameters(), self.config['decoder_max_grad_norm'])
            self.optimiser_task.step()
            # self.log_task(task_reconstruction_loss)

            step += 1
            threshold = task_reconstruction_loss

    def log(self, elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss,
            pretrain_index=None):
        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * self.args.num_vae_updates_per_pretrain + pretrain_index

        if curr_iter_idx % self.args.log_interval == 0:
            if self.args.decode_reward:
                self.logger.add('vae_losses/reward_reconstr_err', rew_reconstruction_loss, curr_iter_idx)
            if self.args.decode_state:
                self.logger.add('vae_losses/state_reconstr_err', state_reconstruction_loss, curr_iter_idx)
            if self.args.decode_task:
                self.logger.add('vae_losses/task_reconstr_err', task_reconstruction_loss, curr_iter_idx)

            if not self.args.disable_kl_term:
                self.logger.add('vae_losses/kl', kl_loss, curr_iter_idx)
            self.logger.add('vae_losses/sum', elbo_loss, curr_iter_idx)

    def log_task(self, task_reconstruction_loss):
        curr_iter_idx = self.get_iter_idx()
        if curr_iter_idx % self.args.log_interval == 0:
            self.logger.add('vae_losses/task_reconstr_err', task_reconstruction_loss, curr_iter_idx)


class RolloutStorageVAE(object):
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
            self.prev_state = torch.zeros((self.max_buffer_size, self.max_traj_len, self.obs_dim))
            self.action = torch.zeros((self.max_buffer_size, self.max_traj_len, action_dim))
            self.reward = torch.zeros((self.max_buffer_size, self.max_traj_len, 1))
            self.next_state = torch.zeros((self.max_buffer_size, self.max_traj_len, self.obs_dim))
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
            self.prev_state[self.insert_idx, self.step_idx] = torch.from_numpy(prev_state)
            self.action[self.insert_idx, self.step_idx] = torch.tensor(action)
            self.reward[self.insert_idx, self.step_idx] = torch.tensor([reward])
            self.next_state[self.insert_idx, self.step_idx] = torch.from_numpy(next_state)
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
            self.prev_state[self.insert_idx, :trajectory_len] = torch.stack(prev_state).squeeze(1)
            self.action[self.insert_idx, :trajectory_len] = torch.stack(action).squeeze(1)
            self.reward[self.insert_idx, :trajectory_len] = torch.stack(reward).unsqueeze(1)
            self.next_state[self.insert_idx, :trajectory_len] = torch.stack(next_state).squeeze(1)
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

    def get_batch(self, batchsize=5, replace=False, return_delta=0):
        batchsize = min(self.buffer_len, batchsize)
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)

        trajectory_lens = self.trajectory_lens[rollout_indices]
        max_lens = np.max(trajectory_lens)
        assert np.min(trajectory_lens) > 0

        prev_obs = self.prev_state[rollout_indices]
        actions = self.action[rollout_indices]
        rewards = self.reward[rollout_indices]
        next_obs = self.next_state[rollout_indices]
        tasks = self.task[rollout_indices]

        #  select 0:max_lens to save memory
        prev_obs = prev_obs[:, :max_lens]
        actions = actions[:, :max_lens]
        rewards = rewards[:, :max_lens]
        next_obs = next_obs[:, :max_lens]
        tasks = tasks

        return prev_obs.to(device), actions.to(device), rewards.to(device), next_obs.to(device), tasks.to(
            device), trajectory_lens
