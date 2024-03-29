import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, NoReturn
import torch

from acrl.environments.minigrid.envs import AEnv, BEnv, CEnv
from acrl.teachers.abstract_teacher import AbstractTeacher
from acrl.teachers.acrl.util import sample_trajectory, trajectory_embedding, get_latent_map
from acrl.teachers.acrl.vae import VAE
from acrl.teachers.sampler import MinigridSampler
from acrl.teachers.dummy_teachers import UniformSampler
from acrl.util.device import device


class ACRL(AbstractTeacher):

    def __init__(self, target, initial_mean, initial_std, context_lb, context_ub, config, log_dir, post_sampler=None):

        # Create an array if we use the same number of bins per dimension
        self.config = config
        self.target = target
        self.context_dim = initial_mean.shape[0]

        self.log_dir = log_dir

        self.context_lb = context_lb
        self.context_ub = context_ub
        # self.uniform_sampler = MinigridSampler(self.context_lb, self.context_ub)
        self.uniform_sampler = UniformSampler(self.context_lb, self.context_ub)
        self.teacher = LatentSpacePrediction(initial_mean, initial_std, self.target, self.uniform_sampler,
                                             self.config)
        self.vae = VAE(self.config)
        self.policy = None  # sample trajectory to train VAE

        self.post_sampler = post_sampler

    def initialize_context_buffer(self):
        self.teacher.initialize(self.uniform_sampler)

    def set_policy(self, policy):
        self.policy = policy

    def update_distribution(self, episode_count, env):
        if episode_count % self.config['update_freq'] == 0:

            ret = env.get_encountered_contexts(reset=False)[0]
            size = len(ret)
            ret = np.array(ret)[-min(size, self.config['task_buffer_size']):]
            print(f'mean of return: {np.mean(ret)}({np.std(ret)})')

            #  evaluate current policy
            if hasattr(env.env, 'plot_evaluate_task'):
                print('plot_evaluate_task...')
                env.env.plot_evaluate_task(env, self.policy, self.vae.transition_encoder, episode_count, self.log_dir)

            if np.mean(ret) > self.config['update_delta']:
                #  plot
                if isinstance(env.env, AEnv):
                    print('plot_latent_cluster...')
                    env.env.plot_latent_cluster(env, self.policy, episode_count, self.teacher,
                                                self.vae.transition_encoder,
                                                self.vae.task_decoder, self.log_dir)
                    print('plot_dist...')
                    env.env.plot_dist(episode_count, self.teacher, self.log_dir)

                for _ in range(self.config['num_vae_update']):
                    self.vae.compute_vae_loss(update=True)

                if not self.config['decode_task']:
                    self.vae.update_task_decoder()

                self.teacher.update(env, self.policy, self.vae)
                print('update distribution...')

    def sample(self):
        task = self._sample()
        if self.post_sampler is not None:
            while not self.post_sampler(task):
                task = self._sample()
        return task

    def _sample(self):
        return self.teacher.sample()

    def save(self, path):
        # self.model.save(os.path.join(path, "teacher_model.pkl"))
        # self.teacher.save(path)
        # self.success_buffer.save(path)
        # self.sampler.save(path)
        pass

    def load(self, path):
        # self.model.load(os.path.join(path, "teacher_model.pkl"))
        # self.teacher.load(path)
        # self.success_buffer.load(path)
        # self.sampler.load(path)
        pass


class LatentSpacePrediction:
    def __init__(self, init_mean, init_std, target, uniform_sampler, config):
        self.config = config
        self.init_mean = torch.from_numpy(np.array(init_mean))
        self.init_std = torch.from_numpy(np.array(init_std))
        self.target = target

        self.return_delta = config['return_delta']
        self.step_size = config['step_size']
        self.curriculum_index = 0
        self.current_tasks = []
        self.current_task_latent = []
        self.random_tasks = []
        self.ls_tasks = []
        self.buffer_size = config['task_buffer_size']
        self.update_lambda = config['lambda']
        self.ebu_ratio = config['ebu_ratio']
        self.task_noise_std = torch.from_numpy(np.array(config['noise_std'])).to(device)

        self.num_target_sample = 20
        self.target_episode_latent_means = None
        self.target_episode_latent_logvars = None
        self.target_return = -np.inf

        self.uniform_sampler = uniform_sampler
        self.initialize(self.uniform_sampler)

    def initialize(self, sampler):
        # init_dist = GaussianTorchDistribution(self.init_mean, self.init_std, use_cuda=False, dtype=torch.float64)
        self.current_tasks = sampler(size=self.buffer_size)
        index = int(max(self.update_lambda, self.ebu_ratio) * self.buffer_size)
        if index > 0:
            self.current_tasks[:index] = torch.stack([self._sample_gaussian(self.init_mean, self.init_std) for i in
                                                      range(index)])
        else:
            self.current_tasks = [self._sample_gaussian(self.init_mean, self.init_std) for i in
                                  range(self.buffer_size)]

    def sample(self, buffer=None):
        if buffer is None:
            buffer = self.current_tasks

        context = self.current_tasks[np.random.randint(len(buffer))] if np.random.random() > 0.1 else self.target
        # context = self.current_tasks[np.random.randint(len(buffer))]

        if isinstance(context, torch.Tensor):
            context = context.numpy()
        context = np.clip(context, self.uniform_sampler.lower_bound, self.uniform_sampler.upper_bound)

        return context

    def update(self, env, policy, vae):
        print('updating task dist...')
        encoder = vae.transition_encoder
        task_decoder = vae.task_decoder
        sampled_tasks = []

        task_latent, task_returns = self.latent_map(vae.rollout_storage, encoder)

        # sample target task embedding
        for i in range(self.num_target_sample):
            target_episode_latent_means, target_episode_latent_logvars, _, _, _, _, target_return = sample_trajectory(
                env,
                policy,
                encoder,
                self.target)
            if target_return > self.target_return:
                print(f'target return update: {self.target_return}->{target_return}')
                self.target_return = target_return
                self.target_episode_latent_means = target_episode_latent_means
                self.target_episode_latent_logvars = target_episode_latent_logvars

        self.target_return = -np.inf  # reset
        target_means, target_logvars = trajectory_embedding(self.target_episode_latent_means,
                                                            self.target_episode_latent_logvars)
        # target_latent = self._sample_gaussian(target_means, target_logvars)
        target_latent = target_means

        task_latent = torch.stack(task_latent)
        task_returns = torch.from_numpy(np.array(task_returns)).cpu().numpy().flatten()
        index = np.argwhere(task_returns > self.return_delta).flatten()
        if index.shape[0] == 0:
            #  nothing to update
            return

        #  latent selection algorithm
        task_latent = task_latent[index]
        latent_distance = (task_latent - target_latent).norm(dim=-1).squeeze(-1)
        _, latent_index = torch.sort(latent_distance)

        #  task decoder prediction method
        task_latent = task_latent[latent_index]  # sort
        task_latent = task_latent + self.step_size * (
                target_latent - task_latent)  # updated latent by linear interpolation
        if self.config['task_pred_type'] == 'param':
            updated_tasks = task_decoder(task_latent).squeeze(1)
            if self.config['add_noise']:
                noise = self.task_noise_std * torch.randn(size=updated_tasks.shape).to(device)
                updated_tasks += noise
        elif self.config['task_pred_type'] == 'categ':
            updated_tasks = torch.argmax(task_decoder(task_latent), dim=-1)
            updated_tasks = torch.cat((updated_tasks // 8 + 1, updated_tasks % 8 + 1), dim=-1)
        else:
            raise NotImplementedError

        if updated_tasks.shape[0] > self.buffer_size:
            update_index = int(self.buffer_size * self.update_lambda)
            self.current_tasks = updated_tasks[:update_index]
            self.current_tasks[update_index:] = self.uniform_sampler(size=self.buffer_size - update_index)
        else:
            update_index = int(min(updated_tasks.shape[0], self.buffer_size * self.update_lambda))
            # if update_index > 0:
            self.current_tasks[:update_index] = updated_tasks[:update_index].cpu().detach().numpy()
            self.current_tasks[update_index:] = self.uniform_sampler(size=self.buffer_size - update_index)
            # self.current_tasks[-16:] = torch.from_numpy(np.array(self.target)).float().to(device)

            self.EBU_update(sampled_tasks, index, update_index, latent_index)

        self.curriculum_index += 1
        # print(self.current_tasks.view(-1, self.current_tasks.shape[0] // 16,
        #                               self.current_tasks.shape[-1]).cpu().detach().numpy())

    def EBU_update(self, sampled_tasks, index, update_index, latent_index):
        # if self.config['enable_latent_selection_sample']:
        if self.ebu_ratio > 0:
            sampled_tasks = torch.stack(sampled_tasks)
            sampled_tasks = sampled_tasks[index]
            # print(f'sampled_tasks shape: {sampled_tasks.shape[0]}')
            # max_x = index.shape[0]
            # print(max_x)
            ls_buffer_size = min(int((self.buffer_size - update_index) * self.ebu_ratio), index.shape[0])
            # if ls_buffer_size > 0:
            ls_task = []
            for i in range(index.shape[0]):
                # random_x = np.random.randint(max_x)
                # s_i = np.searchsorted(self.sample_dist_index, random_x, 'right')
                s_i = min(int(np.random.exponential(1.0)), index.shape[0])
                # print(sampled_tasks[latent_index[s_i]])
                ls_task.append(
                    sampled_tasks[latent_index[s_i]] + self.task_noise_std * torch.randn(
                        self.config['context_dim']))

            ls_task = torch.stack(ls_task)
            # print(ls_task)
            self.current_tasks[-ls_buffer_size:] = ls_task[:ls_buffer_size]
            self.ls_tasks = ls_task

    def latent_map(self, buffer, encoder):
        latent_means, latent_logvars = get_latent_map(buffer, encoder)
        return latent_means, buffer.episode_return

    def _sample_gaussian(self, mu, std, num=None):
        if num is None:
            # std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            # raise NotImplementedError
            # std = torch.exp(0.5 * logvar).repeat(num, 1)
            std = std.repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)
