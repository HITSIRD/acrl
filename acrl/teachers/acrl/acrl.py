import os

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, NoReturn
import torch
from sympy.solvers.diophantine.diophantine import reconstruct

from acrl.teachers.abstract_teacher import AbstractTeacher
from acrl.teachers.acrl.util import sample_trajectory, trajectory_embedding, get_latent_map
from acrl.teachers.acrl.evaluator import Evaluator
from acrl.teachers.acrl.vq_vae import VQVAE
from acrl.teachers.dummy_teachers import UniformSampler
from acrl.util.device import device
from sklearn.neighbors import KernelDensity

from torch.utils.tensorboard import SummaryWriter


# np.seterr(all='raise')

class ACRL(AbstractTeacher):

    def __init__(self, target, initial_state, initial_mean, initial_std, context_lb, context_ub, config, log_dir,
                 post_sampler=None):

        # Create an array if we use the same number of bins per dimension
        self.config = config
        self.target = target
        self.initial_state = initial_state
        self.context_dim = initial_mean.shape[0]

        self.log_dir = log_dir

        self.context_lb = context_lb
        self.context_ub = context_ub
        # self.uniform_sampler = MinigridSampler(self.context_lb, self.context_ub)
        self.uniform_sampler = UniformSampler(self.context_lb, self.context_ub)
        # self.vae = VAE(self.config)

        self.ds_encoder = VQVAE(self.config)
        self.warmup_step = config['warmup_step']
        self.uniform = True
        self.policy = None  # sample trajectory to train VAE

        self.teacher = LatentSpacePrediction(initial_state, initial_mean, initial_std, self.target, initial_mean,
                                             self.uniform_sampler, self.config)

        self.post_sampler = post_sampler
        self.gradient_step = 0
        self.writer = SummaryWriter(self.log_dir + '/log/VQ')
        os.makedirs(f'{self.log_dir}/figures', exist_ok=True)

    def set_policy(self, policy):
        self.policy = policy

    def update_distribution(self, n_episode, n_step, env):
        if n_step > self.warmup_step:
            self.uniform = False

        #  plot
        if n_episode % 200 == 0:
            self.plot_proto_task(n_episode)
            self.plot_dist(n_episode, samples=50)

        if n_episode % self.config['update_freq'] == 0:
            ret = env.get_encountered_contexts(reset=False)[0]
            size = len(ret)
            ret = np.array(ret)[-min(size, 10):]
            # print(f'mean of return: {np.mean(ret)}({np.std(ret)})')

            for _ in range(self.config['num_encoder_update']):
                loss = self.ds_encoder.update()
                for k, v in loss.items():
                    self.writer.add_scalar('train/' + k, v, n_step)
                    # self.writer.add_scalar(k, v, self.gradient_step)
                self.gradient_step += 1

            self.teacher.update_proto_task(self.policy, self.ds_encoder)
            self.writer.add_scalar('target_alpha', self.teacher.target_alpha, n_step)

    def sample(self):
        task = self._sample()
        if self.post_sampler is not None:
            while not self.post_sampler(task):
                task = self._sample()
        return task

    def _sample(self):
        return self.teacher.sample(self.uniform)

    def update_visit(self, state):
        assert self.teacher.visit_count is not None
        self.teacher.update_visit(state, self.ds_encoder)

    def plot_proto_task(self, n_episode):
        if len(self.config['noise_std']) != 2:
            return

        print('plot proto task...')
        plt.clf()

        proto_task = self.teacher.proto_task.cpu().numpy()
        dis = self.teacher.temporal_dis

        vertices = np.array([[self.context_lb[0], self.context_lb[1]],
                             [self.context_ub[0], self.context_lb[1]],
                             [self.context_ub[0], self.context_ub[1]],
                             [self.context_lb[0], self.context_ub[1]],
                             [self.context_lb[0], self.context_lb[1]]])
        plt.plot(vertices[:, 0], vertices[:, 1], 'b-')

        plt.scatter(proto_task[:, 0], proto_task[:, 1])
        for i in range(len(dis)):
            plt.text(proto_task[i, 0], proto_task[i, 1], f'{dis[i]:.2f}')

        plt.savefig(f'{self.log_dir}/figures/proto_task_{n_episode}.pdf')

        plt.clf()

        proto_task = self.teacher.proto_task.cpu().numpy()
        visit_count = self.teacher.visit_count

        vertices = np.array([[self.context_lb[0], self.context_lb[1]],
                             [self.context_ub[0], self.context_lb[1]],
                             [self.context_ub[0], self.context_ub[1]],
                             [self.context_lb[0], self.context_ub[1]],
                             [self.context_lb[0], self.context_lb[1]]])
        plt.plot(vertices[:, 0], vertices[:, 1], 'b-')

        plt.scatter(proto_task[:, 0], proto_task[:, 1])
        for i in range(len(dis)):
            plt.text(proto_task[i, 0], proto_task[i, 1], f'{visit_count[i]}')

        plt.savefig(f'{self.log_dir}/figures/proto_task_visit_{n_episode}.pdf')

    def plot_dist(self, n_episode, samples=100):
        if len(self.config['noise_std']) != 2:
            return

        print('plot dist...')
        plt.clf()

        context = []
        for i in range(samples):
            context.append(self.teacher.sample())
        context = np.array(context)

        vertices = np.array([[self.context_lb[0], self.context_lb[1]],
                             [self.context_ub[0], self.context_lb[1]],
                             [self.context_ub[0], self.context_ub[1]],
                             [self.context_lb[0], self.context_ub[1]],
                             [self.context_lb[0], self.context_lb[1]]])
        plt.plot(vertices[:, 0], vertices[:, 1], 'b-')

        plt.scatter(context[:, 0], context[:, 1])

        plt.savefig(f'{self.log_dir}/figures/dist_{n_episode}.pdf')

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
    def __init__(self, initial_state, init_mean, init_std, target, start, uniform_sampler, config):
        self.config = config
        self.initial_state = torch.from_numpy(initial_state).float().to(device)
        self.init_mean = torch.from_numpy(np.array(init_mean))
        self.init_std = torch.from_numpy(np.array(init_std))
        self.target = target
        self.start = start

        self.target_alpha = 0
        self.task_noise_std = torch.from_numpy(np.array(config['noise_std'])).to(device)
        self.temporal_dis = None
        self.top_k = 5
        self.proto_task = None
        self.visit_count = np.zeros([config['codebook_k']])
        self.uncertainty = None
        self.sort_index = None

        self.achieved_task = []

        self.uniform_sampler = uniform_sampler

    # def initialize(self, sampler):
    #     self.current_tasks = [self._sample_gaussian(self.init_mean, self.init_std) for i in
    #                           range(self.buffer_size)]

    def sample(self, uniform=False):
        if uniform or self.temporal_dis is None:
            return self.uniform_sampler()

        if np.random.random() > self.target_alpha:
            top_index = self.sort_index[:self.top_k]
            # probs = 1. / (torch.from_numpy(self.visit_count[top_index]) + 0.01)
            # probs = torch.softmax(1. / torch.from_numpy(self.visit_count[top_index]), dim=-1)
            probs = torch.softmax(self.temporal_dis[top_index] / np.sqrt(self.visit_count[top_index] + 1.), dim=-1)
            # probs = torch.softmax(self.temporal_dis, dim=-1)

            distribution = torch.distributions.Categorical(probs=probs)
            index = distribution.sample()

            # index = torch.argmax(probs)

            index = self.sort_index[index]
            context = self.proto_task[index]
            context = context + self._sample_noise()
            context = context.cpu().numpy()
        else:
            context = self.target

        if isinstance(context, torch.Tensor):
            context = context.cpu().numpy()
        context = np.clip(context, self.uniform_sampler.lower_bound, self.uniform_sampler.upper_bound)

        return context

    def reset(self):
        self.temporal_dis = None
        self.proto_task = None
        self.visit_count = np.ones([self.config['codebook_k']])
        self.uncertainty = None

    def update_proto_task(self, policy, ds_encoder):
        print('updating task distance...')
        k = self.config['codebook_k']

        self.visit_count = np.zeros([k])
        encoder = ds_encoder.encoder
        decoder = ds_encoder.decoder
        codebook = ds_encoder.codebook
        task_latent = codebook.embedding.weight

        with torch.no_grad():
            reconstruct_state = decoder(task_latent)
            proto_task = self.state2task(reconstruct_state)  # TODO state2task
            # proto_task = reconstruct_state  # TODO state2task
            z_start = encoder(self.initial_state)
            z_q_st, z_q, indices = codebook.straight_through(z_start)
            rec_start = decoder(z_q_st)

            state = {'observation': rec_start.unsqueeze(0).repeat(k, 1),
                     'achieved_goal': self.state2task(rec_start).unsqueeze(0).repeat(k, 1),
                     'desired_goal': proto_task}
            action, _ = policy.actor.action_log_prob(state)
            q = torch.cat(policy.critic_target(state, action), dim=1)
            q, _ = torch.min(q, dim=1, keepdim=True)
            temporal_dis = torch.log(1. + (1. - policy.gamma) * q) / np.log(policy.gamma)

            _, sort_index = torch.sort(temporal_dis.flatten(), descending=True)

        # print(f'dis = {dis}')
        self.temporal_dis = temporal_dis.cpu().flatten()
        self.sort_index = sort_index.cpu().flatten()
        self.uncertainty = np.zeros((task_latent.shape[0],))
        self.proto_task = proto_task

        self.update_alpha()

        # print(self.temporal_dis)
        # self.reset()

    def update_visit(self, state, ds_encoder):
        with torch.no_grad():
            z_e = ds_encoder.encoder(torch.from_numpy(state).to(device))
            _, _, index = ds_encoder.codebook.straight_through(z_e)

        self.visit_count[index] += 1

    def update_alpha(self):
        kde = KernelDensity(kernel='gaussian', bandwidth='silverman')

        # kde.fit(self.proto_task.cpu().numpy())
        kde.fit(np.array(self.achieved_task))

        log_density = kde.score_samples([self.target])
        self.target_alpha = min(1.0, np.exp(log_density)[0] + 0.01)

    def _sample_noise(self):
        return self.task_noise_std * torch.randn(len(self.task_noise_std)).to(device)

    def state2task(self, state):
        if len(state.shape) == 1:
            return state[:2]
        else:
            return state[:, :2]

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
