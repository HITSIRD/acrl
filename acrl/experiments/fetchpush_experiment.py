import os
import gym
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from acrl.environments.minigrid.envs import AEnv
from acrl.experiments.abstract_experiment import AbstractExperiment, Learner
from acrl.teachers.acrl import ACRLWrapper, ACRL
from acrl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from acrl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from acrl.teachers.spl import SelfPacedTeacherV2, SelfPacedWrapper, CurrOT
from acrl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from acrl.teachers.abstract_teacher import BaseWrapper
from acrl.teachers.acl import ACL, ACLWrapper
from acrl.teachers.plr import PLR, PLRWrapper
from acrl.teachers.vds import VDS, VDSWrapper
from acrl.teachers.sampler import Subsampler, MinigridSampler
from scipy.stats import multivariate_normal
from acrl.util.device import device_type

from acrl.teachers.acrl.config.minigrid_a import config


def context_post_processing(context):
    # return np.rint(context).tolist()
    return context


class FetchPushExperiment(AbstractExperiment):
    TARGET_MEANS = np.array([0.15, 0.15, 0.42])
    TARGET_VARIANCES = np.diag([1e-4, 1e-4, 0])

    LOWER_CONTEXT_BOUNDS = np.array([-0.15, -0.15, 0.42])
    UPPER_CONTEXT_BOUNDS = np.array([0.15, 0.15, 0.42])

    def target_sampler(self, n=None, rng=None):
        target = np.array([0.15, 0.15, 0.42])
        if n is None:
            return target
        else:
            return np.repeat(target, n, axis=0)

    def target_log_likelihood(self, cs):
        p0 = multivariate_normal.logpdf(cs, self.TARGET_MEANS[0], self.TARGET_VARIANCES[0])
        p1 = multivariate_normal.logpdf(cs, self.TARGET_MEANS[1], self.TARGET_VARIANCES[1])

        pmax = np.maximum(p0, p1)
        # There is another factor of 0.5 since exactly half of the distribution is out of bounds
        return np.log(0.5 * 0.5 * (np.exp(p0 - pmax) + np.exp(p1 - pmax))) + pmax

    INITIAL_MEAN = np.array([-0.15, -0.15, 0.42])
    # INITIAL_VARIANCE = np.diag(np.square([1., 1.]))
    INITIAL_VARIANCE = np.array([0.01, 0.01, 0.])

    STD_LOWER_BOUND = np.array([0.01, 0.01, 0.01])
    KL_THRESHOLD = 8000.
    KL_EPS = 0.25
    DELTA = 0.3
    METRIC_EPS = 1.0
    EP_PER_UPDATE = 40

    STEPS_PER_ITER = 5000
    DISCOUNT_FACTOR = 0.95
    LAM = 0.99

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.2
    ACL_ETA = 0.025

    PLR_REPLAY_RATE = 0.85
    PLR_BUFFER_SIZE = 100
    PLR_BETA = 0.45
    PLR_RHO = 0.15

    VDS_NQ = 5
    VDS_LR = 1e-3
    VDS_EPOCHS = 3
    VDS_BATCHES = 20

    AG_P_RAND = {Learner.PPO: 0.1, Learner.SAC: None}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: None}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: None}

    GG_NOISE_LEVEL = {Learner.PPO: 0.1, Learner.SAC: None}
    GG_FIT_RATE = {Learner.PPO: 200, Learner.SAC: None}
    GG_P_OLD = {Learner.PPO: 0.2, Learner.SAC: None}

    ACRL_LAMBDA = config['lambda']

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)
        self.env = gym.make('FetchPush-v1')

    def create_environment(self, evaluation=False):
        env = gym.make('FetchPush-v1')

        config['action_dim'] = env.action_space.shape[0]
        config['context_dim'] = self.INITIAL_MEAN.shape[0]
        config['state_dim'] = env.observation_space['observation'].shape[0]
        config['max_episode_len'] = env.env.spec.max_episode_steps
        if len(self.parameters) > 0:
            config['lambda'] = float(self.parameters['ACRL_LAMBDA'])

        if evaluation or self.curriculum.default():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                              context_post_processing=context_post_processing)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                                context_post_processing=context_post_processing)
        elif self.curriculum.goal_gan():
            samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(1000, 2))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples,
                              )
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                                 context_post_processing=context_post_processing)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=True)
        elif self.curriculum.acl():
            bins = 50
            teacher = ACL(bins * bins, self.ACL_ETA, eps=self.ACL_EPS, norm_hist_len=2000)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                [bins, bins]))
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.vds():
            teacher = VDS(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.DISCOUNT_FACTOR, self.VDS_NQ,
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": self.STEPS_PER_ITER})
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=context_post_processing)
        elif self.curriculum.acrl():
            teacher = ACRL(self.TARGET_MEANS.copy(), self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy(),
                           self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, config, self.get_log_dir())
            env = ACRLWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                              context_visible=True, context_post_processing=context_post_processing)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                              context_post_processing=context_post_processing)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device=device_type,
                                policy_kwargs=dict(net_arch=[128, 128, 128], activation_fn=torch.nn.Tanh),
                                tensorboard_log=self.get_log_dir()),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, batch_size=128),
                    sac=dict(learning_rate=3e-4, buffer_size=10000, learning_starts=500, batch_size=64,
                             train_freq=5, target_entropy="auto"))

    def create_experiment(self):
        timesteps = 201 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(np.ones(147, )[None, :], contexts.shape[0], axis=0),
                 contexts], axis=-1)
            env.teacher.initialize_teacher(env, interface, state_provider)

        if isinstance(env, ACRLWrapper):
            env.teacher.set_policy(model)

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def state_provider(self, contexts):
        return np.concatenate(
            [self.env.sample_initial_state(contexts=contexts), contexts], axis=-1)

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(200, 2))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS, self.EP_PER_UPDATE,
                          wb_max_reuse=1)

    def get_env_name(self):
        return "fetch_push"

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
        for i in range(0, 50):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_statistics()[0]
