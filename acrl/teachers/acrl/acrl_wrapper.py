import numpy as np
from acrl.teachers.util import Buffer
from acrl.teachers.abstract_teacher import BaseWrapper


class ACRLWrapper(BaseWrapper):

    def __init__(self, env, teacher, discount_factor, context_visible, reward_from_info=False,
                 use_undiscounted_reward=False, episodes_per_update=50, context_post_processing=None):
        self.use_undiscounted_reward = use_undiscounted_reward
        BaseWrapper.__init__(self, env, teacher, discount_factor, context_visible, reward_from_info=reward_from_info,
                             context_post_processing=context_post_processing)

        self.context_buffer = Buffer(3, episodes_per_update + 1, True)
        self.episodes_per_update = episodes_per_update
        self.episode_count = 0

    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)

    def reset(self, context=None, with_context=False):
        if context is None:
            self.cur_context = self.teacher.sample()
        else:
            self.cur_context = context
        if self.context_post_processing is None:
            self.processed_context = self.cur_context
        else:
            self.processed_context = self.context_post_processing(self.cur_context)
        self.env.context = self.processed_context
        # self.processed_context = self.env.unwrapped.context
        obs = self.env.reset()
        obs_wo_context = obs

        if self.context_visible:
            obs = np.concatenate((obs, self.env.context))

        self.last_obs = obs.copy()
        self.cur_initial_state = obs.copy()
        if not with_context:
            return obs
        else:
            return obs, obs_wo_context

    def step(self, action, update=True, with_context=False):
        step = self.env.step(action)
            # step = np.concatenate((step[0], self.env.unwrapped.context)), step[1], step[2], step[3]
        if with_context:
            step = step[0], np.concatenate((step[0], self.env.context)), step[1], step[2], step[3]
        else:
            step = np.concatenate((step[0], self.env.context)), step[1], step[2], step[3]
        self.last_obs = step[0].copy()
        if update:
            self.update(step)
        return step

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        # We currently rely on the learner being set on the environment after its creation
        self.episode_count += 1
        self.teacher.update_distribution(self.episode_count, self)
