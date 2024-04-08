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

    def reset(self, context=None, wo_context=False):
        if context is None:
            self.cur_context = self.teacher.sample()
        else:
            self.cur_context = context
        if self.context_post_processing is None:
            self.processed_context = self.cur_context
        else:
            self.processed_context = self.context_post_processing(self.cur_context)

        obs = self.env.reset(context=self.processed_context.copy())

        if isinstance(obs, dict):
            # print(self.processed_context)
            # print('desired')
            self.processed_context = obs['desired_goal']
            # print('achieved')
            # print(obs['achieved_goal'])
            # print(self.processed_context)
            obs = obs['observation']

        obs_wo_context = obs

        if self.context_visible:
            obs = np.concatenate((obs, self.processed_context))

        self.last_obs = obs.copy()
        self.last_obs_wo_context = obs_wo_context
        self.cur_initial_state = obs.copy()
        if not wo_context:
            return obs
        else:
            return obs, obs_wo_context

    def step(self, action, update=True, wo_context=False, insert=True):
        step = self.env.step(action)
        obs = step[0]
        # print('achieved')
        # print(obs['achieved_goal'])
        # print(obs['desired_goal'])
        if isinstance(step[0], dict):
            obs = step[0]['observation']

        current_step = obs, action, step[1], self.last_obs_wo_context, self.cur_context
        done = step[2]

        self.last_obs_wo_context = obs.copy()

        # step = np.concatenate((step[0], self.env.unwrapped.context)), step[1], step[2], step[3]
        if wo_context:
            step = obs, np.concatenate((obs, self.cur_context)), step[1], step[2], step[3]
        else:
            step = np.concatenate((obs, self.cur_context)), step[1], step[2], step[3]
        self.last_obs = obs.copy()

        # insert VAE buffer
        if insert:
            self.teacher.vae.rollout_storage.insert(step=current_step, done=done)
        if update:
            assert wo_context is False
            self.update(step)
        return step

    def update(self, step):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if step[2]:
            self.done_callback(step, self.cur_initial_state.copy(), self.cur_context, self.discounted_reward,
                               self.undiscounted_reward)

            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, self.step_length))
            self.context_trace_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward,
                                                     self.processed_context.copy()))
            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.processed_context = None
            self.cur_initial_state = None

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        # We currently rely on the learner being set on the environment after its creation
        self.episode_count += 1
        self.teacher.update_distribution(self.episode_count, self)
