import gym.spaces
import numpy as np
from torch.ao.nn.quantized.functional import threshold

from acrl.teachers.util import Buffer
from acrl.teachers.abstract_teacher import BaseWrapper


class ACRLWrapper(BaseWrapper):

    def __init__(self, env, teacher, discount_factor, context_visible, success_thres, reward_from_info=False,
                 use_undiscounted_reward=False, context_post_processing=None, eval_mode=False):
        self.use_undiscounted_reward = use_undiscounted_reward
        BaseWrapper.__init__(self, env, teacher, discount_factor, context_visible, reward_from_info=reward_from_info,
                             context_post_processing=context_post_processing)

        self.eval_mode = eval_mode
        self.threshold = success_thres
        context = self.teacher.sample()
        if context_visible:
            self.observation_space = gym.spaces.Dict({'observation': self.env.observation_space,
                                                      'desired_goal': gym.spaces.Box(-np.inf * np.ones_like(context),
                                                                                     np.inf * np.ones_like(context)),
                                                      'achieved_goal': gym.spaces.Box(-np.inf * np.ones_like(context),
                                                                                      np.inf * np.ones_like(context))})
        self.episode_count = 0
        self.n_step = 0

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
        obs_dict = {'observation': obs,
                    'desired_goal': self.cur_context,
                    'achieved_goal': self.env.env.env.wrapped_env.get_xy()}

        obs_wo_context = obs

        if self.context_visible:
            obs = np.concatenate((obs, self.processed_context))

        self.last_obs = obs.copy()
        self.last_obs_wo_context = obs_wo_context
        self.cur_initial_state = obs.copy()
        if not wo_context:
            return obs_dict
        else:
            return obs

    def step(self, action, wo_context=False):
        step = self.env.step(action)
        original_obs = step[0]

        # if isinstance(step[0], dict):
        #     obs = step[0]['observation']

        current_step = step[0], action, step[1], self.last_obs_wo_context, self.cur_context
        # done = step[2]

        self.last_obs_wo_context = original_obs.copy()

        # step = np.concatenate((step[0], self.env.unwrapped.context)), step[1], step[2], step[3]
        if wo_context:
            obs = original_obs
            step = obs, np.concatenate((original_obs, self.cur_context)), step[1], step[2], step[3]
        else:
            # step = np.concatenate((obs, self.cur_context)), step[1], step[2], step[3]

            obs = {'observation': original_obs,
                   'desired_goal': self.cur_context,
                   'achieved_goal': self.env.env.env.wrapped_env.get_xy()}
            step = obs, step[1], step[2], step[3]
        self.last_obs = obs.copy()

        if not self.eval_mode:
            # insert replay buffer
            self.teacher.ds_encoder.rollout_storage.insert(step=current_step)
            self.teacher.update_visit(original_obs)

        assert wo_context is False
        self.update(step)
        self.n_step += 1

        return step

    def compute_reward(self, achieved_goal, desired_goal, info):
        threshold = 1.0
        distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        rewards = (distances < threshold).astype(float) - 1
        return rewards

    def update(self, step):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if step[2]:
            if not self.eval_mode:
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
        self.teacher.update_distribution(self.episode_count, self.n_step, self)
        if undiscounted_reward > self.threshold:
            self.teacher.teacher.achieved_task.append(cur_context)


    def get_encountered_contexts(self, reset=None, batch_size=None):
        if batch_size is None:
            return self.context_trace_buffer.read_buffer(reset=reset)
        else:
            buffer = self.context_trace_buffer.read_buffer(reset=reset)
            ret, context = np.array(buffer[0]), np.array(buffer[2])
            batch_size = min(len(ret), batch_size)
            index = np.random.choice(len(ret), batch_size)
            return context[index], ret[index]
