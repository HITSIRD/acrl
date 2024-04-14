import numpy as np
from acrl.teachers.util import Buffer
from acrl.teachers.abstract_teacher import BaseWrapper


class SelfPacedWrapper(BaseWrapper):

    def __init__(self, env, sp_teacher, discount_factor, context_visible, reward_from_info=False,
                 use_undiscounted_reward=False, episodes_per_update=50, log_dir=None):
        self.use_undiscounted_reward = use_undiscounted_reward
        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible, reward_from_info=reward_from_info)

        self.context_buffer = Buffer(3, episodes_per_update + 1, True)
        self.episodes_per_update = episodes_per_update
        self.episode = 0
        self.env = env
        self.log_dir = log_dir

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        ret = undiscounted_reward if self.use_undiscounted_reward else discounted_reward
        self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))

        self.episode += 1
        if hasattr(self.teacher, "on_rollout_end"):
            self.teacher.on_rollout_end(cur_context, ret)

        if len(self.context_buffer) >= self.episodes_per_update:
            __, contexts, returns = self.context_buffer.read_buffer()
            self.teacher.update_distribution(np.array(contexts), np.array(returns), self.env, self.episode, log_dir=self.log_dir)

    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)
