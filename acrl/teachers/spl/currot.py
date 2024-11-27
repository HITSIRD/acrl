import os
import time
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, NoReturn
from scipy.optimize import linear_sum_assignment
from acrl.teachers.util import NadarayaWatson
from acrl.teachers.spl.sliced_wasserstein import sliced_wasserstein
from acrl.teachers.abstract_teacher import AbstractTeacher
from acrl.teachers.spl.wasserstein_interpolation import SamplingWassersteinInterpolation


class CurrOT(AbstractTeacher):

    def __init__(self, context_bounds, init_samples, target_sampler, perf_lb, epsilon, callback=None,
                 wait_until_threshold=False, post_sampler=None):
        self.context_bounds = context_bounds
        self.threshold_reached = False
        self.wait_until_threshold = wait_until_threshold
        self.teacher = SamplingWassersteinInterpolation(init_samples, target_sampler, perf_lb, np.sqrt(epsilon),
                                                        self.context_bounds, callback=callback)
        self.success_buffer = WassersteinSuccessBuffer(perf_lb, init_samples.shape[0], epsilon,
                                                       context_bounds=context_bounds)
        self.fail_context_buffer = []
        self.fail_return_buffer = []
        self.sampler = UniformSampler(self.context_bounds)

        self.post_sampler = post_sampler

    def on_rollout_end(self, context, ret):
        self.sampler.update(context, ret)

    def update_distribution(self, contexts, returns, env, episode, log_dir):
        t_up1 = time.time()
        fail_contexts, fail_returns = self.success_buffer.update(contexts, returns,
                                                                 self.teacher.target_sampler(
                                                                     self.teacher.current_samples.shape[0]))

        if self.threshold_reached:
            self.fail_context_buffer.extend(fail_contexts)
            self.fail_context_buffer = self.fail_context_buffer[-self.teacher.n_samples:]
            self.fail_return_buffer.extend(fail_returns)
            self.fail_return_buffer = self.fail_return_buffer[-self.teacher.n_samples:]

        success_contexts, success_returns = self.success_buffer.read_train()
        if len(self.fail_context_buffer) == 0:
            train_contexts = success_contexts
            train_returns = success_returns
        else:
            train_contexts = np.concatenate((np.stack(self.fail_context_buffer, axis=0), success_contexts), axis=0)
            train_returns = np.concatenate((np.stack(self.fail_return_buffer, axis=0), success_returns), axis=0)
        model = NadarayaWatson(train_contexts, train_returns, 0.3 * self.teacher.epsilon)
        t_up2 = time.time()

        t_mo1 = time.time()
        avg_perf = np.mean(model.predict_individual(self.teacher.current_samples))
        if self.threshold_reached or avg_perf >= self.teacher.perf_lb:
            self.threshold_reached = True
            self.teacher.update_distribution(model, self.success_buffer.read_update())
        else:
            print("Current performance: %.3e vs %.3e" % (avg_perf, self.teacher.perf_lb))
            if self.wait_until_threshold:
                print("Not updating sampling distribution, as performance threshold not met")
            else:
                # Which update is better is better?
                self.teacher.update_distribution(model, self.success_buffer.read_update())
                # self.teacher.current_samples = self.success_buffer.read_update()
        t_mo2 = time.time()

        print("Total update took: %.3e (Buffer/Update: %.3e/%.3e)" % (t_mo2 - t_up1, t_up2 - t_up1, t_mo2 - t_mo1))

        # env.plot_dist(episode, self, log_dir)

    def _sample(self):
        sample = self.sampler(self.teacher.current_samples)
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])
    def sample(self):
        sample = self._sample()
        if self.post_sampler is not None:
            while not self.post_sampler(sample):
                sample = self._sample()
        return sample

    def save(self, path):
        self.teacher.save(path)
        self.success_buffer.save(path)
        self.sampler.save(path)

    def load(self, path):
        self.teacher.load(path)
        self.success_buffer.load(path)
        self.sampler.load(path)


class AbstractSuccessBuffer(ABC):

    def __init__(self, delta: float, n: int, epsilon: float, context_bounds: Tuple[np.ndarray, np.ndarray]):
        context_exts = context_bounds[1] - context_bounds[0]
        self.delta_stds = context_exts / 4
        self.min_stds = 0.005 * epsilon * np.ones(len(context_bounds[0]))
        self.context_bounds = context_bounds
        self.delta = delta
        self.max_size = n
        self.contexts = np.zeros((1, len(context_bounds[0])))
        self.returns = np.array([-np.inf])
        self.delta_reached = False
        self.min_ret = None

    @abstractmethod
    def update_delta_not_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        pass

    @abstractmethod
    def update_delta_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        pass

    def update(self, contexts, returns, current_samples):
        assert contexts.shape[0] < self.max_size

        if self.min_ret is None:
            self.min_ret = np.min(returns)

        if not self.delta_reached:
            self.delta_reached, self.contexts, self.returns, mask = self.update_delta_not_reached(contexts, returns,
                                                                                                  current_samples)
        else:
            self.contexts, self.returns, mask = self.update_delta_reached(contexts, returns, current_samples)

        return contexts[mask, :], returns[mask]

    def read_train(self):
        return self.contexts.copy(), self.returns.copy()

    def read_update(self):
        # Compute the Gaussian search noise that we add to the samples
        var_scales = np.clip(self.delta - self.returns, 0., np.inf) / (self.delta - self.min_ret)
        stds = self.min_stds[None, :] + var_scales[:, None] * self.delta_stds[None, :]

        # If we did not yet reach the desired threshold we enforce exploration by scaling the exploration noise w.r.t.
        # the distance to the desired threshold value
        if not self.delta_reached:
            offset = self.returns.shape[0] // 2
            sub_returns = self.returns[offset:]
            sub_contexts = self.contexts[offset:, :]
            sub_stds = stds[offset:, :]

            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = sub_returns - self.returns[offset - 1]
            norm = np.sum(probs)
            if norm == 0:
                probs = np.ones(sub_returns.shape[0]) / sub_returns.shape[0]
            else:
                probs = probs / norm

            sample_idxs = np.random.choice(sub_returns.shape[0], self.max_size, p=probs)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_stds = sub_stds[sample_idxs, :]
        else:
            to_fill = self.max_size - self.returns.shape[0]
            add_idxs = np.random.choice(self.returns.shape[0], to_fill)
            sampled_contexts = np.concatenate((self.contexts, self.contexts[add_idxs, :]), axis=0)
            sampled_stds = np.concatenate((stds, stds[add_idxs, :]), axis=0)

        contexts = sampled_contexts + np.random.normal(0, sampled_stds, size=(self.max_size, self.contexts.shape[1]))
        invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                       contexts > self.context_bounds[1][None, :]), axis=-1)
        count = 0
        while np.any(invalid) and count < 10:
            new_noise = np.random.normal(0, sampled_stds[invalid, :], size=(np.sum(invalid), self.contexts.shape[1]))
            contexts[invalid, :] = sampled_contexts[invalid, :] + new_noise
            invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                           contexts > self.context_bounds[1][None, :]), axis=-1)
            count += 1

        return np.clip(contexts, self.context_bounds[0], self.context_bounds[1])

    def get_data(self) -> Any:
        return None

    def set_data(self, data: Any) -> NoReturn:
        pass

    def save(self, path):

        with open(os.path.join(path, "teacher_success_buffer.pkl"), "wb") as f:
            pickle.dump((self.delta, self.max_size, self.min_stds, self.delta_stds, self.contexts, self.returns,
                         self.delta_reached, self.min_ret, self.get_data()), f)

    def load(self, path):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "rb") as f:
            self.delta, self.max_size, self.min_stds, self.delta_stds, self.contexts, self.returns, \
            self.delta_reached, self.min_ret, data = pickle.load(f)
        self.set_data(data)


class WassersteinSuccessBuffer(AbstractSuccessBuffer):

    def __init__(self, delta: float, n: int, epsilon: float, context_bounds: Tuple[np.ndarray, np.ndarray]):
        super().__init__(delta, n, epsilon, context_bounds)

    def update_delta_not_reached(self, contexts: np.ndarray, returns: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        # Only add samples that have a higher return than the median return in the buffer (we do >= here to allow
        # for binary rewards to work)
        med_idx = self.returns.shape[0] // 2
        mask = returns >= self.returns[med_idx]
        n_new = np.sum(mask)
        print("Improving buffer quality with %d samples" % n_new)

        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_returns = np.concatenate((returns[mask], self.returns[offset_idx:]), axis=0)
        new_contexts = np.concatenate((contexts[mask, :], self.contexts[offset_idx:, :]), axis=0)
        sort_idxs = np.argsort(new_returns)

        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_returns.shape[0]

        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_returns.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            rem_mask[mask] = [i not in sort_idxs for i in np.arange(n_new)]

        new_delta_reached = self.returns[self.returns.shape[0] // 2] > self.delta
        return new_delta_reached, new_contexts[sort_idxs, :], new_returns[sort_idxs], rem_mask

    def update_delta_reached(self, contexts: np.ndarray, returns: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        # Compute the new successful samples
        mask = returns >= self.delta
        n_new = np.sum(mask)

        if n_new > 0:
            remove_mask = self.returns < self.delta
            if not np.any(remove_mask) and self.returns.shape[0] >= self.max_size:
                extended_contexts = np.concatenate((self.contexts, contexts[mask, :]), axis=0)
                extended_returns = np.concatenate((self.returns, returns[mask]), axis=0)

                # At this stage we use the optimizer
                dists = np.sum(np.square(extended_contexts[:, None, :] - current_samples[None, :, :]), axis=-1)
                assignments = linear_sum_assignment(dists, maximize=False)
                ret_idxs = assignments[0]

                # Select the contexts using the solution from the MIP solver. The unique functions sorts the data
                new_contexts = extended_contexts[ret_idxs, :]
                new_returns = extended_returns[ret_idxs]

                # We update the mask to indicate only the kept samples
                mask[mask] = [idx in (ret_idxs - self.contexts.shape[0]) for idx in np.arange(n_new)]

                avg_dist = sliced_wasserstein(new_contexts, current_samples)
                print("Updated success buffer with %d samples. New Wasserstein distance: %.3e" % (n_new, avg_dist))
            else:
                # We replace the unsuccessful samples by the successful ones
                if n_new < np.sum(remove_mask):
                    remove_idxs = np.argpartition(self.returns, kth=n_new)[:n_new]
                    remove_mask = np.zeros(self.returns.shape[0], dtype=bool)
                    remove_mask[remove_idxs] = True

                new_returns = np.concatenate((returns[mask], self.returns[~remove_mask]), axis=0)
                new_contexts = np.concatenate((contexts[mask, :], self.contexts[~remove_mask, :]), axis=0)

                if new_returns.shape[0] > self.max_size:
                    new_returns = new_returns[:self.max_size]
                    new_contexts = new_contexts[:self.max_size, :]

                # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
                assert self.contexts.shape[0] <= new_contexts.shape[0]
                assert new_contexts.shape[0] == new_returns.shape[0]
        else:
            new_contexts = self.contexts
            new_returns = self.returns

        return new_contexts, new_returns, ~mask


class AbstractSampler(ABC):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        self.noise = 1e-3 * (context_bounds[1] - context_bounds[0])

    def update(self, context: np.ndarray, ret: float) -> NoReturn:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.select(samples) + np.random.uniform(-self.noise, self.noise)

    @abstractmethod
    def select(self, samples: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass


class UniformSampler(AbstractSampler):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        super(UniformSampler, self).__init__(context_bounds)

    def select(self, samples: np.ndarray) -> np.ndarray:
        return samples[np.random.randint(0, samples.shape[0]), :]
