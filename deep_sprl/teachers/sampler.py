from typing_extensions import NoReturn
import numpy as np

from deep_sprl.environments.minigrid.envs import AEnv, BEnv, CEnv

class MinigridSampler:
    def __init__(self, context_lb, context_ub):
        self.LOWER_CONTEXT_BOUNDS = context_lb
        self.UPPER_CONTEXT_BOUNDS = context_ub

    def sample(self, samples=None, size=None):
        count = 0
        if samples is None:
            if size is None:
                sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
                while not AEnv._is_feasible(sample):
                    sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
                return sample
            else:
                return np.array([self.sample() for i in range(size)])
        else:
            sample = samples[np.random.randint(0, samples.shape[0]), :]
            while not AEnv._is_feasible(sample):
                sample = samples[np.random.randint(0, samples.shape[0]), :]
                count += 1
                if count > 100:
                    print(sample)
                    return sample
            return sample

    def __call__(self, samples=None, size=None):
        return self.sample(samples, size)

    def update(self, context: np.ndarray, ret: float) -> NoReturn:
        pass

    def select(self, samples: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass


class Subsampler:

    def __init__(self, lb, ub, bins):
        eval_points = [np.linspace(lb[i], ub[i], bins[i] + 1)[:-1] for i in range(len(bins))]
        eval_points = [s + 0.5 * (s[1] - s[0]) for s in eval_points]
        self.bin_sizes = np.array([s[1] - s[0] for s in eval_points])
        self.eval_points = np.stack([m.reshape(-1, ) for m in np.meshgrid(*eval_points)], axis=-1)

    def __call__(self, discrete_sample):
        return self.eval_points[discrete_sample, :] + np.random.uniform(-0.5 * self.bin_sizes, 0.5 * self.bin_sizes)
