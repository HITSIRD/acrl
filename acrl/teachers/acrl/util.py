import numpy as np
import torch
from torch import nn
from acrl.util.device import device


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function, layers=None):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            if isinstance(input_size, tuple) and len(input_size) > 0:
                input_size = np.array(input_size).prod()
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def trajectory_embedding(episode_latent_means, episode_latent_logvars):
    if isinstance(episode_latent_means, torch.Tensor):
        means = episode_latent_means
        logvars = episode_latent_logvars
    else:
        means = torch.stack(episode_latent_means)
        logvars = torch.stack(episode_latent_logvars)
    vars = logvars.exp()
    vars = vars.clamp(min=1e-7)
    vars_rec = vars.reciprocal()
    vars = 1. / torch.sum(vars_rec, dim=0)
    task_means = vars * torch.sum(means * vars_rec, dim=0)
    task_logvars = torch.log(vars)

    # task_means = (2 * torch.rand(*task_means.shape) - 1).float().to(device)
    return task_means, task_logvars


def sample_trajectory(env, policy, encoder, task):
    episode_prev_obs = []
    episode_next_obs = []
    episode_actions = []
    episode_rewards = []
    episode_returns = 0

    episode_latent_samples = []
    episode_latent_means = []
    episode_latent_logvars = []

    # --- roll out policy ---
    prev_state, prev_state_wo_context = env.reset(task, wo_context=True)
    prev_state = torch.from_numpy(prev_state).unsqueeze(0).float()
    prev_state_wo_context = torch.from_numpy(prev_state_wo_context).float()
    task = torch.tensor(task).float()

    if hasattr(env.env, 'max_steps'):
        max_steps = env.env.max_steps
    else:
        max_steps = env.env.env.spec.max_episode_steps

    for step_idx in range(0, max_steps):
        episode_prev_obs.append(prev_state_wo_context)
        with torch.no_grad():
            action = policy.predict(prev_state, deterministic=False)[0]  # TODO: SAC policy needs 3 parameters
        # action = torch.from_numpy(action).squeeze(0)
        action = action.flatten()

        # observe reward and next obs
        next_state_wo_context, next_state, rew_raw, done, info = env.step(action, update=False, wo_context=True,
                                                                          insert=False)

        # action = torch.from_numpy(action).squeeze(0)

        # next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
        # next_state_wo_context = torch.from_numpy(next_state_wo_context).to(device)

        # episode_next_obs.append(next_state_wo_context)
        episode_rewards.append(torch.tensor(rew_raw))
        # episode_actions.append(action)

        # curr_latent_sample, curr_latent_mean, curr_latent_logvar = encoder(
        #     prev_states=prev_state_wo_context,
        #     actions=action.to(device),
        #     rewards=torch.tensor([rew_raw]).to(device),
        #     next_states=next_state_wo_context,
        #     tasks=task)
        #
        # episode_latent_samples.append(curr_latent_sample.clone())
        # episode_latent_means.append(curr_latent_mean.clone())
        # episode_latent_logvars.append(curr_latent_logvar.clone())

        episode_returns += rew_raw
        # episode_lengths = step_idx + 1

        if done:
            break

        prev_state = next_state
        prev_state_wo_context = next_state_wo_context

    return episode_returns


def get_latent_map(buffer, encoder):
    # episode_latent_samples = []
    # latent_means = []
    # latent_logvars = []
    # episode_return = []
    # tasks = []
    #
    # for i in range(buffer.buffer_len):
    #     if buffer.trajectory_lens[i] <= 1:
    #         continue
    #     episode_latent_means = []
    #     episode_latent_logvars = []
    #     episode_return.append(buffer.episode_return[i])
    #     # for j in range(buffer.trajectory_lens[i]):
    #     curr_latent_sample, curr_latent_mean, curr_latent_logvar = encoder(
    #         prev_states=buffer.prev_state[i][:buffer.trajectory_lens[i]].to(device),
    #         actions=buffer.action[i][:buffer.trajectory_lens[i]].to(device),
    #         rewards=buffer.reward[i][:buffer.trajectory_lens[i]].to(device),
    #         next_states=buffer.next_state[i][:buffer.trajectory_lens[i]].to(device),
    #         tasks=buffer.task[i].to(device))
    #
    #     # episode_latent_samples.append(curr_latent_sample.clone())
    #     episode_latent_means.append(curr_latent_mean.clone())
    #     episode_latent_logvars.append(curr_latent_logvar.clone())
    #
    #     # mean, logvar = trajectory_embedding(episode_latent_means, episode_latent_logvars)
    #     mean, logvar = trajectory_embedding(curr_latent_mean, curr_latent_logvar)
    #     latent_means.append(mean)
    #     latent_logvars.append(logvar)
    #     tasks.append(buffer.task[i])
    #
    # return latent_means, latent_logvars, tasks, np.array(episode_return)

    latent = []
    episode_return = []
    tasks = []
    horizon = []
    for i in range(buffer.buffer_len):
        if buffer.trajectory_lens[i] <= 1:
            continue
        episode_latent_means = []
        episode_latent_logvars = []
        episode_return.append(buffer.episode_return[i])
        # for j in range(buffer.trajectory_lens[i]):
        # curr_latent_sample, curr_latent_mean, curr_latent_logvar = encoder(
        #     prev_states=buffer.prev_state[i][:buffer.trajectory_lens[i]].to(device),
        #     actions=buffer.action[i][:buffer.trajectory_lens[i]].to(device),
        #     rewards=buffer.reward[i][:buffer.trajectory_lens[i]].to(device),
        #     next_states=buffer.next_state[i][:buffer.trajectory_lens[i]].to(device),
        #     tasks=buffer.task[i].to(device))

        # episode_latent_samples.append(curr_latent_sample.clone())
        # episode_latent_means.append(curr_latent_mean.clone())

        # mean, logvar = trajectory_embedding(episode_latent_means, episode_latent_logvars)
        latent.append(encoder(buffer.task[i].to(device)))
        tasks.append(buffer.task[i])
        horizon.append(buffer.trajectory_lens[i])

    return latent, tasks, np.array(episode_return), horizon