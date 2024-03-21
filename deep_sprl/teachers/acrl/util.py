import numpy as np
import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda:0')

else:
    device = torch.device('cpu')


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
    prev_state = env.reset(task)
    prev_state = torch.from_numpy(prev_state.reshape((1, -1))).float().to(device)
    task = torch.from_numpy(np.array(env.env.context)).float().view(-1)

    if hasattr(env.env, 'max_steps'):
        max_steps = env.env.max_steps
    else:
        max_steps = 500

    for step_idx in range(0, max_steps):
        episode_prev_obs.append(prev_state)
        with torch.no_grad():
            # latent = utl.get_latent_for_policy(args,
            #                                    latent_sample=curr_latent_sample,
            #                                    latent_mean=curr_latent_mean,
            #                                    latent_logvar=curr_latent_logvar)

            action, _, _ = policy(prev_state)
            # _, action = policy.act(state=None, pos=info['pos'], dir=info['dir'], task=info['task'])
        # action = action.view((1, *action.shape))
        action = action.squeeze(0).cpu()

        # observe reward and next obs
        next_state, rew_raw, done, info = env.step(action, update=False)
        next_state = next_state.reshape((1, -1))

        next_state = torch.from_numpy(next_state).to(device)

        episode_next_obs.append(next_state)
        episode_rewards.append(torch.from_numpy(np.array(rew_raw)))
        episode_actions.append(action)

        curr_latent_sample, curr_latent_mean, curr_latent_logvar = encoder(
            prev_states=prev_state,
            actions=action.unsqueeze(0).to(device),
            rewards=torch.from_numpy(np.array(rew_raw)).view(1, 1).to(device),
            next_states=next_state,
            tasks=task)

        episode_latent_samples.append(curr_latent_sample.clone())
        episode_latent_means.append(curr_latent_mean.clone())
        episode_latent_logvars.append(curr_latent_logvar.clone())

        episode_returns += rew_raw
        # episode_lengths = step_idx + 1

        if done:
            break

        prev_state = next_state

    return episode_latent_means, episode_latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, episode_returns
