import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Function
import torch

from acrl.teachers.acrl.decoder import TaskDecoder, StateDecoder
from acrl.teachers.acrl.encoder import Encoder
from acrl.util.device import device


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        latents = vq(z_e_x, self.embedding.weight)

        return latents

    def straight_through(self, z_e_x, inactive_count=None):
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach(), inactive_count)
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x)

        return z_q_x, z_q_x_bar, indices


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook, inactive_count):
        # with torch.no_grad():
        #     embedding_size = codebook.size(1)
        #     inputs_size = inputs.size()
        #     inputs_flatten = inputs.view(-1, embedding_size)
        #
        #     codebook_sqr = torch.sum(codebook ** 2, dim=1)
        #     inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        #
        #     distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
        #
        #     _, indices_flatten = torch.min(distances, dim=1)
        #     if len(inputs_size) == 1:
        #         indices = indices_flatten.view(-1)
        #     else:
        #         indices = indices_flatten.view(*inputs_size[:-1])
        #
        #     ctx.mark_non_differentiable(indices)

        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            # Compute distances between inputs and codebook
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            # Find nearest code for each input
            _, indices_flatten = torch.min(distances, dim=1)
            if len(inputs_size) == 1:
                indices = indices_flatten.view(-1)
            else:
                indices = indices_flatten.view(*inputs_size[:-1])

            if inactive_count is not None:
                threshold = 10

                # Update usage statistics
                indices_np = indices_flatten.cpu().numpy()

                # Update usage statistics
                inactive_count += 1
                inactive_count[indices_np] = 0

                # Resample inactive codes
                inactive_codes = np.where(inactive_count >= threshold)[0]
                if len(inactive_codes) > 0:
                    # Compute distances between batch embeddings and codebook
                    batch_sqr = torch.sum(inputs_flatten ** 2, dim=1)
                    batch_distances = torch.addmm(
                        batch_sqr.unsqueeze(1),
                        inputs_flatten,
                        codebook.t(),
                        alpha=-2.0,
                        beta=1.0
                    )

                    # Probability proportional to squared distance
                    dq2 = batch_distances.min(dim=1).values ** 2
                    probabilities = dq2 / dq2.sum()

                    # Sample new embeddings for inactive codes
                    sampled_indices = np.random.choice(
                        np.arange(len(inputs_flatten)),
                        size=len(inactive_codes),
                        p=probabilities.cpu().numpy()
                    )
                    sampled_embeddings = inputs_flatten[sampled_indices]

                    # Re-initialize inactive codes
                    for idx, code_idx in enumerate(inactive_codes):
                        codebook[code_idx] = sampled_embeddings[idx]

                    # Reset inactive count
                    inactive_count[inactive_codes] = 0

            ctx.mark_non_differentiable(indices)

            return indices


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook, inactive_count):
        indices = vq(inputs, codebook, inactive_count)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_putput_flatten = (grad_output.contiguous().view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_putput_flatten)

        return grad_inputs, grad_codebook, None


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]


class VQVAE(nn.Module):
    def __init__(self,  # network size
                 config):
        super(VQVAE, self).__init__()

        self.config = config
        self.encoder = self.initialize_encoder()
        self.decoder = self.initialize_decoder()
        self.codebook = self.initialize_codebook(k=config['codebook_k'], dim=config['latent_dim'])
        self.beta = config['vq_beta']
        self.batch_size = config['vq_batch_size']
        self.used_count = np.zeros([config['codebook_k']])
        self.inactive_count = np.ones([self.config['codebook_k']])

        self.rollout_storage = RolloutStorage(buffer_size=int(self.config['vq_buffer_size']),
                                              state_dim=self.config['state_dim'],
                                              task_dim=self.config['context_dim'])

        self.update_count = 0

        self.opt = optim.Adam([*self.encoder.parameters(), *self.decoder.parameters(), *self.codebook.parameters()],
                              lr=config['lr_vqvae'])

    def initialize_encoder(self):
        encoder = Encoder(
            encoder_layers=self.config['encoder_layers'],
            latent_dim=self.config['latent_dim'],
            state_dim=self.config['state_dim'],
            state_embed_dim=self.config['state_embedding_size'],
            # task_dim=self.config['context_dim'],
            # task_embed_dim=self.config['task_embedding_size']
        ).to(device)

        return encoder

    def initialize_decoder(self):
        decoder = StateDecoder(
            layers=self.config['decoder_layers'],
            latent_dim=self.config['latent_dim'],
            state_dim=self.config['state_dim'],
            # task_dim=self.config['context_dim'],
        ).to(device)

        return decoder

    def initialize_codebook(self, k, dim):
        codebook = VQEmbedding(k, dim).to(device)
        return codebook

    def forward(self, x):
        z = self.encoder(x)
        return z

    def update(self):
        if self.update_count % 20 == 0:
            self.used_count = np.zeros([self.config['codebook_k']])
        self.update_count += 1

        inputs = self.rollout_storage.get_batch(self.batch_size)

        z_e_x = self.encoder(inputs)

        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x, inactive_count=self.inactive_count)
        self.used_count[indices.cpu().numpy().flatten()] += 1

        # decode
        reconstruction = self.decoder(z_q_x_st)
        mse_loss = torch.nn.MSELoss()

        # reconstruction loss
        rec_loss = mse_loss(reconstruction, inputs)

        # VQ loss
        vq_loss = mse_loss(z_q_x, z_e_x.detach())

        # commitment loss
        commitment_loss = self.beta * mse_loss(z_e_x, z_q_x.detach())

        loss = rec_loss + vq_loss + self.beta * commitment_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        unused_rate = np.sum(self.used_count == 0) / self.config['codebook_k']

        loss_log = {}
        loss_log['unused_rate'] = unused_rate
        loss_log['rec_loss'] = rec_loss.mean().item()
        loss_log['vq_loss'] = vq_loss.mean().item()
        loss_log['commitment_loss'] = commitment_loss.mean().item()
        loss_log['loss'] = loss.mean().item()
        return loss_log


class RolloutStorage(object):
    def __init__(self, buffer_size, state_dim, task_dim):
        """
        Store everything that is needed for the VAE update
        """

        self.task_dim = task_dim
        if isinstance(state_dim, tuple) and len(state_dim) > 0:
            self.obs_dim = np.array(state_dim).prod()
        else:
            self.obs_dim = state_dim

        # self.obs_dim += self.task_dim  # cat context

        self.max_buffer_size = buffer_size  # maximum buffer len (number of transitions)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.step_idx = 0  # at which index we're currently inserting the new step
        self.buffer_len = 0  # how much of the buffer has been fill

        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_buffer_size, self.obs_dim))
            self.next_state = torch.zeros((self.max_buffer_size, self.obs_dim))
            self.task = torch.zeros((self.max_buffer_size, task_dim))

    def clear(self):
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.step_idx = 0
        self.buffer_len = 0  # how much of the buffer has been fill

    def insert(self, step=None):
        if step is not None:
            prev_state, action, reward, next_state, task = step
            self.prev_state[self.insert_idx] = torch.from_numpy(prev_state)
            self.next_state[self.insert_idx] = torch.from_numpy(next_state)
            self.task[self.insert_idx] = torch.from_numpy(np.array(task))

            self.insert_idx = (self.insert_idx + 1) % self.max_buffer_size
            self.buffer_len = max(self.buffer_len, self.insert_idx + 1)

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batch_size=5, replace=False):
        batch_size = min(self.buffer_len, batch_size)
        # assert index.size > 0
        size = min(self.buffer_len, batch_size)
        index = np.array(range(size))
        rollout_indices = index[np.random.choice(len(index), size, replace=replace)]

        states = self.next_state[rollout_indices].to(device)

        return states
