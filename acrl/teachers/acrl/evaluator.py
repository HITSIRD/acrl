import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from acrl.teachers.acrl.util import FeatureExtractor
from acrl.util.device import device


class Evaluator:
    def __init__(self, config):
        self.config = config

        # initialise the evaluator net
        self.net = self.initialize_net(config['evaluator_layers'], config['context_dim']).to(device)
        self.min_ret = None
        self.max_ret = None
        self.available = False

        if self.net is not None:
            self.opt = torch.optim.Adam([*self.net.parameters()], lr=self.config['lr_evaluator'])

    def initialize_net(self, layers, context_dim):
        return EvaluatorNet(layers, context_dim)

    def predict(self, context):
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context).float().to(device)
        return self.net(context).detach().cpu().numpy(), self.min_ret, self.max_ret

    def update(self, buffer):
        self.available = True

        ret = buffer(reset=False)[0]
        self.min_ret = min(ret)
        self.max_ret = max(ret)
        # bce_loss = torch.nn.BCELoss(reduction='none')
        mse_loss = torch.nn.MSELoss(reduction='none')

        for i in range(self.config['num_evaluator_update']):
            context, ret = buffer(reset=False, batch_size=self.config['evaluator_batchsize'])
            context = torch.from_numpy(context).float().to(device)
            ret = torch.from_numpy(ret).float().to(device)

            # ret = (ret > 0.5).float()

            pred_ret = self.net(context).flatten()
            # pred_ret = torch.clamp(pred_ret, min=1e-7, max=1 - 1e-7)
            loss = mse_loss(ret, pred_ret).mean()
            assert loss.requires_grad
            print(f'loss: {loss}')

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def plot(self, episode):
        pos = []
        pred = []
        for i in range(1, 15):
            for j in range(1, 15):
                pos.append([i, j])
                pred.append(self.predict(np.array([i, j]))[0])

        fig = plt.figure()
        ax = fig.add_subplot()
        pos = np.array(pos)
        pred = np.array(pred)
        ax.add_patch(plt.Rectangle((0, 0), 15, 15, color=(0.8, 0.8, 0.8)))
        ax.scatter(pos[:, 0], pos[:, 1], c=pred.flatten(), cmap='viridis')
        plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(min(pred), max(pred)), cmap='viridis'), ax=ax)
        plt.tight_layout()
        plt.grid()

        plt.savefig(f'/home/wenyongyan/图片/experiments/minigrid_g/acrl_wo_rl/pred_iter_{episode}.pdf')


class EvaluatorNet(nn.Module):
    def __init__(self, layers, context_dim, embedding_dim=8):
        super(EvaluatorNet, self).__init__()

        self.fc_layers = nn.ModuleList([])
        self.encoder = FeatureExtractor(context_dim, embedding_dim, F.relu)

        curr_input_dim = embedding_dim
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        self.fc_output = nn.Linear(curr_input_dim, 1)

    def forward(self, context):
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context).float().to(device)
        h = self.encoder(context)
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        return self.fc_output(h)
