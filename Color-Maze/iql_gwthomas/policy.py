import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .util import mlp, ConvNet, DEFAULT_DEVICE, ConvNetMLP


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.convnet = ConvNet(obs_dim, DEFAULT_DEVICE, feature_dim=hidden_dim)
        # + 3 for goal info dim
        self.mlp = mlp([self.convnet.feature_dim + 3, *([hidden_dim] * n_hidden), act_dim])

        self.net = ConvNetMLP(
            convnet=self.convnet,
            mlp=self.mlp,
            device=DEFAULT_DEVICE
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs, goalinfo):
        mean = self.net(obs, goalinfo)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, goalinfo, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs, goalinfo)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.convnet = ConvNet(obs_dim, DEFAULT_DEVICE, feature_dim=hidden_dim)
        # + 3 for goal info dim
        self.mlp = mlp([self.convnet.feature_dim + 3, *([hidden_dim] * n_hidden), act_dim])

        self.net = ConvNetMLP(
            convnet=self.convnet,
            mlp=self.mlp,
            device=DEFAULT_DEVICE
        )

    def forward(self, obs, goalinfo):
        return self.net(obs, goalinfo)

    def act(self, obs, goalinfo, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs, goalinfo)
