import torch
import torch.nn as nn
from .util import mlp, ConvNet, ConvNetMLP, DEFAULT_DEVICE


class TwinQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        # + 3 for goal info dim
        mlp_dims = [hidden_dim + 3 + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = ConvNetMLP(
            convnet=ConvNet(obs_dim, DEFAULT_DEVICE, feature_dim=hidden_dim),
            mlp=mlp(mlp_dims, squeeze_output=True),
            device=DEFAULT_DEVICE
        )
        self.q2 = ConvNetMLP(
            convnet=ConvNet(obs_dim, DEFAULT_DEVICE, feature_dim=hidden_dim),
            mlp=mlp(mlp_dims, squeeze_output=True),
            device=DEFAULT_DEVICE
        )

    def both(self, state, goalinfo, action):
        additional_info = torch.cat((goalinfo, action), dim=1)
        return self.q1(state, additional_info), self.q2(state, additional_info)

    def forward(self, state, goalinfo, action):
        return torch.min(*self.both(state, goalinfo, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        mlp_dims = [state_dim + 3, *([hidden_dim] * n_hidden), 1]
        self.v = ConvNetMLP(
            convnet=ConvNet(state_dim, DEFAULT_DEVICE, feature_dim=hidden_dim),
            mlp=mlp(mlp_dims, squeeze_output=True),
            device=DEFAULT_DEVICE
        )

    def forward(self, observation, goalinfo):
        return self.v(observation, goalinfo)