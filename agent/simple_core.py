import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from enum import Enum
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 activation,
                 **kwargs):

        super().__init__()

        # Initialize Gaussian Parameters and Network Architecture
        self.act_dim = act_dim
        self.mean_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.logstd = nn.Parameter(torch.zeros(self.act_dim, dtype=torch.float32))

    def _distribution(self, obs):
        batch_mean = self.mean_net(obs)
        scale_tril = torch.diag(torch.exp(self.logstd))
        action_distribution = MultivariateNormal(
            batch_mean,
            scale_tril=scale_tril,
        )

        return action_distribution

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(64,64),
                 activation=nn.Tanh,
                 action_low = None,
                 action_high = None):

        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # Gather action space limits, create lob prob correction funcs, use Gaussian Actor
        # if action space is Box
        if isinstance(action_space, Box):
            self.is_discrete = False

            # Get action space bounds and masks specifying closed and half-open dims
            self.pi = MLPGaussianActor(obs_dim,
                                       act_dim,
                                       hidden_sizes,
                                       activation,
                                       )

        # Use Categorical Actor if action space is Discrete
        elif isinstance(action_space, Discrete):
            self.is_discrete = True
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs)

            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)

            v = self.v(obs)

        return np.array(a), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
