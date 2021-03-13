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

def validate_bounds(action_space):
    bounded = (action_space.low[0] != float('-inf'))
    if bounded:
        err_str = "closed action spaces must be [-1, 1]^n"
        assert all([low == -1 for low in action_space.low]), err_str
        assert all([high == 1 for high in action_space.high]), err_str
    else:
        err_str = "open action spaces must be open on all dimensions"
        assert all([low == float('-inf') for low in action_space.low]), err_str
        assert all([high == float('inf') for high in action_space.high]), err_str
    return bounded


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

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPGaussianActor(Actor):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 activation,
                 std_dim=1,
                 std_source=None,
                 std_value=0.5,
                 squash=True):
        super().__init__()
        self.act_dim = act_dim
        self.squash = squash

        # Initialize Mean Network Architecture
        self.base_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], act_dim), nn.Tanh())
   
        # Initialize Variance Parameters / Network Architecture
        assert std_dim in [0, 1]
        if std_source is None:
            self.log_std = torch.tensor(np.log(std_value))
        elif not std_source:
            self.log_std = nn.Parameter(torch.squeeze(np.log(std_value)*torch.ones(pow(act_dim, std_dim))))
        else:
            self.log_layer = nn.Linear(hidden_sizes[-1], pow(act_dim, std_dim))
        self.std_dim = std_dim
        self.std_source = std_source

    def _distribution(self, obs):
        mu = self.mu_layer(self.base_net(obs))
        log_std = self.log_std if not self.std_source else self.log_layer(self.base_net(obs)) 
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Distribution
        logp_pi = pi.log_prob(act).sum(axis=-1)
        # Make adjustment to log prob if squashing
        if self.squash:
            logp_pi += (2 * (act + F.softplus(-2 * act) - np.log(2))).sum(axis=-1)
        return logp_pi


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
                 std_dim=1,
                 std_source=False,
                 std_value=0.5,
                 squash=False):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Gather action space limits, create lob prob correction funcs, use Gaussian Actor
        # if action space is Box
        if isinstance(action_space, Box):
            self.is_discrete = False
            err_str = "multidimensional action space shape not supported by MLPGaussianActor"
            assert len(action_space.shape) == 1, err_str

            self.squash = squash

            # Get bounded parameter and validate action space bounds
            self.bounded = validate_bounds(action_space)
            
            # Build Policy (pass in corrections to log probs as lambdas taking action)
            self.pi = MLPGaussianActor(obs_dim,
                                       action_space.shape[0],
                                       hidden_sizes,
                                       activation,
                                       std_dim=std_dim,
                                       std_value=std_value,
                                       std_source=std_source,
                                       squash=squash)

        # Use Categorical Actor if action space is Discrete
        elif isinstance(action_space, Discrete):
            self.is_discrete = True
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            if deterministic and not self.is_discrete:
                a = pi.mean
            elif deterministic:
                values = pi.enumerate_support()
                a = values[torch.argmax(torch.tensor([pi.log_prob(act) for act in values]))]
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if not self.is_discrete and self.squash:
                a = torch.tanh(a)
            elif not self.is_discrete and self.bounded:
                a = torch.clamp(a, -1, 1)
            v = self.v(obs)
        return np.array(a), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs, deterministic=True)[0]
