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

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPGaussianActor(Actor):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 activation,
                 dim_rand = 1, # TODO: hard code, fix fix fix @kiran or @surya; how to pass in cleanly?
                 **kwargs):
        super().__init__()


        # Initialize Gaussian Parameters and Network Architecture
        self.act_dim = act_dim
        self.sigma_num_dims = dim_rand
        self.base_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_layer = nn.Linear(hidden_sizes[-1], pow(act_dim, dim_rand))
        self.has_masks = False
        if 'closed_mask' in kwargs:
            self.has_masks = True
            self.closed_mask = kwargs['closed_mask']
            assert 'half_open_mask' in kwargs and 'correction' in kwargs
            self.half_open_mask = kwargs['half_open_mask']
            self.closed_bound_correction = kwargs['correction']

    def _distribution(self, obs):
        mu = self.mu_layer(self.base_net(obs))
        log_std = self.log_layer(self.base_net(obs))
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        new_shape = (-1,) + self.sigma_num_dims * (self.act_dim,)
        if self.sigma_num_dims == 2:
            return MultivariateNormal(mu, torch.squeeze(torch.reshape(std, new_shape)))
        return Normal(mu, torch.squeeze(torch.reshape(std, new_shape)))

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Distribution
        logp_pi = pi.log_prob(act).sum(axis=-1)

        # Make coordinate-wise adjustment to log prob
        if self.has_masks:
            # corr = -log(2 * (high - low)) is the non-action-dependent term
            # in the logp correction for closed interval action space dimensions
            corr = self.closed_bound_correction + 2 * (act + F.softmax(-2 * act, dim=-1))
            logp_pi += (corr * self.closed_mask).sum(axis=-1)
            logp_pi -= (act * self.half_open_mask).sum(axis=-1)
        return logp_pi


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

def mask_constructor(action_space):
    action_limits = zip(action_space.low, action_space.high)
    low_open, high_open = lambda l: l == float('-inf'), lambda h: h == float('inf')
    closed_mask, open_above_mask, open_below_mask = [], [], []
    for low, high in action_limits:
        closed_func = lambda x: 2 * (x + F.softplus(-2 * x)) - np.log(2 * (high - low))
        # For two-sided limits, we need both limit bounds to calculate logp diff.
        # Function _log_prob_from_distribution needs correction in the form of:
        # 2 * (a + F.softplus(-2 * a)) - np.log(2 * (high - low))
        # Where a is (pre-squashed) action and high and low are action space bounds.
        # The second term here is supplied to the Actor as a 'correction' tensor in kwargs,
        # And a mask of fully closed action dimensions is passed to calculate first term.
        if not low_open(low) and not high_open(high):
            closed_mask += [1]
            open_above_mask += [0]
            open_below_mask += [0]
        # For one-sided limits, logp diff is always just -a, where a is the presquashed action.
        # The sum of the open_above and open_below masks is given to the Actor as mask
        # for this logp correction. Masks are kept separate for action-mapping purposes.
        elif not high_open(high) or not low_open(low):
            closed_mask += [0]
            open_above_mask += [1 if high_open(high) else 0]
            open_below_mask += [1 if low_open(low) else 0]
        # Fully open action dimensions require no action-mapping or logp correction.
        else:
            closed_mask += [0]
            open_above_mask += [0]
            open_below_mask += [0]
    return tuple([torch.as_tensor(el) for el in [closed_mask, open_above_mask, open_below_mask]])


class MLPActorCritic(nn.Module):


    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(64,64),
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]


        # Gather action space limits, create lob prob correction funcs, use Gaussian Actor
        # if action space is Box
        if isinstance(action_space, Box):
            self.is_discrete = False
            err_str = "multidimensional action space shape not supported by MLPGaussianActor"
            assert len(action_space.shape) == 1, err_str

            # Get action space bounds and masks specifying closed and half-open dims
            self.lows, self.highs = np.array(action_space.low), np.array(action_space.high)
            self.lows = torch.as_tensor(np.nan_to_num(self.lows, neginf=0, posinf=0))
            self.highs = torch.as_tensor(np.nan_to_num(self.highs, neginf=0, posinf=0))
            self.closed, self.open_above, self.open_below = mask_constructor(action_space)
            # Build Policy (pass in corrections to log probs as lambdas taking action)
            self.pi = MLPGaussianActor(obs_dim,
                                       action_space.shape[0],
                                       hidden_sizes,
                                       activation,
                                       closed_mask=self.closed,
                                       half_open_mask=self.open_above + self.open_below,
                                       correction=-torch.log(2 * (self.highs - self.lows)))

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
            elif deterministic and self.is_discrete:
                values = pi.enumerate_support()
                a = values[torch.argmax(torch.tensor([pi.log_prob(act) for act in values]))]
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if not self.is_discrete:
                # Map actions to squashed action space using masks generated in __init__.
                lows, highs = self.lows, self.highs
                a = np.where(self.closed, lows + (highs - lows) * (np.tanh(a) + 1) / 2, a)
                a = np.where(self.open_above, lows + torch.from_numpy(np.exp(a)), a)
                a = np.where(self.open_below, highs - torch.from_numpy(np.exp(a)), a)
                a = np.clip(a, lows, highs)
            v = self.v(obs)
        return np.array(a), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs, deterministic=True)[0]
