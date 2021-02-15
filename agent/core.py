import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from enum import Enum
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
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
                 logprob_corrections, 
                 dim_rand=1):
        super().__init__()

        
        # Initialize Gaussian Parameters and Network Architecture
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.base_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.corrections = logprob_corrections
        assert len(self.corrections) == act_dim
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_layer = nn.Linear(hidden_sizes[-1], pow(act_dim, dim_rand))
        
    def _distribution(self, obs):
        mu = self.mu_layer(self.base_net(obs))
        log_std = self.log_layer(self.base_net(obs))
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Distribution
        logp_pi = pi.log_prob(act).sum(axis=-1)
        
        # Move action space dimension to front
        act = torch.squeeze(torch.transpose(torch.unsqueeze(act, 0), 0, -1), -1)

        # Make coordinate-wise adjustment to log prob
        logp_pi += sum([corr(a) for a, corr in zip([x for x in act], self.corrections)])
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
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]


        # Gather action space limits, create lob prob correction funcs, use Gaussian Actor
        # if action space is Box
        if isinstance(action_space, Box):
            self.is_discrete = False
            err_str = "multidimensional action space shape not supported by MLPGaussianActor"
            assert len(action_space.shape) == 1, err_str
            # Use action space bounds to get coord-wise mappings from raw actor output
            # to action space, and corrections to log probabilities as a func of action
            self.action_limits = zip(action_space.low, action_space.high)
            low_open, high_open = lambda l: l == float('-inf'), lambda h: h == float('inf')
            self.action_maps = []
            correction_maps = []
            for low, high in self.action_limits:
                zero_func, inv_func = lambda x: 0, lambda x: -x
                above_func, below_func = lambda x: high - np.exp(x), lambda x: low + np.exp(x)
                squash_func = lambda x: low + (high - low) * (np.tanh(x) + 1) / 2
                closed_func = lambda x: 2 * (x + F.softplus(-2 * x)) - np.log(2 * (high - low))
                if low_open(low) and high_open(high):
                    self.action_maps += [nn.Identity]
                    correction_maps += [zero_func]
                elif low_open(low):
                    self.action_maps += [above_func]
                    correction_maps += [inv_func]
                elif high_open(high):
                    self.action_maps += [below_func]
                    correction_maps += [inv_func]
                else:
                    self.action_maps += [squash_func]
                    correction_maps += [closed_func]
    
            # Build Policy (pass in corrections to log probs as lambdas taking action)
            self.pi = MLPGaussianActor(obs_dim, 
                                       action_space.shape[0], 
                                       hidden_sizes, 
                                       activation, 
                                       correction_maps)
        
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
                a = [amap(el) for amap, el in zip(self.action_maps, a)]
            v = self.v(obs)
        return np.array(a), v.numpy(), logp_a.numpy()
        
    def act(self, obs):
        return self.step(obs, deterministic=True)[0]


##########
# Squash
##########





class SquashedGaussianMLPActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limits):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limits = act_limits

    def _distribution(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        logp_pi = pi.log_prob(act).sum(axis=-1)
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding 
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi -= (2*(np.log(2) - act - F.softplus(-2*act))).sum(axis=-1)
        return logp_pi    # Last axis sum needed for Torch Normal distribution


    def old_forward(self, obs, deterministic=False, with_logprob=True):
        # Pre-squash distribution and sample
        pi_distribution = self._distribution(obs)
        pi_action = mu if deterministic else pi_distribution.rsample()
        logp_pi = pi._log_prob_from_distribution(pi_action) if with_logprob else None
        pi_action = torch.tanh(pi_action)
        delta = (self.act_limits[1] - self.act_limits[0]) / 2
        pi_action = self.act_limits[0] + delta * (pi_action + 1)
        return pi_action, logp_pi


class MLPSquashActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limits = (action_space.low, action_space.high)


        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, 
                                           act_dim, 
                                           hidden_sizes, 
                                           activation, 
                                           self.act_limits)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def act(self, obs):
        return self.step(obs)[0]

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.rsample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            delta = (self.act_limits[0] - self.act_limits[1]) / 2
            a = self.act_limits[0] + delta * (torch.tanh(a) + 1)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()
        
##########
# Squash
########## 
