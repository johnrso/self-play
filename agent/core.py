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
from util import combined_shape, mlp, count_vars, get_action_bound, InputNormalizer



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



class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, input_norm=False):
        super().__init__()
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.logits_net = mlp(sizes, activation, input_norm=input_norm)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)



LOG_STD_MAX = 2
LOG_STD_MIN = -20



class MLPGaussianActor(Actor):
    """
    An actor for continuous action spaces.
    Args:
        obs_dim: Number of dimensions of observation space
        act_dim: Number of dimensions of action space
        hidden_sizes: List of hidden sizes for policy net
        activation: Nonlinearity to use in policy net
        std_dim: The dimension of the standard deviation variable that is stored.
            0 -> meaning same standard deviation on all action space dimensions
            1 -> meaning potentially differing standard deviation on action space dimensions
        std_source: Represents the source of the standard deviation.
            True  -> Predict standard deviation from base network
            False -> Store standard deviation as learned parameter
            None  -> Keep constant standard deviation
        std_value: The constant value to use for the standard deviation if std_source is None
        squash: Whether to squash action outputs by applying tanh. Used in logprob correction.
            Only active if action space is bounded, regardless of value passed in.
        squash_mean: Whether to squash mean action predicted by network by applying tanh.
            Only active if squash is inactive and action space is bounded, regardless of 
            value passed in.
        weight_ratio: Ratio with which to initialize final layer weights. Set low.
        input_norm: Whether to normalize inputs to the policy network (specifically, normalize
            them relative to previous inputs/observations).
        bound: The bound extracted from the action space. None if action space is unbounded.
    """
    class BoundMultiplier(nn.Module):
        """
        Tiny module used when we have a bounded space but are not squashing. In this case
        we add a Tanh layer to the end of the network that predicts the action distribution
        mean, which maps onto [-1, 1]^n, so we must multiply by a bound a to map to the correct
        action space [-a, a]^n.
        """
        def __init__(self, bound=1):
            super().__init__()
            self.bound = bound
        
        def forward(self, act):
            return self.bound * act
            

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 activation,
                 std_dim=1,
                 std_source=None,
                 std_value=0.5,
                 squash=True,
                 squash_mean=False,
                 weight_ratio=0.01,
                 input_norm=False,
                 bound=None):
        super().__init__()

        # Initialize parameters describing mapping to action space
        self.act_dim = act_dim
        self.squash = squash and bound is not None
        self.squash_mean = squash_mean and not squash and bound is not None
        self.bound = bound

        # Initialize Mean Network Architecture
        self.base_net = mlp([obs_dim] + list(hidden_sizes), activation, input_norm=input_norm)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.mu_layer.weight = nn.Parameter(weight_ratio * self.mu_layer.weight)
        if self.squash_mean:
            self.mu_layer = nn.Sequential(self.mu_layer, nn.Tanh(), BoundMultiplier(bound))
   
        # Initialize Variance Parameters / Network Architecture
        assert std_dim in [0, 1]
        if std_source is None:                       # std_source == None indicates constant
            self.log_std = torch.tensor(np.log(std_value))
        elif not std_source:                         # std_source == False indicates parameter
            self.log_std = nn.Parameter(np.log(std_value)*torch.ones(pow(act_dim, std_dim)))
        else:                                        # std_source == True indicates network
            self.log_layer = nn.Linear(hidden_sizes[-1], pow(act_dim, std_dim))
        self.std_dim = std_dim
        self.std_source = std_source

    def _distribution(self, obs):
        mu = self.mu_layer(self.base_net(obs))
        # use self.log_std if std_source is constant or parameter, use log_layer if network
        log_std = self.log_std if not self.std_source else self.log_layer(self.base_net(obs)) 
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Distribution
        logp_pi = pi.log_prob(act).sum(axis=-1)
        # Make adjustment to log prob if squashing
        if self.squash and self.bound is not None:
            logp_pi += (2 * (act + F.softplus(-2 * act)) - np.log(4 * self.bound)).sum(axis=-1)
        return logp_pi



class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, input_norm=False):
        super().__init__()
        sizes = [obs_dim] + list(hidden_sizes) + [1]
        self.v_net = mlp(sizes, activation, input_norm=input_norm)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):

    """
    The full learning agent.
    Args:
        observation_space: The OpenAI Gym space for observations
        action_space: The OpenAI Gym space for actions
        pi_width: The width of the policy net
        pi_depth: The depth of the policy net
        vf_width: The width of the value net
        vf_depth: The depth of the value net
        activation: Nonlinearity to use in policy net
        std_dim: The dimension of the standard deviation variable that is stored.
            0 -> meaning same standard deviation on all action space dimensions
            1 -> meaning potentially differing standard deviation on action space dimensions
        std_source: Represents the source of the standard deviation.
            True  -> Predict standard deviation from base network
            False -> Store standard deviation as learned parameter
            None  -> Keep constant standard deviation
        std_value: The constant value to use for the standard deviation if std_source is None
        squash: Whether to squash action outputs by applying tanh. Used in logprob correction.
            Only active if action space is bounded, regardless of value passed in.
        squash_mean: Whether to squash mean action predicted by network by applying tanh.
            Only active if squash is inactive and action space is bounded, regardless of 
            value passed in.
        pi_weight_ratio: Ratio with which to initialize policy net final layer weights
        pi_input_norm: Whether to normalize inputs to the policy network (specifically,
            normalize them relative to previous inputs/observations)
        vf_input_norm: Whether to normalize inputs to the policy network
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 pi_width=64,
                 pi_depth=2,
                 vf_width=64,
                 vf_depth=2,
                 activation=nn.Tanh,
                 std_dim=1,
                 std_source=False,
                 std_value=0.5,
                 squash=False,
                 squash_mean=False,
                 pi_weight_ratio=0.01,
                 pi_input_norm=False,
                 vf_input_norm=False):
        super().__init__()

        obs_dim = observation_space.shape[0]
        
        # Use a GaussianActor for continuous domains and a CategoricalActor for discrete
        # domains, only squashing if squash is True and domain is continuous
        if isinstance(action_space, Box):
            self.is_discrete = False
            err_str = "multidimensional action space shape not supported by MLPGaussianActor"
            assert len(action_space.shape) == 1, err_str

            # Get bound on action space dimensions             
            bound = get_action_bound(action_space)

            if squash and bound is not None:
                self.squash = lambda a: bound * torch.tanh(a)
            elif bound is not None:
                self.squash = lambda a: torch.clamp(a, -bound, bound)
            else:
                self.squash = lambda a: a
            
            # Build Policy (pass in corrections to log probs as lambdas taking action)
            self.pi = MLPGaussianActor(obs_dim,
                                       action_space.shape[0],
                                       [pi_width] * pi_depth,
                                       activation,
                                       std_dim=std_dim,
                                       std_value=std_value,
                                       std_source=std_source,
                                       squash=squash,
                                       squash_mean=squash_mean,
                                       weight_ratio=pi_weight_ratio,
                                       input_norm=pi_input_norm,
                                       bound=bound)

        # Use Categorical Actor if action space is Discrete
        elif isinstance(action_space, Discrete):
            self.is_discrete = True
            self.squash = lambda a: a
            self.pi = MLPCategoricalActor(obs_dim, 
                                          action_space.n, 
                                          [pi_width] * pi_depth, 
                                          activation,
                                          input_norm=pi_input_norm)

        # build value function
        self.v  = MLPCritic(obs_dim, [vf_width] * vf_depth, activation, vf_input_norm)

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
            # always squash since it is the identity if we don't want to squash
            a = self.squash(a)
            v = self.v(obs)
            # critical to ensure a has the right dimensions
            a = torch.squeeze(a)
            if not a.shape:
                a = torch.unsqueeze(a, axis=0)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs, deterministic=True)[0]
