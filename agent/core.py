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


class InputNormalizer(nn.Module):
    """
    Keeps track of the mean and (unbiased) standard deviation of all observations
    seen so far, and normalizes inputs according to these statistics. Let
        t be the sample size,
        N be the number of inputs we have seen including this sample,
        prev_mean be the mean of inputs we have previously seen,
        samp_mean be the mean of sample inputs,
        prev_std be the unbiased standard deviation of inputs we have previously seen, and
        samp_std be the biased standard deviation of inputs we have previously seen.
    Then the new mean is given by
        ((N - t) * prev_mean + t * samp_mean) / N
    and the new standard deviation can be derived as
        (N - t - 1) / (N - 1) * prev_std^2 + t / (N - 1) * samp_std^2
        + t (N - t) / (N (N - 1)) * (prev_mean - samp_mean)^2
    
    Edge cases:
    
    If N - t == 0, and we have seen no data so far, then neither prev_mean nor prev_std
    exist, so we must instead set self.mean and self.std to the mean and std of the sample.
    (The latter is only set if the batch size is > 1, since otherwise unbiased std is
    undefined. We only normalize obs before returning if the batch size is > 1.)

    If N - t == 1, we have a defined prev_mean but not a defined prev_std (because we store
    the unbiased std), but in this case the coefficient on prev_std^2 becomes zero in the 
    formula above, so we can simply leave out this term unless N - t > 1.
    """
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
        self.count = 0

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch.unsqueeze(obs, 0)
        t = obs.shape[0]                                # batch size
        self.count = N = self.count + t                 # new size
        prev_mean, samp_mean = self.mean, obs.mean(axis=0)
        prev_std, samp_std = self.std, obs.std(axis=0, unbiased=False)
        if N - t == 0:                                  # if we've never seen any data,
            self.mean = obs.mean(axis=0)                # make mean and std those of sample,
            if N > 1:                                   # only using samp_std if it exists
                self.std = obs.std(axis=0, unbiased=True)
                return (obs - self.mean) / self.std
            return obs
        else:                                           # update rule when we've seen data
            self.mean = ((N - t) * prev_mean + t * samp_mean) / N
            self.std = t / (N - 1) * samp_std * samp_std
            if N - t > 1:                               # only include prev_std if it exists
                self.std += (N - t - 1) / (N - 1) * prev_std * prev_std
            diff = (prev_mean - samp_mean) * (prev_mean - samp_mean)
            self.std = torch.sqrt(self.std + t * (N - t) / (N * (N - 1)) * diff)
        return (obs - self.mean) / self.std


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


def mlp(sizes, activation, output_activation=nn.Identity, input_norm=False):
    layers = [InputNormalizer()] if input_norm else []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def get_action_bound(action_space):
    """
    Gets the bound for closed action spaces, returns None for open action spaces.
    Assumption: if action_space is a closed domain, it is [-a, a]^n for some a
    Assumption: if action_space is an open domain, it is [-inf, inf]^n (no closed dims)
    Assumption: action_space is a continuous Box domain
    """
    bounded = (action_space.low[0] != float('-inf'))
    if bounded:
        err_str = "closed action spaces must be [-a, a]^n"
        assert all([low == action_space.low[0] for low in action_space.low]), err_str
        assert all([high == action_space.low[0] for high in action_space.high]), err_str
        return action_space.low[0]
    else:
        err_str = "open action spaces must be open on all dimensions"
        assert all([low == float('-inf') for low in action_space.low]), err_str
        assert all([high == float('inf') for high in action_space.high]), err_str
        return None


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
            bound = get_action_bounds(action_space)

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
                                       input_norm=pi_input_norm
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
        return np.array(a), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs, deterministic=True)[0]
