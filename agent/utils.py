import numpy as np
import scipy.signal

import torch
import torch.nn as nn

###############################################################################
#                                                                             #
#                              General Utils                                  #
#                Taken from SpinningUp's VPG Implementation                   #
#                                                                             #
###############################################################################

def mlp(sizes, act, output_act=nn.Identity):
    """
    A generalized MLP constructor used to construct the neural nets which our
    VPG will train to predict state values and optimize policies.
    """
    layers = []
    for j in range(len(sizes) - 1):
        activation = act if j < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[j], sizes[j+1], activation()))
    return nn.Sequential(*layers)

def num_params(module):
    """
    Gets the total number of parameters of a module.
    """
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_sum(x, discount):
    """
    Some scipy magic for performing a discounted sum over an array.

    input: [x0, x1, x2]
    output: [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

##############################################################################
#                                                                            #
#                        Actor-Critic Implementations                        #
#     Taken from SpinningUp's VPG Implementation and modified for PPO        #
#                                                                            #
##############################################################################

class Actor(nn.Module):

    def _policy(self, obs):
        raise NotImplementedError

    def _log_prob(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._policy(obs)
        logp = None
        if act is not None:
            logp = self._log_prob(pi, act)
        return pi, logp

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _policy(self, obs):
        return Categorical(logits=self.logits(obs))

    def _log_prob(self, pi, act):
        return pi.log_prob(act)
        
class MLPGaussianActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.logits = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _policy(self, obs):
        mean = self.logits(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class MLPCritic(nn.Module):
    
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.value = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        
    def foward(self, obs):
        return torch.squeeze(self.value(obs), -1)

class MLPActorCritic(nn.Module):

    def __init__(self, obs_space, act_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

