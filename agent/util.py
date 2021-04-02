import numpy as np
import torch
import torch.nn as nn
import scipy
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


##################################################################
#                                                                #
#                              PARSERS                           #  
#                                                                #
##################################################################

def parse_boolean(arg):
    arg = str(arg).upper()
    if 'TRUE'.startswith(arg):
        return True
    elif 'FALSE'.startswith(arg):
        return False
    else:
        print("failed to parse boolean {}".format(arg))
        pass

def parse_std_source(arg):
    arg = str(arg).upper()
    if 'NETWORK'.startswith(arg):
        return True
    if 'PARAMETER'.startswith(arg):
        return False
    if 'CONSTANT'.startswith(arg):
        return None
    else:
        print("failed to parse std_source {}".format(arg))
        pass

def parse_activation(arg):
    arg = str(arg).upper()
    if 'TANH'.startswith(arg):
        return nn.Tanh
    if 'RELU'.startswith(arg):
        return nn.ReLU
    if 'SIGMOID'.startswith(arg):
        return nn.Sigmoid
    else:
        print("failed to parse activation {}".format(arg))
        pass

def parse_metric(arg):
    arg = str(arg).upper()
    if 'NONE'.startswith(arg):
        return None
    if 'ENTROPY'.startswith(arg):
        return 'entropy'
    if 'KL_DIVERGENCE'.startswith(arg) or \
       'KL DIVERGENCE'.startswith(arg):
        return 'kl'
    if 'REVERSE_KL_DIVERGENCE'.startswith(arg) or \
       'REVERSE KL DIVERGENCE'.startswith(arg) or \
       'REV_KL_DIVERGENCE'.startswith(arg) or \
       'REV KL DIVERGENCE'.startswith(arg):
        return 'rev_kl'
    if 'REFERENCE_KL_DIVERGENCE'.startswith(arg) or \
       'REFERENCE KL DIVERGENCE'.startswith(arg) or \
       'REF_KL_DIVERGENCE'.startswith(arg) or \
       'REF KL DIVERGENCE'.startswith(arg):
        return 'ref_kl'
    else:
        print("failed to parse metric {}".format(arg))
        pass


##################################################################
#                                                                #
#                           NETWORK TOOLS                        #  
#                                                                #
##################################################################

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
    def __init__(self, eps=1e-20):
        super().__init__()
        self.mean = None
        self.std = None
        self.count = 0
        self.eps = eps  # smoothing and dealing with std = 0

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
            else: 
                return obs
        else:                                           # update rule when we've seen data
            self.mean = ((N - t) * prev_mean + t * samp_mean) / N
            self.std = t / (N - 1) * samp_std * samp_std
            if N - t > 1:                               # only include prev_std if it exists
                self.std += (N - t - 1) / (N - 1) * prev_std * prev_std
            diff = (prev_mean - samp_mean) * (prev_mean - samp_mean)
            self.std = torch.sqrt(self.std + t * (N - t) / (N * (N - 1)) * diff)
        return (obs - self.mean) / (self.std + self.eps)

def mlp(sizes, activation, output_activation=nn.Identity, input_norm=False):
    layers = [InputNormalizer()] if input_norm else []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


##################################################################
#                                                                #
#                        ACTION SPACE BOUNDS                     #  
#                                                                #
##################################################################

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
        assert all([high == -action_space.low[0] for high in action_space.high]), err_str
        return action_space.high[0]
    else:
        err_str = "open action spaces must be open on all dimensions"
        assert all([low == float('-inf') for low in action_space.low]), err_str
        assert all([high == float('inf') for high in action_space.high]), err_str
        return None


##################################################################
#                                                                #
#                          BUFFERING TOOLS                       #  
#                                                                #
##################################################################

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

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

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


##################################################################
#                                                                #
#                          RENDERING TOOLS                       #  
#                                                                #
##################################################################

def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
