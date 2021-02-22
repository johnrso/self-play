import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
import gym
from gym.spaces import Discrete, Box
import time
import core as core
import torch.nn as nn

from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import dmc2gym
from dmc2gym.wrappers import DMCWrapper
import os
import wandb


# tmp
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'

class ReplayBuffer:
    """
    A buffer for storing (s, a, s'. r) rewards experienced by the DQN agent.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.obs_prime_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
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

    def resize_buffer(self):
        if self.ptr >= self.max_size:
            new_obs_buf = np.zeros(core.combined_shape(self.max_size, obs_dim), dtype=np.float32)
            self.obs_buf = np.concatenate(self.obs_buf, new_obs_buf, axis = 0)
            new_act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
            self.act_buf = np.concatenate(self.act_buf, new_act_buf, axis = 0)
            new_obs_prime_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
            self.obs_prime_buf = np.concatenate(self.obs_prime_buf, new_obs_prime_buf, axis = 0)




    def get(self, n_t = 100):
        """
        Call this to get a dictionary of (s, a, s', r) tuples.

        return:
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

default_ac_kwargs = {
    "activation": nn.Tanh,
    # "dim_rand": 1,
}
