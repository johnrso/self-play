# a simple test using spinningup's ppo implementation out of box on a dm control environment
import argparse

from dm_control import suite
import core

import numpy as np
import spinup

import gym
import os

def main(env):
    spinup.ppo_pytorch(env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--domain_name', type=str, default='quadruped')
    parser.add_argument('--task_name', type=str, default='run')

    # hyper params
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--clip_ratio', type=float, default=0.2)
    # parser.add_argument('--pi_lr', type=float, default=.0003)
    # parser.add_argument('--vf_lr', type=float, default=0.001)
    # parser.add_argument('--lam', type=float, default=0.97)
    # parser.add_argument('--target_kl', type=float, default=0.01)

    args = parser.parse_args()

    env = suite.load(domain_name=parser.domain_name, task_name=args.task_name)
    core.AC
    main()
