import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
import gym
from gym.spaces import Discrete, Box
import time
import simple_core as core #important
import buffer
from buffer import buffer

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

def sac(env_fn,
        Q_fn,
        V_fn,
        pi_fn,
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        target_kl=0.01,
        clip_kl=False,
        entropy_reg=False,
        logger_kwargs=dict(),
        save_freq=25,
        sweep=True,
        video=False,
        domain_name='cartpole',
        task_name='balance'):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Instantiate environment
    env = env_fn()
    eval_env = env_fn()
    env_name = env.unwrapped.spec.id
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    #initialize networks
    q1 = MLPQFunction()


    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = Buffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing SAC policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp - logp_old).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        # Add distribution entropy to loss
        if entropy_reg:
            loss_pi -= 0.01 * ent

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    pi_decay = .97
    pi_scheduler = lr_scheduler.ExponentialLR(optimizer=pi_optimizer, gamma=pi_decay)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if clip_kl and kl > 2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for j in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len,
                          flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model & evaluate
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            if proc_id()==0:
                # eval rollout
                o, eval_ep_ret, _ = eval_env.reset(), 0, 0
                frames = []
                print('Evaluating and generating video')
                for i in range(max_ep_len):
                    a = ac.act(torch.as_tensor(o, dtype=torch.float32))
                    next_o, r, d, _ = eval_env.step(a)
                    o = next_o
                    eval_ep_ret += r
                    kwargs = dict()
                    if isinstance(eval_env, DMCWrapper):
                        kwargs['width'] = 256
                        kwargs['height'] = 256
                    img = eval_env.render(mode='rgb_array', **kwargs)
                    if args.video:
                        frames.append(img)

                if args.video:
                    print("logging video")
                    # log video frames
                    video = np.transpose(np.array(frames),(0,3,1,2))[::4,...]
                    wandb.log({"video": wandb.Video(video, fps=30, format="gif")},
                              step=epoch)

                # log ep reward
                wandb.log({'eval_returns':eval_ep_ret},step=epoch)

if __name__ == '__main__':
