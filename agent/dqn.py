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
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, obs_prime, rew):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.resize_buffer()
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.obs_prime_buf[self.ptr] = obs_prime
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def resize_buffer(self):
        if self.ptr >= self.max_size:
            new_obs_buf = np.zeros(core.combined_shape(self.max_size, obs_dim), dtype=np.float32)
            self.obs_buf = np.concatenate(self.obs_buf, new_obs_buf, axis = 0)

            new_act_buf = np.zeros(core.combined_shape(self.max_size, act_dim), dtype=np.float32)
            self.act_buf = np.concatenate(self.act_buf, new_act_buf, axis = 0)

            new_obs_prime_buf = np.zeros(core.combined_shape(self.max_size, obs_dim), dtype=np.float32)
            self.obs_prime_buf = np.concatenate(self.obs_prime_buf, new_obs_prime_buf, axis = 0)

            new_rew_buf = np.zeros(self.max_size, dtype=np.float32)
            self.rew_buf = np.concatenate(self.rew_buf, new_rew_buf, axis = 0)

            self.max_size *= 2

    def get(self, n_t = 100):
        """
        Call this to get a dictionary of (s, a, s', r) tuples.

        return:
        """
        indices = np.random.randint(low = 0, high = self.ptr, shape = min(self.ptr, n_t)

        data = dict(obs=self.obs_buf[indices],
                    act=self.act_buf[indices],
                    obs_prime = self.obs_prime_buf[indices],
                    ret=self.ret_buf[indices])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

default_ac_kwargs = {
    "activation": nn.Tanh,
    # "dim_rand": 1,
}

def ppo(env_fn,
        val_net=core.MLPCritic,
        hidden_sizes=(64,64),
        ac_kwargs=default_ac_kwargs,
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
        logger_kwargs=dict(),
        save_freq=25,
        sweep=True,
        video=False,
        domain_name='cartpole',
        task_name='balance'):
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on approximate KL
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        hidden_sizes (tuple): A tuple of hidden sizes of the policy net.
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    if proc_id() == 0:
        hyperparameter_defaults = dict(gamma=gamma,
                                       clip_ratio=clip_ratio,
                                       pi_lr=pi_lr,
                                       vf_lr=vf_lr,
                                       lam=lam,
                                       target_kl=target_kl)

        if args.sweep:

            wandb.init(config=hyperparameter_defaults,
                       project='ppo-hyperparameter-sweep',
                       entity='self-play-project')
            config = wandb.config
            wandb.run.name = "{}_{}_".format(domain_name, task_name) + wandb.run.name
            # sweep params
            gamma=config.gamma
            clip_ratio=config.clip_ratio
            pi_lr=config.pi_lr
            vf_lr=config.vf_lr
            lam=config.lam
            target_kl=config.target_kl
        else:
            wandb.init(config=hyperparameter_defaults,project="ppo")

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    eval_env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space,
                      env.action_space,
                      hidden_sizes=hidden_sizes,
                      **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        #ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        #pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        pi_info = dict(kl=approx_kl, ent=0, cf=clipfrac)

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
            if kl > 2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
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

        # Perform PPO update!
        update()
        pi_scheduler.step()
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

def parse_boolean(arg):
    arg = str(arg).upper()
    if 'TRUE'.startswith(arg):
        return True
    elif 'FALSE'.startswith(arg):
        return False
    else:
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--domain_name', type=str, default='quadruped')
    parser.add_argument('--task_name', type=str, default='run')
    parser.add_argument('--dmc', type=parse_boolean, default=True)
    parser.add_argument('--sweep', type=parse_boolean, default=True)
    parser.add_argument('--video', type=parse_boolean, default=True)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--exp_name', type=str, default='ppo')

    # hyper params
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--pi_lr', type=float, default=.0003)
    parser.add_argument('--vf_lr', type=float, default=0.001)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--target_kl', type=float, default=0.01)

    args = parser.parse_args()
    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    if args.dmc:
        ppo(lambda : dmc2gym.make(domain_name=args.domain_name,
                                  task_name=args.task_name,
                                  seed=args.seed),
            actor_critic=core.MLPActorCritic,
            hidden_sizes=[args.hid]*args.l,
            ac_kwargs=default_ac_kwargs,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            pi_lr=args.pi_lr,
            vf_lr=args.vf_lr,
            lam=args.lam,
            target_kl=args.target_kl,
            seed=args.seed,
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            logger_kwargs=logger_kwargs,
            sweep=args.sweep,
            video=args.video,
            domain_name=args.domain_name,
            task_name=args.task_name)
    else:
        print("using gym env")
        ppo(lambda : gym.make(args.env),
            actor_critic=core.MLPActorCritic,
            hidden_sizes=[args.hid]*args.l,
            ac_kwargs=default_ac_kwargs,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            pi_lr=args.pi_lr,
            vf_lr=args.vf_lr,
            lam=args.lam,
            target_kl=args.target_kl,
            seed=args.seed,
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            logger_kwargs=logger_kwargs,
            sweep=args.sweep,
            video=args.video,
            domain_name=args.domain_name,
            task_name=args.task_name)
