import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
import gym
from gym.spaces import Discrete, Box
import time
import core
import random
import torch.nn as nn
from torch.distributions import Normal

from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from util import parse_boolean, parse_std_source, parse_activation, parse_metric
from util import PPOBuffer
from util import disable_view_window
import dmc2gym
from dmc2gym.wrappers import DMCWrapper
import os
import wandb
from enum import Enum

# tmp
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'


def ppo(env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs={},
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
        stop_metric=None,
        stop_cutoff=0.01,
        reg_metric=None,
        reg_coeff=0.01,
        logger_kwargs=dict(),
        save_freq=25,
        video=True):
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on one of several KL metrics or entropy
    and regularization based on one of these metrics
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
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO. Specifically contains:
                activation (nn.Module): An activation to use in policy/value network
                    hidden layers.
                std_dim (int): The dimension of the standard deviation used in policy
                    network outputs. Either 0 (isotropic Gaussian) or 1 (anisotropic 
                    Gaussian, but with a diagonal covariance matrix).
                std_source (bool): Represents the source of the values of the variances
                    used in policy network output distributions. If True, predict
                    logarithms of standard deviations from network, if False, store
                    logarithms as parameters, and if None, use constant standard deviation.
                std_value (float): Standard deviation to use as constant if std_source is None.
                squash (bool): Whether or not to squash actions drawn from distribution
                    with tanh. If false but action space is still bounded, actions are clipped.
                squash_mean (bool): If clipping actions (because action space is bounded but
                    squash is False), this determines whether to add tanh layer to mean network
                    to squash the mean of the action distribution into the action space bounds.
                pi_width (int): Width of policy network hidden layers.
                pi_depth (int): Number of policy network hidden layers.
                vf_width (int): Width of value network hidden layers.
                vf_depth (int): Number of value network hidden layers.
                pi_weight_ratio (float): Ratio of initiliazation values of last layer of
                    policy network to the default values. Used to lower initial policy variance.
                pi_input_norm (bool): Whether to apply batch norm to inputs to the policy 
                    network.
                vf_input_norm (bool): Whether to apply batch norm to inputs to the value 
                    network.
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
        stop_metric (ppo.Metric): An enum value representing the metric we use
            to decide early stopping. See ppo.Metric for more information about options.
        stop_cutoff (float): A cutoff value for the stop_metric. If the stop_metric
            exceeds this cutoff value, we stop the current epoch.
        reg_metric (ppo.Metric): An enum value representing the metric we use for
            regularizaiton. See ppo.Metric for more information about options.
        reg_coeff (float): The coefficient of the regularization term in the loss.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        video (bool): Whether or not to record and save video of model evals.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Instantiate environment
    env = env_fn()
    eval_env = env_fn()
    #disable_view_window()
    env_name = env.unwrapped.spec.id
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    if proc_id() == 0:
        hyperparameter_defaults = dict(gamma=gamma,
                                       clip_ratio=clip_ratio,
                                       pi_lr=pi_lr,
                                       vf_lr=vf_lr,
                                       lam=lam,
                                       stop_metric=stop_metric,
                                       stop_cutoff=stop_cutoff,
                                       reg_metric=reg_metric,
                                       reg_coeff=reg_coeff)

        wandb.init(config=hyperparameter_defaults,
                   project='ppo-hyperparameter-sweep',
                   entity='self-play-project')
        config = wandb.config

        wandb.run.name = "{}_".format(env_name) + wandb.run.name


    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())
    logger.log("\nName of environment: {}\n".format(env_name))

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create actor-critic module
    ac = actor_critic(env.observation_space,
                      env.action_space,
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
        ref_mean, ref_std = torch.zeros(act.shape[-1]), torch.ones(act.shape[-1])
        ref_logp = Normal(ref_mean, ref_std).log_prob(act).sum(axis=-1)
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(cf = clipfrac)
        pi_info['entropy'] = pi.entropy().mean()
        pi_info['kl'] = (logp - logp_old).mean()
        pi_info['rev_kl'] = (torch.exp(logp_old / logp) * (logp_old - logp)).mean()
        pi_info['ref_kl'] = (logp - ref_logp).mean()
        
        # Regularize
        if reg_metric:
            sign = -1 if reg_metric == 'entropy' else 1
            loss_pi += sign * reg_coeff * pi_info[reg_metric]

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
            c = -1 if stop_metric is 'entropy' else 1 # reverse comparison order for entropy
            if stop_metric and c * mpi_avg(pi_info[stop_metric].detach()) > c * stop_cutoff:
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
        logger.store(LossPi=pi_l_old, LossV=v_l_old, ClipFrac=pi_info['cf'],
                     KL=mpi_avg(pi_info['kl'].item()), 
                     Entropy=mpi_avg(pi_info['entropy'].item()),
                     Reverse_KL=mpi_avg(pi_info['rev_kl'].item()), 
                     Reference_KL=mpi_avg(pi_info['ref_kl'].item()),
                     DeltaLossPi=mpi_avg(loss_pi.item() - pi_l_old),
                     DeltaLossV=mpi_avg(loss_v.item() - v_l_old))

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
                # log training returns
                wandb.log({'train_returns': logger.get_stats('EpRet')[0]}, 
                          step=epoch)

                # eval rollout
                o, eval_ep_ret, _ = eval_env.reset(), 0, 0
                frames = []
                print('Evaluating and generating video')
                for i in range(max_ep_len):
                    a = ac.act(torch.as_tensor(o, dtype=torch.float32))
                    o, r, d, _ = eval_env.step(a)
                    eval_ep_ret += r
                    kwargs = dict()
                    if isinstance(eval_env, DMCWrapper):
                        kwargs['width'] = 512
                        kwargs['height'] = 512
                    img = eval_env.render(mode='rgb_array', **kwargs)
                    if video:
                        frames.append(img)
                    if d:
                        break
                if video:
                    print("logging video")
                    # log video frames
                    vid_frames = np.transpose(np.array(frames),(0,3,1,2))[::4,...]
                    wandb.log({"video": wandb.Video(vid_frames, fps=30, format="gif")},
                              step=epoch)

                # log eval reward
                wandb.log({'eval_returns':eval_ep_ret}, step=epoch)

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
        logger.log_tabular('Reverse_KL', average_only=True)
        logger.log_tabular('Reference_KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()



if __name__ == '__main__':
    """
    Runs PPO on an OpenAI Gym or DeepMind Control environment (the latter handled
    through a dmc2gym wrapper).
    Program args:
        env (str): Name of the OpenAI Gym environment to use, or the domain and task name of
            the DeepMind Control environment to use separated by a space. e.g. "Cartpole-v1"
            or "quadruped run". Only used if not sweeping, for otherwise bayes hyperparam
            optimization will only select environments we perform well on.
        env_list (str): Names of the OpenAI Gym environments and DeepMind Control environments
            from which to select randomly when sweeping.
        sweep (bool): Whether to treat this run as a wandb sweep. If True, environment is
                      sampled randomly from lists `gym_env_list` and `dmc_env_list` in the
                      wandb config. If False, we get environment from command line options.
                      This is to avoid wandb bayes search from only choosing environments
                      on which the agent succeeds.
        video (bool): Whether to save rendered videos of our agent at test time.

        The following model hyperparameters and training hyperparameters have identical
        definitions as defined in the docstring for ppo:
        activation (nn.Module), pi_width (int), pi_depth (int), vf_width (int), 
        vf_depth (int), pi_weight_ratio (float), pi_input_norm (bool), vf_input_norm (bool),
        std_dim (int), std_source (bool), std_value (float), squash (bool), squash_mean (bool),
        seed (int), steps_per_epoch (int), epochs (int), gamma (float), clip_ratio (float), 
        pi_lr (float), vf_lr (float), train_pi_iters (int), train_v_iters (int), lam (float),
        max_ep_len (int)
        
        stop_metric (str): A string parsed into an enum value representing the metric we use
            to decide early stopping. See ppo.Metric for more information about options.
        min_entropy (float): If stop_metric is ppo.Metric.ENTROPY, we stop episodes when
            entropy dips below this cutoff value.
        max_kl (float): If stop_metric is ppo.Metric.KL_DIV, we stop episodes when KL
            divergence (that is, D_KL(pi_old || pi)) exceeds this cutoff value.
        max_rev_kl (float): If stop_metric is ppo.Metric.REVERSE_KL, we stop episodes when
            reverse KL (that is, D_KL(pi || pi_old)) exceeds this cutoff value.
        max_ref_kl (float): If stop_metric is ppo.Metric.REF_KL, we stop episodes when
            the policy's KL against a reference N(0, 1) distribution exceeds this cutoff.
        
        reg_metric (str): A string parsed into an enum value representing the metric we use
            to regularize the model. See ppo.Metric for more information about options.
            The corresponding metric is multiplied by a coefficient and subtracted from the
            loss (or added in the case of entropy).
        entropy_coeff (float): If reg_metric is ppo.Metric.ENTROPY, we use this float as
            the coefficient of the entropy bonus in our loss.
        kl_coeff (float): If reg_metric is ppo.Metric.KL_DIV, we use this float as the 
            coefficient of the KL penalty (D_KL(pi_old || pi)) in our loss.
        rev_kl_coeff (float): If reg_metric is ppo.Metric.REV_KL, we use this float as the
            coefficient of the reverse KL penalty (D_KL(pi || pi_old)) in our loss.
        ref_kl_coeff (float): If reg_metric is ppo.Metric.REF_KL, we use this float as the
            coefficient of the KL penalty (against a reference N(0, 1) distribution) in
            our loss.
    """
    import argparse
    parser = argparse.ArgumentParser()
    
    # Run params
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--env_list', type=str, default=None, nargs='+')
    parser.add_argument('--sweep', type=parse_boolean, default=False)
    parser.add_argument('--video', type=parse_boolean, default=True)
    parser.add_argument('--exp_name', type=str, default='ppo')
    
    # Network architecture params
    parser.add_argument('--activation', type=parse_activation, default=nn.Tanh)
    parser.add_argument('--pi_width', type=int, default=64)
    parser.add_argument('--pi_depth', type=int, default=2)
    parser.add_argument('--vf_width', type=int, default=64)
    parser.add_argument('--vf_depth', type=int, default=2)
    parser.add_argument('--pi_weight_ratio', type=float, default=0.01)
    parser.add_argument('--pi_input_norm', type=parse_boolean, default=False)
    parser.add_argument('--vf_input_norm', type=parse_boolean, default=False)

    # Training stochasticity params
    parser.add_argument('--std_dim', type=int, default=1)
    parser.add_argument('--std_value', type=float, default=0.5)
    parser.add_argument('--std_source', type=parse_std_source, default=None)

    # Training loop hyperparams
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=75)

    # Model hyperparams
    parser.add_argument('--squash', type=parse_boolean, default=True)
    parser.add_argument('--squash_mean', type=parse_boolean, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--pi_lr', type=float, default=.0003)
    parser.add_argument('--vf_lr', type=float, default=0.001)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--max_ep_len', type=int, default=1000)

    # Epoch clipping using KL / Entropy as metric
    parser.add_argument('--stop_metric', type=parse_metric, default=None)
    parser.add_argument('--min_entropy', type=float, default=0.01)
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--max_rev_kl', type=float, default=0.01)
    parser.add_argument('--max_ref_kl', type=float, default=0.01)

    # Regularization with KL / Entropy bonuses
    parser.add_argument('--reg_metric', type=parse_metric, default=None)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kl_coeff', type=float, default=0.01)
    parser.add_argument('--rev_kl_coeff', type=float, default=0.01)
    parser.add_argument('--ref_kl_coeff', type=float, default=0.01)

    args = parser.parse_args()

    # Setup actor-critic kwargs
    ac_kwargs = dict()
    ac_kwargs['activation'] = args.activation
    ac_kwargs['pi_width'] = args.pi_width
    ac_kwargs['pi_depth'] = args.pi_depth
    ac_kwargs['vf_width'] = args.vf_width
    ac_kwargs['vf_depth'] = args.vf_depth
    ac_kwargs['pi_weight_ratio'] = args.pi_weight_ratio
    ac_kwargs['pi_input_norm'] = args.pi_input_norm
    ac_kwargs['vf_input_norm'] = args.vf_input_norm
    ac_kwargs['std_dim'] = args.std_dim
    ac_kwargs['std_source'] = args.std_source
    ac_kwargs['std_value'] = args.std_value
    ac_kwargs['squash'] = args.squash
    
    # Get name of environment
    random.seed(args.seed)
    env_str = (random.choice(args.env_list) if args.sweep else args.env).split()

    # Process early stop/regularization parameters
    get_objective = lambda metric: "min_" if metric == "entropy" else "max_"
    get_cutoff = lambda metric: getattr(args, get_objective(metric) + metric)
    get_coeff = lambda metric: getattr(args, metric + "_coeff")
    stop_cutoff = None if args.stop_metric is None else get_cutoff(args.stop_metric)
    reg_coeff = None if args.reg_metric is None else get_coeff(args.reg_metric)
    
    # Setup logger
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    mpi_fork(args.cpu)  # run parallel code with mpi
    
    # Instantiate environment
    if len(env_str) == 2:
        env_fn = lambda : dmc2gym.make(domain_name=env_str[0],
                                       task_name=env_str[1],
                                       seed=args.seed)
    else:
        env_fn = lambda : gym.make(env_str[0])

    ppo(env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        lam=args.lam,
        max_ep_len=args.max_ep_len,
        stop_metric=args.stop_metric,
        stop_cutoff=stop_cutoff,
        reg_metric=args.reg_metric,
        reg_coeff=reg_coeff,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        video=args.video)
