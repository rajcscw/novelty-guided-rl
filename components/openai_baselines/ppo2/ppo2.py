import time
import numpy as np
from collections import deque
from components.openai_baselines.common import set_global_seeds
from components.openai_baselines.common.policies import build_policy
from functools import reduce
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from components.openai_baselines.ppo2.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, network, env, seed=None, nsteps, nEps, nItrs, nbatch_train, log_every, ent_coef, lr,
            vf_coef,  max_grad_norm, gamma, lam, noptepochs, cliprange,
          load_path=None, model_fn=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Fix the batch size to be 500
    nbatch_train = nbatch_train

    # and fix the noptepochs at each epoch to be 1 for now
    noptepochs = noptepochs

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from components.openai_baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    # UPDATE 1: for each iteration, collect a sample of experience data
    # For us, this is equivalent to number of iterations
    running = 0
    episodic_total_reward = []
    for iter in range(1, nItrs+1):
        # assert nbatch % nminibatches == 0  (not necessary)
        # Start timer
        tstart = time.time()
        frac = 1.0 - (iter - 1.0) / nItrs
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # UPDATE 2: Get minibatch of experiences (here, it is fixed to 2048 or something)
        # For us, this is equivalent to number of episodes
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)

        # batch of experiences is dynamic (depending on the number of pre-defined episodes)
        nbatch = obs.shape[0] - (obs.shape[0] % nbatch_train)

        # UPDATE 3: we have collected the experiences (could be more than 2048)
        # but we have to do a mini-batch of updates for certain number of epochs
        # this is straightforward
        # fix a batch size and in each epoch, iterate over the entire data set
        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        # compute the mean reward at this iteration
        sum = reduce(lambda x, y: y["r"] + x, epinfos, 0)
        mean = sum / len(epinfos)

        # add to the log
        episodic_total_reward.append(mean)

        print("\r Processed iteration {} of {}".format(iter+1, nItrs), end="")
        running += mean
        if (iter+1) % log_every == 0:
            running = running / log_every
            print(", Episodic Reward {}".format(running))
            running = 0

    return model, episodic_total_reward

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



