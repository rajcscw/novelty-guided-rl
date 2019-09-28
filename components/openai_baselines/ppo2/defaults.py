def mujoco():
    return dict(
        nEps=50,
        nItrs= 1000,
        nbatch_train = 500,
        log_every = 50,
        ent_coef=0.0,
        lr=1e-2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        noptepochs=5,
        cliprange=0.2,
        nsteps=2048, # this will not be used
        #nminibatches=32, # this will not be used
        #log_interval=1, # this will not be used
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
    )

def retro():
    return atari()
