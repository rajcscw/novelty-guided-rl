
# Goals
- Perform experiments on Minigrid tasks with sparse rewards
- If we can work out these experiments, we can definitely get accepted
- Specific goals: Be better than classic novelty search and be better than plain RL and PPO

- Can we also refine the mujoco based experiments?

# Tasks

**Fix the baseline parameters and run the baseline experiments only once**
**Add notes which one to change and which one not to...** 

- Phase I: Run the baselines for lava and simple cross (Will finish before Sep 30) 

In general, in simple cross environments, KNN do not show much improvements, but let's check with AE approach one
In lava experiments, it shows good improvements over RL. PPO is unstable in most.

Baselines done for Lava experiments; Now, running with AE methods.

- Phase II: Run the baselines for minigrid and unlock environments

Minigrid is running now, Next unlock to be started

- Collect numbers to beat - DONE 
- Crucial Phase III: Start tuning for NSRA encoders for all experiments - Doing now
- Pick atleast six/eight experiments
- We need atleast six/eight experiments
- **Definitely** Must include Multi room experiments where pure novelty failed
- Merge all pure novelty and adaptive novelty experiments into the same plot
- We need some ablation analysis experiments such as sequence length, model complexity etc on some experiment
- Would be cool to plot some behavior representations (more plots, higher is the chance of getting accepted)

# Open questions:

##1. Can we just add the behavior of the updated model for learning instead of all the learning? 
In that case, may be just need to run for many epochs...
I think this is more close for classic novelty search methods

This is not working and it is very unstable..So, going with the previous approach of using

##2. In case of mini grid experiments, we can also limit the episode steps to very few so that we can see the difference with novelty-search methods?

This makes sense. We can simulate every such experiment like robot falling type so that it times out and a novel policy will aways be the one that goes far.

##3. If time permits, run the experiment for Mujoco again for selected...

|Experiment| Novelty  | Adaptive  | Status |
|---|---|---|---|
|  MiniGrid-LavaCrossingS9N1-v0    | Not needed  | Baselines collected, tuning AE    |  Target of 0.8 (All of them)  |
|  MiniGrid-LavaCrossingS9N2-v0    | Not needed  | Baselines collected, tuning AE    |  Target of  0.6 (PPO)|
|  MiniGrid-LavaCrossingS9N3-v0    | Not needed  | Baselines collected, tuning AE    |  Target of 0.6 (PPO)  |
|  MiniGrid-LavaCrossingS11N5-v0   | Not needed  | Baselines collected, tuning AE    |  Target of  0.35 (KNN)|
|  MiniGrid-MultiRoom-N4-S5-v0     | Not needed  | Baselines collected, tuning AE    |  Target of  0.8 (All of them)|
|  MiniGrid-MultiRoom-N6-v0        | Not needed  | Baselines collected, tuning AE    |  Target of  0.2 (KNN)|
|  MiniGrid-Unlock-v0              | Not needed  | Baselines collected, tuning AE    |  Target of  0.9 (PPO)|
|  MiniGrid-UnlockPickup-v0        | Not needed  | Baselines collected, tuning AE    |  Target of  0.2 (KNN)|
|  MiniGrid-BlockedUnlockPickup-v0 | Not needed  | Baselines collected, tuning AE    |  Target of  0.2 (KNN)|



