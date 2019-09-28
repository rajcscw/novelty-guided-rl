
# Goals
- Perform experiments on Minigrid tasks with sparse rewards
- If we can work out these experiments, we can definitely get accepted
- Specific goals: Be better than classic novelty search and be better than plain RL and PPO

- Can we also refine the mujoco based experiments?

# Tasks

**Fix the baseline parameters and run the baseline experiments only one**
**Add notes which one to change and which one not to...**

- Phase I: Run the baselines for lava and simple cross
- Phase II: Run the baselines for minigrid and unlock environments
- Crucial Phase III: Start tuning for NSRA encoders for all experiments
- Run preferably parallely for many tasks
- We need atleast six/eight experiments
- May be 2 simple crossing, 2 lava crossing, 2 multi room settings or some unlock the key experiments  
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
|  MiniGrid-SimpleCrossingS9N1-v0  | Running once| Running once  | PPO target of 0.84, RL target of 0.65, NSRA close to 0.8  |
|  MiniGrid-SimpleCrossingS9N2-v0  | Yet to start  | Yet to start  |   |
|  MiniGrid-SimpleCrossingS9N3-v0  | Yet to start  | Testing  | This seems promising because PPO fails completely, just have to tune NSRA, and RL target of 0.6, KNN target of 0.65 |
|  MiniGrid-SimpleCrossingS11N5-v0 | Yet to start  | Yet to start  |   |
|  MiniGrid-LavaCrossingS9N1-v0    | Yet to start  | Testing  |  RL target of 0.3 only. KNN target of 0.6 |
|  MiniGrid-LavaCrossingS9N2-v0    | Yet to start  | Yet to start  |   |
|  MiniGrid-LavaCrossingS9N3-v0    | Yet to start  | Yet to start  |   |
|  MiniGrid-LavaCrossingS11N5-v0   | Yet to start  | Yet to start  |   |

