import numpy as np


class NoveltyAdaptor:
    def __init__(self, rl_weight, rl_weight_delta, t_max):
        self.current_rl_weight = rl_weight
        self.rl_weight_delta = rl_weight_delta
        self.t_max = t_max

        # for performance tracking
        self.f_best = -np.inf
        self.t_best = 0

    def adapt(self, current_f):
        """
        Refer NSRA-ES algorithm from
        """
        # if the performance has improved, increase the reward pressure so that we don't explore
        # with novelty weights anymore
        if current_f > self.f_best:

            # increase the reward pressure
            self.current_rl_weight = min(1.0, self.current_rl_weight + self.rl_weight_delta)

            # reset the stagnation count to zero
            self.t_best = 0

            # set the new best
            self.f_best = current_f
        else: # if the performance, was not updated, just update the stagnation counter
            self.t_best = self.t_best + 1

        # now, we have to see if we have passed the maximum stagnation time, if so
        # we decrease the weight for reward thereby encouraging novel behavoirs
        if self.t_best >= self.t_max:
            self.current_rl_weight = max(0, self.current_rl_weight - self.rl_weight_delta)
            self.t_best = 0

