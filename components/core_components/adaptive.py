
def adapt_novelty_pressure(current_f, f_best, t_best, current_rl_weight, rl_weight_delta, t_max):
    """
    Refer NSRA-ES algorithm from
    """
    # if the performance has improved, increase the reward pressure so that we don't explore
    # with novelty weights anymore
    if (current_f > f_best):

        # increase the reward pressure
        current_rl_weight = min(1.0, current_rl_weight + rl_weight_delta)

        # reset the stagnation count to zero
        t_best = 0

        # set the new best
        f_best = current_f
    else: # if the performance, was not updated, just update the stagnation counter
        t_best = t_best + 1

    # now, we have to see if we have passed the maximum stagnation time, if so
    # we decrease the weightage for reward thereby encouraging novel behavoirs
    if t_best > t_max:
        current_rl_weight = max(0, current_rl_weight - rl_weight_delta)
        t_best = 0

    return current_rl_weight, f_best, t_best

