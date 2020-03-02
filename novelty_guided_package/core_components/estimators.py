import numpy as np
from torch.multiprocessing import Pool
import torch
from functools import reduce


class PerturbRunnerES:
    def __init__(self, loss_function):
        self.loss_function = loss_function

        # temp variables (to avoid passing to a function because of map())
        self.sigma = None
        self.current_estimate = None
        self.current_parameter = None
        self.device = None

    def run_for_perturb(self, delta, seed):
        scale = self.sigma

        # perturb
        param_perturb = self.current_estimate + delta * scale

        # measure the loss function at perturbations
        self.loss_function.parameter_name = self.current_parameter
        reward, beh, _ = self.loss_function(param_perturb, self.device)

        return reward, beh, delta


class ES:
    """
    An optimizer class that implements Evolution Strategies (ES)
    """
    def __init__(self, sigma, k, loss_function, model, novelty_detector, device_list, use_parallel_gpu=True, rl_weight=1.0, parallel_workers=4):
        self.sigma = sigma
        self.k = k
        self.loss_function = loss_function
        self.model = model
        self.novelty_detector = novelty_detector
        self.device_list = device_list
        self.use_parallel_gpu = use_parallel_gpu
        self.rl_weight = rl_weight

        # generate a pool of processes
        self.p = Pool(processes=parallel_workers)

        # runner
        self.runner = PerturbRunnerES(loss_function)

    def _parse_obj_values(self, obj_values):

        # for the novelty, let's just do batching
        # only implemented for seq_ae
        # TBD: fix other novelty detection modules
        # if self.novelty_detector is not None:
        #     behs = [beh for _,beh,_ in obj_values]
        #     novelties = self.novelty_detector.get_novelties(behs)

        for i in range(len(obj_values)):
            # let's just get the novelty values corresponding to the behavior values
            obj, beh, delta = obj_values[i]
            if self.novelty_detector is not None:
                nov = self.novelty_detector.get_novelty(beh)

                # fit the model with new observed behavoirs
                self.novelty_detector.save_behaviors([beh])
            else:
                nov = 0.0

            obj_values[i] = (obj, nov, delta)

        return obj_values

    def _compute_ranks(self, x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        source: from ES starter repo
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def _compute_centered_ranks(self, x):
        y = self._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y

    def _fit_rank_scaler(self, values):

        # convert to centered-rank values
        ranked_values = self._compute_centered_ranks(values)

        # now, create a mapping of value as key and its rank transformed
        scaler_mapping = {}
        for value, ranked in zip(values, ranked_values):
            scaler_mapping[value] = ranked
        return scaler_mapping

    def _compute_weighted_total_objective(self, obj, nov, delta):

        # scaled total objective - weighted sum of objective and novelty
        total_obj = delta * (self.rl_weight * obj + (1.0 - self.rl_weight) * nov)

        return total_obj

    def _gradients_from_objectives(self, current_estimate, obj_values):
        total_gradients = torch.zeros(current_estimate.shape)

        # get novelty scores corresponding to behavior values
        obj_values = self._parse_obj_values(obj_values)

        # here, we have to collect all novelty scores and reward scores and create a rank scaler objects
        # that does mapping of novelty/reward to scores
        all_reward_scores = reduce(lambda x,y: x + [y[0]], obj_values, [])
        all_novelty_scores = reduce(lambda x,y: x + [y[1]], obj_values, [])

        reward_scaler = self._fit_rank_scaler(np.array(all_reward_scores)) # it's just a dict mapping
        novelty_scaler = self._fit_rank_scaler(np.array(all_novelty_scores)) # it's just a dict mapping

        for obj in obj_values:

            # separate the novelty values and objective values
            obj, nov, delta = obj

            # here, we just apply the scaler objective
            obj = reward_scaler[obj]
            nov = novelty_scaler[nov]

            # compute gradient
            gradient = self._compute_weighted_total_objective(obj, nov, delta)

            total_gradients += gradient

        # average the gradients
        g_t = total_gradients / (self.k * self.sigma)

        return g_t

    def _run_parellely(self, current_estimate, current_parameter):

        # create mirror sampled perturbations
        perturbs = [torch.randn_like(current_estimate) for i in range(int(self.k/2))]
        mirrored = [-i for i in perturbs]
        perturbs += mirrored

        # seeds for grid envs (also mirrored)
        seeds = list(range(int(self.k/2)))
        seeds += seeds

        # run parallely for all k mirrored candidates
        self.runner.sigma = self.sigma
        self.runner.current_estimate = current_estimate
        self.runner.current_parameter = current_parameter
        self.runner.device = self.device_list[0]
        obj_values = self.p.starmap(self.runner.run_for_perturb, zip(perturbs, seeds))
        behaviors = [item[1] for item in obj_values]
        gradient = self._gradients_from_objectives(current_estimate, obj_values)
        return gradient, behaviors

    def estimate_gradient(self, current_parameter, current_estimate):
        """
        :param current_estimate: This is the current estimate of the parameter vector
        :return: returns the estimate of the gradient at that point
        """

        # run parallely
        g_t, behaviors = self._run_parellely(current_estimate, current_parameter)
        return g_t, behaviors