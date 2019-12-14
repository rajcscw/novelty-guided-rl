import unittest
from novelty_guided_package.novelty_components.adaptor import NoveltyAdaptor


class TestNoveltyAdaptor(unittest.TestCase):
    def test_stagnation(self):
        t_max, rl_weight, rl_weight_delta, perf = 5, 1.0, 0.1, 10
        novelty_adaptor = NoveltyAdaptor(rl_weight=rl_weight, rl_weight_delta=rl_weight_delta, t_max=t_max)
        for i in range(t_max):
            novelty_adaptor.adapt(perf)

        # weight does not change yet
        self.assertEqual(rl_weight, novelty_adaptor.current_rl_weight)

        # it must detect stagnation and change
        novelty_adaptor.adapt(perf)
        self.assertEqual(rl_weight-rl_weight_delta, novelty_adaptor.current_rl_weight)

    def test_improvement(self):
        t_max, rl_weight, rl_weight_delta, perf = 5, 0.9, 0.1, 10
        novelty_adaptor = NoveltyAdaptor(rl_weight=rl_weight, rl_weight_delta=rl_weight_delta, t_max=t_max)
        novelty_adaptor.adapt(perf+1)  # improve the perf

        # now, it must increment the weight
        self.assertEqual(rl_weight+rl_weight_delta, novelty_adaptor.current_rl_weight)