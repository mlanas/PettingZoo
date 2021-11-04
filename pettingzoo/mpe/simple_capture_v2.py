from pettingzoo.utils.conversions import parallel_wrapper_fn

from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.simple_capture import Scenario


class raw_env(SimpleEnv):
    def __init__(self, n_agents=3, n_landmarks=8, local_ratio=0.5, max_cycles=25, continuous_actions=False):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(n_agents, n_landmarks)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata['name'] = "simple_capture_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
