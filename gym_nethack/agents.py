import numpy as np
from libs.rl.core import Agent

class TestAgent(Agent):
    '''A Keras-RL agent for purely heuristic policies, i.e., no learning done, no neural network model required.'''
    def __init__(self, test_policy=None, **kwargs):
        super(TestAgent, self).__init__(**kwargs)
        self.policy = test_policy
        self.compile()

    def forward(self, observation, valid_action_indices):
        if len(valid_action_indices) == 0:
            action = None
        else:
            action = self.policy.select_action(q_values=None, valid_action_indices=valid_action_indices)
        self.recent_action = action
        self.recent_observation = observation
        return action

    def backward(self, reward, valid_action_indices, terminal):
        metrics = [np.nan for _ in self.metrics_names]
        return metrics

    def compile(self):
        self.compiled = True

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        if policy is not None:
            self.__policy._set_agent(self)
