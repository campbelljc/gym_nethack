import random
import numpy as np

from gym_nethack.policies.core import Policy

class LinearAnnealedPolicy(Policy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy "{}" does not have attribute "{}".'.format(attr))

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps
        super().__init__(name=inner_policy.name + str(value_max) + 'to' + str(value_min))        

    def get_current_value(self):
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.step) + b)
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        config = super(LinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config

class EpsGreedyPossibleQPolicy(Policy):
    def __init__(self, eps=.1):
        super().__init__(name='egreedy')
        self.eps = eps
    
    def select_action(self, q_values, valid_action_indices):
        if len(valid_action_indices) == 1:
            return valid_action_indices[0]
        
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        
        if np.random.uniform() < self.eps:
            action = np.random.choice(valid_action_indices)
        else:
            mask = np.ones(len(q_values), np.bool)
            mask[valid_action_indices] = 0
            q_values[mask] = -100
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

class BoltzmannPossibleQPolicy(Policy):
    def __init__(self, tau=1., clip=(-500., 500.)):
        super().__init__(name='boltzmann')
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values, valid_action_indices):
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        
        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        
        # set the exp values of impossible actions to 0
        exp_values = [e_val if i in valid_action_indices else 0 for i, e_val in enumerate(exp_values)]
        
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(BoltzmannPossibleQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config
