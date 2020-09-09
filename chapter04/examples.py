import numpy as np

from mdp import MarkovDecisionProcess, PolicyFunction, ValueFunction


# Recycling robot example

def convert_to_numeric(descriptive_list):
    return {item: num for num, item in enumerate(descriptive_list)}

states= convert_to_numeric(['high', 'low'])
actions = convert_to_numeric(['search', 'wait', 'recharge'])

alpha, beta, r_wait, r_search, discount = 0.7, 0.3, 5., 2., 0.9

transitions = {
    ('high', 'wait', 'high'): 1.,
    ('high', 'search', 'high'): alpha,
    ('high', 'search', 'low'): 1. - alpha,
    ('low', 'search', 'high'): 1. - beta,
    ('low', 'search', 'low'): beta,
    ('low', 'recharge', 'high'): 1.,
    ('low', 'wait', 'low'): 1.
}

transition_matrix = np.array(
    [[[transitions.get((state, action, next_state)) or 0.0 \
    for next_state in states] for action in actions] for state in states]
)

rewards_dict = {
    ('high', 'wait', None): r_wait,
    ('high', 'search', None): r_search,
    ('low', 'search', 'high'): -3,
    ('low', 'search', 'low'): r_search,
    ('low', 'wait', None): r_wait,
    ('low', 'recharge', None): 0.
}

def rewards(state, action, next_state=None):
    reward = rewards_dict.get((state, action, next_state))
    return reward if reward is not None else np.finfo(np.float64).min

mdp = MarkovDecisionProcess(
    states, actions, transition_matrix, rewards, discount
)

values = ValueFunction(mdp)
values.reset()

for _ in range(20):
    policy = values.value_step()
    print(values, policy)
