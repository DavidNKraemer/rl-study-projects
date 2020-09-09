from collections import namedtuple
from operator import mul

MarkovDecisionProcess = namedtuple(
    'MarkovDecisionProcess',
    ['states', 'actions', 'transition_matrix', 'rewards', 'discount']
)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda tup: tup[1])

def dot(iter1, iter2):
    return sum(map(mul, iter1, iter2))


class DictFunction(dict):
    
    def __call__(self, state):
        return self.get(state)


class PolicyFunction(DictFunction):

    def __init__(self, mdp):
        self.mdp = mdp


class ValueFunction(DictFunction):

    def __init__(self, mdp):
        self.mdp = mdp
    
    def reset(self):
        for state in self.mdp.states:
            self[state] = 0.

    def value_step(self):
        policy = PolicyFunction(self.mdp)
        for state in self.mdp.states:
            policy[state], self[state] = argmax(dot(
                self.mdp.transition_matrix[self.mdp.states[state], self.mdp.actions[a]],
                (self.mdp.rewards(state, a, next_state=s) + self.mdp.discount * self[s]\
                for s in self.mdp.states)) for a in self.mdp.actions)
        return policy






# def value_iteration(mdp):
#     values = ValueFunction(mdp)
#     while True:
#         policy = values.value_step()
#         for state in mdp.states:
#             values[state] = max(
#                 sum(mdp.transition_matrix[(state, action), next_state] *\
#                     (mdp.rewards(state, action) + mdp.discount *\
#                      values[next_state]) for next_state in mdp.states) \
#                 for action in mdp.actions
#             )

    
