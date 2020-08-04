import numpy as np

from collections import defaultdict


def argmax(iterable, labels=None):
    """
    Return the pair (max, amax) where max is the max of iterable and amax is the
    arg max of the iterable
    """
    iterable = enumerate(iterable) if labels is None else zip(labels, iterable)
    am, m = max(iterable, key=lambda tuple: tuple[1])
    return m, am


class MarkovDecisionProcess:

    def __init__(self, states, actions, rewards, transitions, discount):
        """
        Params
        ------
        states: list
        actions: list
        rewards: dict (keys: (s, a) where s in states, a in actions)
        transitions: dict (keys: (s, a, s') where s, s' in states, a in actions)
        discount: float

        """
        self.states = states
        self.actions = actions
        self._actions_to_index = {a: i for i, a in enumerate(actions)}
        self._index_to_actions = {i: a for i, a in enumerate(actions)}

        # Construct the full rewards function (where the dictionary is
        # undefined, assume rewards of -infinity)
        self._rewards = defaultdict(lambda: np.finfo(np.float64).min)
        for key, reward in rewards.items():
            self._rewards[key] = reward

        # Construct the full transition kernel (where the dictionary is
        # undefined, assume probability 0)
        self._transitions = np.zeros((len(self.states), len(self.actions),
                                     len(self.states)))
        for (s, a, x), pr in transitions.items():
            self._transitions[s,self._actions_to_index[a],x] = pr

        self.discount = discount


    def rewards(self, state, action):
        """
        This makes the rewards function look like a function even though it's
        just looking through a dictionary
        """
        return self._rewards[(state, action)]

    def tp(self, state, action, next_state):
        """
        This gives a shorthand lookup for the transition matrix
        """
        return self._transitions[state, self._actions_to_index[action], next_state]


def value_iteration(mdp, tol=1e-5):
    """
    Given an MDP and a tolerance (tol), perform value iterations until the
    error is less than tol.

    Params
    ------
    mdp: a Markov Decision Process object (TBD)
    tol: float

    Returns
    -------
    value: dict
        a tol-suboptimal value function
    """
    # Initialization
    value = defaultdict(float)
    policy = {}

    def bellman_step(state, action, mdp=mdp, value=value):

        update = 0.
        for next_state in mdp.states:
            update += mdp.rewards(state, action)
            pr = mdp.tp(state, action, next_state)
            if pr > 0:
                update += mdp.discount * pr * value[next_state]
        return update

    while True:
        delta = 0.

        for state in mdp.states:
            old_value = value[state]

            value[state], policy[state] = argmax(
                (bellman_step(state, action) for action in mdp.actions),
                labels=mdp.actions
            )

            delta = max(delta, abs(old_value - value[state]))

        # Fixed point loop continuation
        if delta < tol:
            break

    return value, policy
