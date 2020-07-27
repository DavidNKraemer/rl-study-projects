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


class FindOasis1D(MarkovDecisionProcess):
    """
    The FindOasis1D MDP is a game where the object is to find the "oasis"
    locations on a 1-dimensional grid.

    The states are [0, ..., n-1]. Each state is either "desert" or "oasis".

    At each state, the player can move left, move right, or stay put.

    There is a "desert wind" blowing against whichever direction the player
    moves. This means that with a certain probability, the wind will push the
    player in the direction *opposite* of motion. When the player chooses to
    stay put, there is a certain probability that they will end up *either* to
    the left or right of the original location. Boundaries are "clamped", so
    for example moving left at state 0 is the same as staying.

    The "desert" states have a small reward, the "oasis" states have a huge
    reward.

    By playing this game, the player will determine the surest route to the
    oases nearest to her.
    """

    @staticmethod
    def _get_states(num_states):
        return list(range(num_states))

    @staticmethod
    def _get_actions():
        return ["left", "right", "stay"]

    @staticmethod
    def _get_rewards(states, actions, oasis_states, oasis_reward,
                     desert_reward):
        return {
            (s, a): oasis_reward if s in oasis_states else desert_reward \
            for s in states for a in actions
        }

    @staticmethod
    def _get_transitions(states, desert_wind):
        transitions = {}

        # Tabulate "left" actions
        for state, next_state in product(states, repeat=2):
            if next_state == states[0] and state == states[0]:
                transitions[(state, "left", next_state)] = 1.-desert_wind
            elif state - next_state == 1:
                if state == states[-1]:
                    transitions[(state, "left", state)] = desert_wind
                transitions[(state, "left", next_state)] = 1.-desert_wind
            elif next_state - state == 1:
                transitions[(state, "left", next_state)] = desert_wind

        # Tabulate "right" actions
        for state, next_state in product(states, repeat=2):
            if next_state == states[-1] and state == states[-1]:
                transitions[(state, "right", next_state)] = 1.-desert_wind
            elif state - next_state == -1:
                if state == states[0]:
                    transitions[(state, "right", state)] = desert_wind
                transitions[(state, "right", next_state)] = 1.-desert_wind
            elif next_state - state == -1:
                transitions[(state, "right", next_state)] = desert_wind

        # Tabulate "stay" actions
        for state in states:
            transitions[(state, "stay", state)] = 1. - desert_wind
            if state == states[0]:
                transitions[(state, "stay", state)] += 0.5 * desert_wind
                transitions[(state, "stay", state+1)] = 0.5 * desert_wind
            elif state == states[-1]:
                transitions[(state, "stay", state)] += 0.5 * desert_wind
                transitions[(state, "stay", state-1)] = 0.5 * desert_wind
            else:
                transitions[(state, "stay", state-1)] = 0.5 * desert_wind
                transitions[(state, "stay", state+1)] = 0.5 * desert_wind

        return transitions

    def __init__(self, num_states, oasis_states, oasis_reward=1.,
                 desert_reward=0., desert_wind=0.25, discount=0.99):

        states = FindOasis1D._get_states(num_states)
        actions = FindOasis1D._get_actions()
        rewards = FindOasis1D._get_rewards(states, actions, oasis_states,
                                           oasis_reward, desert_reward)

        transitions = FindOasis1D._get_transitions(states, desert_wind) 
        super().__init__(states, actions, rewards, transitions, discount)


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
