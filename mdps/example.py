from itertools import product

from mdp import MarkovDecisionProcess, value_iteration


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


# Main code


num_states = 15
oases = [0, 14]  # state indices with oases
mdp = FindOasis1D(num_states, oases)

value, policy = value_iteration(mdp)

print(value)
print(policy)
