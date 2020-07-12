import gym
import numpy as np


class NormalBandit:
    """
    A bandit is a slot machine. The only thing to do with a bandit is pull the
    lever and get a reward. 

    A NormalBandit is a special kind of bandit where the reward is a normally
    distributed random variable.
    """

    def __init__(self, mean, var):
        """
        Params
        ------
        mean: float
            mean of the reward distribution
        var: float
            variance of the reward distribution
        """
        self.mean = mean
        self.var = var
        self.std = np.sqrt(self.var)

    def pull(self):
        """
        Pulls the bandit and gets the reward

        Returns
        -------
        reward: float
            normally distributed with the specified normal distribution
        """
        return np.random.randn() * np.sqrt(self.std) + self.mean


class NormalBanditEnv(gym.core.Env):
    """
    In this environment, you specify one of a list of bandits to pull and
    receive its reward.

    The bandits are specified by two lists of means and variances that
    completely determine their reward distributions.

    This is a subclass of the Env object in the OpenAI gym library. We will use
    this library going forward.
    """

    def __init__(self, means, variances):
        """
        Params
        ------
        means: np.ndarray
            means of each bandit reward distribution
        variances: np.ndarray
            variances of each bandit reward distribution

        Preconditions
        -------------
        means.size == variances.size
        """
        self.bandits = [NormalBandit(mean, var) for mean, var in zip(means, variances)]

    def reset(self):
        """
        Standard method for gym.core.Env objects. Call this before running any
        experiments.
        """
        self.action_space = gym.spaces.Discrete(len(self.bandits))
        self.observation_space = None
        self.state = None

    def step(self, action):
        """
        Update the environment based on a specified action. The idea is that the
        agent produces the action and then the system evolves.

        Params
        ------
        action: int
            the chosen bandit

        Returns
        -------
        state, reward, done, info: None, float, Bool, dict
            state is always None because this environment has no state
            reward is the reward of the pulled bandit
            done is a Bool saying that the environment has terminated (doesn't
                happen for bandit environments)
            info is a dict of anything interesting going on in the environment
                (empty for bandit environments)
        """
        err_msg = f"{action} {type(action)} invalid"
        assert self.action_space.contains(action), err_msg

        reward = self.bandits[action].pull()

        return None, reward, False, {}


class EpsilonGreedyPolicy:
    """
    An epsilon-greedy policy selects the greedy action with probability
    1-epsilon and a random action with probability epsilon.
    """

    def __init__(self, action_space, epsilon):
        """
        Params
        ------
        action_space: gym.core.Space
            a somewhat advanced object that specifies the available actions in
            the environment (for bandits, this is just the numbers 0,..., num
            bandits)
        epsilon: float
            epsilon is the exploration parameter
        """
        self.action_space = action_space
        self.epsilon = epsilon

    def act(self, values):
        """
        Given a collection of values associated with each action, select the
        corresponding epsilon-greedy action.

        Params
        ------
        values: np.ndarray
            the "value" of each action based on some determination

        Returns
        -------
        action: int
            the epsilon-greedy action for the values array
        """
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(values)

