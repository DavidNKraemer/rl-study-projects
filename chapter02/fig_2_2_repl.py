import gym
import numpy as np
import matplotlib.pyplot as plt

from bandits import NormalBanditEnv, EpsilonGreedyPolicy

np.random.seed(0)

## Learning loop ##
num_bandits  =   10
num_runs     = 1000
num_episodes = 1000

epsilons = np.array([0.0, 0.01, 0.1])

## Plotting
fig, axes = plt.subplots(2, 1, figsize=(5,8))

for eidx, epsilon in enumerate(epsilons):
    avg_rewards = np.zeros((num_episodes))
    for run in range(num_runs):
        # Set up the bandits
        means     = 2 * np.random.rand(num_bandits) - 1
        variances = np.ones(means.size)
        env = NormalBanditEnv(means, variances)
        env.reset()
    
        # epsilon-greedy policy for the loop
        policy = EpsilonGreedyPolicy(env.action_space, epsilon)
        rewards  = np.zeros(num_episodes)
        values   = np.zeros(num_bandits)
        counts   = np.zeros(num_bandits)
        optimals = np.zeros(num_episodes)
    
        for episode in range(num_episodes):
            # print((
            #     f"run {run} / {num_runs} | "
            #     f"episode {episode} / {num_episodes} | "
            #     f"epsilon {epsilon}"
            # ))

            # get an action from the policy
            action = policy.act(values)
            print(f"{np.argmax(means)}, {action}")
            counts[action] += 1
            optimals[episode] = float(action == np.argmax(means))
        
            # give the action to the environment, update the environment, get the reward
            _, rewards[episode], _, _ = env.step(action)
        
            # update the average reward over time based on the new data
            values[action] += (rewards[episode] - values[action]) / counts[action]

        avg_rewards += rewards.cumsum() / np.arange(1, num_episodes+1)
        optimals += optimals.cumsum() / np.arange(1, num_episodes+1)
    avg_rewards /= num_runs
    optimals /= num_runs
    axes[0].plot(avg_rewards, label=f"$\\varepsilon = {epsilon}$")
    axes[1].plot(optimals, label=f"$\\varepsilon = {epsilon}$")

axes[0].legend()
axes[0].set(
    xlabel="Steps",
    ylabel="Average reward",
)

axes[1].legend()
axes[1].set(
    xlabel="Steps",
    ylabel="% Optimal action"
)

fig.savefig("fig.eps", bbox_inches="tight")
