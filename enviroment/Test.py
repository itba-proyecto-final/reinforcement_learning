import gym
import numpy as np
import gym_chase

# env = gym.make('FrozenLake-v0')
env = gym.make('chase-v0')

# Initialize Q-table with all zeros
Q = np.zeros([env.observation_space, env.action_space])

# Set learning parameters
learning_rate = .8
y = .95
num_episodes = 100  # TODO ver que rompe con mas de un episode, elige acciones que no son validas
# create lists to contain total rewards and steps per episode
rList = []
"""
Train the agent
"""
num_steps = 0
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    # The Q-Table learning algorithm
    for j in range(100):
        # Choose an action by greedily (with noise) picking from Q table
        sorted_actions = np.argsort(Q[state, :] + np.random.randn(env.action_space) * (1. / (i + 1)))
        for a in sorted_actions:  # Check that we are using a valid action
            if env.is_valid_action(a):
                action = a
                break
        new_state, reward, is_done, _ = env.step(action)
        # Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + learning_rate * (reward + y * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state
        if is_done:
            print(env.number_of_steps)
            num_steps += env.number_of_steps
            break

print("Final Q-Table Values")
print(Q)
print(rList)
print("Average amount of steps when training:" + str(num_steps/num_episodes))

"""
Use trained model
"""
state = env.reset()
num_steps = 0

for i in range(num_episodes):
    state = env.reset()
    for j in range(100):
        sorted_actions = np.argsort(Q[state, :])
        for a in sorted_actions:  # Check that we are using a valid action
            if env.is_valid_action(a):
                action = a
                break
        state_new, reward, is_done, _ = env.step(action)
        state = state_new
        if is_done:
            num_steps += env.number_of_steps
            break
print("Average amount of steps when testing: " + str(num_steps/num_episodes))

