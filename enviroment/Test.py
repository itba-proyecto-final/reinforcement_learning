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
accum_rewards = 0
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    # The Q-Table learning algorithm
    for j in range(100):
        env.render()
        print("")
        # Choose an action by greedily (with noise) picking from Q table
        # action = np.argmax((Q[state, :] + np.random.randn(1, env.action_space) * (1. / (i + 1))).tolist())
        sorted_actions = np.argsort(Q[state, :] + np.random.randn(1, env.action_space) * (1. / (i + 1)))
        for a in sorted_actions[0]:  # Check that we are using a valid action
            if env.is_valid_action(a):
                action = a
                break
        # valid_actions = [x for x in [0, 1, 2, 3] if env.is_valid_action(x)]
        new_state, reward, is_done, _ = env.step(action)
        # Update Q-Table with new knowledge
        # if action in valid_actions:  # Guarantee that the table will only have non-zero values for valid action/states
        Q[state, action] = Q[state, action] + learning_rate * (reward + y * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state
        if is_done:
            print("GOAL")
            accum_rewards += reward
            break

print("Final Q-Table Values")
print(Q)
print(rList)
#
# """
# Use trained model
# """
# all_rewards = 0
# state = env.reset()
# accum_rewards = 0
#
# for i in range(num_episodes):
#     state = env.reset()
#     for j in range(100):
#         action = np.argmax(Q[state, :])
#         state_new, reward, is_done, _ = env.step(action)
#         state = state_new
#         if is_done:
#             # env.render()
#             accum_rewards += reward
#             break
# print("Average score when testing: " + str(accum_rewards/num_episodes))
#
