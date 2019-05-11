import gym
import numpy as np
import gym_chase

from enviroment.q_learning import write_q_table_file, test_q_table

env = gym.make('chase-v0')
LEARNING_RATE = .2
Y = .3


def train_q_algorithm(training_episodes=100, learning_rate=.8, y=.95):
    """
    Train Q table and return it
    :return: Q Table
    """
    # Initialize Q-table with all zeros
    Q = np.zeros([env.observation_space, env.action_space])
    num_steps = 0
    for i in range(training_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        # The Q-Table learning algorithm
        for j in range(2000):
            # Choose an action by greedily (with noise) picking from Q table
            sorted_actions = reversed(np.argsort(Q[state, :] + np.random.randn(env.action_space) * (1. / (i + 1))))
            for a in sorted_actions:  # Check that we are using a valid action
                if env.is_valid_action(a):
                    action = a
                    break
            new_state, reward, is_done, _ = env.step(action)
            # Update Q-Table with new knowledge
            Q[state, action] = Q[state, action] + learning_rate * (reward + y * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            if is_done:
                num_steps += env.number_of_steps
                print(env.number_of_steps)
                break
    write_q_table_file(Q)
    print("Final Q-Table Values")
    print(Q)
    print("Average amount of steps when training:" + str(num_steps/training_episodes))
    return Q


Q_table = train_q_algorithm()
test_q_table(env, Q_table)