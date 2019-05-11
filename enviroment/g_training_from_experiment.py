import gym
import numpy as np
import gym_chase

from enviroment.q_tools import write_q_table_file

env = gym.make('chase-mental-v0')
LEARNING_RATE = .2
Y = .3


def train_q_algorithm(training_episodes=300, learning_rate=.8, y=.95):
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
            new_state, reward, is_done, action = env.step()
            # Update Q-Table with new knowledge
            Q[state, action] = Q[state, action] + learning_rate * (reward + y * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            if is_done:
                num_steps += env.number_of_steps
                break
    write_q_table_file(Q)
    print("Final Q-Table Values")
    print(Q)
    print("Average amount of steps when training:" + str(num_steps/training_episodes))
    return Q


Q_table = train_q_algorithm(training_episodes=1)