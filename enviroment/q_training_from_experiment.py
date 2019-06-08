import gym
import numpy as np
import gym_chase
from argparse import ArgumentParser
from enviroment.q_tools import write_q_table_file, test_q_table


parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", help="File containing game states", metavar="FILE")
parser.add_argument("-q", "--quiet", dest="verbose", default=True, help="don't print status messages to stdout")

args = parser.parse_args()
game_file = args.filename


LEARNING_RATE = .2
Y = .3


def train_q_algorithm(env, training_episodes=1, learning_rate=.8, y=.95):
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
        is_done = False
        while not is_done:
            new_state, reward, is_done, action = env.step()
            # Update Q-Table with new knowledge
            Q[state, action] = Q[state, action] + learning_rate * (reward + y * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
        num_steps += env.number_of_steps
    write_q_table_file(Q)
    print("Final Q-Table Values")
    print(Q)
    print("Average amount of steps when training:" + str(num_steps/training_episodes))
    return Q


env = gym.make('chase-mental-v0')
env.set_game_file(game_file)
Q_table = train_q_algorithm(env)
test_env = gym.make('chase-v0')
test_env.set_num_row_cols(5)
test_env.goal = (4,4)
test_q_table(env=test_env, q_table=Q_table)