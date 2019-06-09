import gym
import numpy as np
import gym_chase
from argparse import ArgumentParser
from enviroment.q_tools import write_q_table_file, test_q_table


def train_q_algorithm(env, training_episodes=1, learning_rate=.8, y=.95, q_table=None):
    """
    Train Q table and return it
    :return: Q Table
    """
    # Initialize Q-table with all zeros
    if q_table is None:
        q_table = np.zeros([env.observation_space, env.action_space])
    num_steps = 0
    for i in range(training_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        # The Q-Table learning algorithm
        is_done = False
        while not is_done:
            new_state, reward, is_done, action = env.step()
            # Update Q-Table with new knowledge
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + y * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state
        num_steps += env.number_of_steps
    write_q_table_file(q_table)
    print("Final Q-Table Values")
    print(q_table)
    print("Average amount of steps when training:" + str(num_steps/training_episodes))
    return q_table


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-q", "--quiet", dest="verbose", default=True, help="don't print status messages to stdout")
    parser.add_argument('-f', '--files', dest="filenames", nargs='+', help='List of files that contain game information', required=True, type=str)

    args = parser.parse_args()
    game_files = args.filenames

    env = gym.make('chase-mental-v0')
    Q_table = None
    for game_file in game_files:
        env.reset()
        env.set_game_file(game_file)
        Q_table = train_q_algorithm(env, q_table=Q_table)

    test_env = gym.make('chase-v0')
    test_env.set_num_row_cols(5)
    test_env.goal = (4,4)
    test_q_table(env=test_env, q_table=Q_table)
