import gym
import gym_chase
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from enviroment import q_training_from_experiment
from enviroment.q_tools import test_q_table

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--files', dest="filenames", nargs='+', help='<Required> Files to train the QTable',
                        required=True)
    args = parser.parse_args()

    experiments = args.filenames
    steps = []
    q_table = None

    for experiment in experiments:
        env = gym.make('chase-mental-v0')
        env.set_game_file(experiment)
        if q_table is None:
            q_table = np.zeros([env.observation_space, env.action_space])
        q_table = q_training_from_experiment.train_q_algorithm(env, training_episodes=3, q_table=q_table)
        test_env = gym.make('chase-v0')
        test_env.set_num_row_cols(5)
        test_env.goal = (4, 4)
        all_steps, average_steps = test_q_table(env=test_env, q_table=q_table)
        steps.append(average_steps)

    plt.plot(range(len(steps)), steps)
    plt.title("Average steps to reach goal", fontsize=19)
    plt.xlabel("Experiments", fontsize=10)
    plt.ylabel("Average steps", fontsize=10)
    plt.show()