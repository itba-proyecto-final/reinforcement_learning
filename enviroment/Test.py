import gym
import numpy as np
import gym_chase


env = gym.make('chase-v0')
LEARNING_RATE = .8
Y = .95


def write_q_table_file(q_table, q_file="Q_Table.txt"):
    """
    Write the q table in a file that will be used to play game
    :param q_table: Q table to be written in file, has to be a matrix
    :param q_file: File name to write in
    """
    file = open(q_file, "w+")
    rows = len(q_table)
    cols = len(q_table[0])
    file.write(str(rows) + "x" + str(cols) + "\n")
    for i in range(len(q_table)):
        file.write(str(i) + "-" + "13\n")  # TODO: deshardcodear el objetivo del juego
    file.write("UP\n")
    file.write("RIGHT\n")
    file.write("DOWN\n")
    file.write("LEFT\n")
    for row in q_table:
        for col in row:
            file.write(str(col) + "\n")
    file.close()


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
    print("Average amount of steps when training:" + str(num_steps/training_episodes))
    write_q_table_file(Q)
    return Q


def test_q_table(q_table, testing_episodes=50):
    """
    Test Q Table, print the average amount of steps
    """
    num_steps = 0
    for i in range(testing_episodes):
        state = env.reset()
        for j in range(100):
            sorted_actions = np.argsort(q_table[state, :])
            for a in sorted_actions:  # Check that we are using a valid action
                if env.is_valid_action(a):
                    action = a
                    break
            state_new, reward, is_done, _ = env.step(action)
            state = state_new
            if is_done:
                num_steps += env.number_of_steps
                break
    print("Average amount of steps when testing: " + str(num_steps/testing_episodes))


Q_table = train_q_algorithm()
test_q_table(Q_table)