import numpy as np


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
        file.write(str(i) + "-" + "24\n")  # TODO: deshardcodear el objetivo del juego
    file.write("UP\n")
    file.write("RIGHT\n")
    file.write("DOWN\n")
    file.write("LEFT\n")
    for row in q_table:
        for col in row:
            file.write(str(col) + "\n")
    file.close()


def test_q_table(env, q_table, testing_episodes=50):
    """
    Test Q Table, print the average amount of steps
    """
    num_steps = 0
    for i in range(testing_episodes):
        state = env.reset()
        for j in range(100):
            sorted_actions = reversed(np.argsort(q_table[state, :]))
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

