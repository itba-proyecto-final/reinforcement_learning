import random
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from enviroment.MountainCart.GameRunner import GameRunner
from enviroment.MountainCart.Memory import Memory
from enviroment.MountainCart.Model import Model

BATCH_SIZE = 5
MAX_EPSILON = .12
MIN_EPSILON = .08
LAMBDA = .99

env_name = 'MountainCar-v0'
env = gym.make(env_name)

num_states = env.env.observation_space.shape[0]
num_actions = env.env.action_space.n

model = Model(num_states, num_actions, BATCH_SIZE)
mem = Memory(50000)

with tf.Session() as sess:
    sess.run(model.var_init)
    gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
    num_episodes = 10
    cnt = 0
    while cnt < num_episodes:
        if cnt % 10 == 0:
            print('Episode {} of {}'.format(cnt+1, num_episodes))
        gr.run()
        cnt += 1
    plt.plot(gr.reward_store)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    plt.close("all")
    plt.plot(gr.max_x_store)
    plt.show()