import math
import random
import numpy as np

MAX_EPSILON = .12
MIN_EPSILON = .08
LAMBDA = .99
GAMMA = .99 # maximum discounted future reward expected according to our own table for the next state

class GameRunner:
    """
    This class is the main training and agent control class
    """
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        self.sess = sess
        self.env = env
        self.model = model
        self.memory = memory
        self.render = render
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.decay = decay
        self.eps = self.max_eps
        self.steps = 0
        self.reward_store = []
        self.max_x_store = []

    def run(self):
        state = self.env.reset()
        tot_reward = 0
        max_x = -100
        while True:
            if self.render:
                self.env.render()

            action = self._choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            if next_state[0] >= 0.1:
                reward += 10
            elif next_state[0] >= 0.25:
                reward += 20
            elif next_state[0] >= 0.5:
                reward += 100

            if next_state[0] > max_x:
                max_x = next_state[0]
            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None

            self.memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self.steps += 1
            self.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self.reward_store.append(tot_reward)
                self.max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self.steps, tot_reward, self.eps))

    def _choose_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.model.num_actions - 1)
        else:
            return np.argmax(self.model.predict_one(state, self.sess))

    def _replay(self):
        batch = self.memory.sample(self.model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self.model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self.model.predict_batch(states, self.sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predict_batch(next_states, self.sess)
        # setup training arrays
        x = np.zeros((len(batch), self.model.num_states))
        y = np.zeros((len(batch), self.model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                # np.amax(q_s_a_d[i]) = maximum Q value possible in the next state, the agent starts in state s, takes
                # action a, ends up in state s’ and then the code determines the maximum Q value in state s’
                # value is discounted by γ to take into account that it isn’t ideal for the agent to wait forever for
                # a future reward – it is best for the agent to aim for the maximum award in the least period of time.
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self.model.train_batch(self.sess, x, y)