import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q(state, a) for a in self.actions]
        return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        future_q = max([self.get_q(next_state, a) for a in self.actions])
        self.q_table[(state, action)] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def save(self, path="q_table.pkl"):
        pickle.dump(self.q_table, open(path, "wb"))

    def load(self, path="q_table.pkl"):
        self.q_table = pickle.load(open(path, "rb"))