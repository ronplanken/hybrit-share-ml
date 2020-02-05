import random
from collections import deque
import numpy as np
import logging

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.initializers import VarianceScaling

from keras.models import load_model, model_from_json

class Agent:
    """ Gamer agent """

    def __init__(self, env, input_size, action_size, model_from_memory=False):
        self.env = env
        self.memory = deque()
        self.input_size = input_size
        self.action_size = action_size

        self.first_iter = True
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.initializer = VarianceScaling()
        self.model_name = 'player'

        if model_from_memory:
            self.model = self.model_load()
        else:
            self.model = self.model()

    def model(self):
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.input_size, kernel_initializer=self.initializer))

        model.add(Activation('relu'))
        model.add(Dense(units=64, kernel_initializer=self.initializer))

        model.add(Activation('relu'))
        model.add(Dense(units=self.action_size, kernel_initializer=self.initializer))

        model.compile(loss='mse', optimizer='adam')

        return model

    def remember(self, state, action, reward, next_state, done):
        """ Adds relevant data to memory. """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, env, state, is_eval=False):
        """ Take action from given possible set of actions. """
        if not is_eval and random.random() <= self.epsilon:
            return env.action_space.sample()
        if self.first_iter:
            self.first_iter = False
            return 1
        options = self.model.predict(state)
        return np.argmax(options[0])

    def train_experience_replay(self):
        """ Train on previous experiences in memory. """
        logging.info('Learning network ...')
        randomized_memory = random.sample(self.memory, len(self.memory))

        X_train, y_train = [], []

        for state, action, reward, next_state, done in randomized_memory:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            X_train.append(state[0])
            y_train.append(target_f[0])

        self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Reset memory for a new game
        self.memory = deque()

    def model_load(self):
        return load_model('models/{}.{}'.format(self.model_name, "h5"))

    def model_save(self):
        """ Save model weights """
        self.model.save('models/{}.{}'.format(self.model_name, "h5"))