import sys
import os
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

class DQNCartPoleSolver():
    def __init__(self, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, model_file=None):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = 10000
        self.n_win_ticks = 195
        self.batch_size = batch_size

        # Init model
        if model_file is None:
            self.model = Sequential()
            self.model.add(Dense(24, input_dim=4, activation='tanh'))
            self.model.add(Dense(48, activation='tanh'))
            self.model.add(Dense(2, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        else:
            self.model = load_model(model_file)
            print("Model loaded")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                if reward != 1.0:
                    print(reward)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if e % 100 == 0:
                    self.env.render()

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                print('Ran {} episodes. Solved after {} trials :)'.format(e, e - 100))
                return e - 100
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)
        
        print('Did not solve after {} episodes :('.format(e))
        return e
    
    def test(self):
        overall_scores = []
        scores = deque(maxlen=10)
        for e in range(100):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, 0.0)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                state = next_state
                i += 1
                if e % 10 == 0 and i % 3 == 0:
                    self.env.render()

            scores.append(i)
            overall_scores.append(i)
            if (e + 1) % 10 == 0:
                print('[Episode {}-{}] - Mean survival time: {}'.format(e - 8, e + 1, np.mean(scores)))
        print('OVERALL MEAN SURVIVAL TIME: {}'.format(np.mean(overall_scores)))

if __name__ == '__main__':
    is_testing = False
    agent = None
    if len(sys.argv) > 1:
        is_testing = sys.argv[1] == "--test"
    try:
        if is_testing:
            file = sys.argv[2]
            if os.path.exists(file):
                agent = DQNCartPoleSolver(model_file=file, monitor=True)
                agent.test()
            else:
                raise Exception("The file {} does not exist.".format(file))
        else:
            agent = DQNCartPoleSolver()
            agent.run()
    except (KeyboardInterrupt, ImportError):
        pass
    except OSError:
        raise Exception("You are trying to load an invalid file type. It must be of type .h5")
    finally:
        if agent is not None:
            if agent.env is not None:
                agent.env.close()
        if not is_testing:
            agent.model.save("model.h5")
            print("Saved model as model.h5")
