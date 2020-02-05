import gym
import logging
import coloredlogs

from agent import Agent


def main():
    ep_count = 2000
    input_size = 1
    action_size = 17

    env = gym.make('Assault-ram-v0') #https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

    agent = Agent(env, input_size, action_size)

    for episode in range(1, ep_count + 1):
        train(env, agent, episode)

    env.close()

def train(env, agent, episode):
    state = env.reset()
    done = False
    logging.info('Playing game {}'.format(episode))
    while not done:
        env.render()
        action = agent.act(env, state)

        next_state, reward, done, info = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state

    agent.train_experience_replay()

    logging.info('Game result {}'.format(info))

    if episode % 100 == 0:
        agent.model_save()

if __name__ == '__main__':
    coloredlogs.install(level='DEBUG')
    try:
        main()
    except KeyboardInterrupt:
        print('Aborted!')