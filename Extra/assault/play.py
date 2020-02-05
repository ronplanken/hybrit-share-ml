import gym
from time import sleep
from agent import Agent


def main():
    input_size = 1
    action_size = 17

    env = gym.make('Assault-ram-v0')  # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

    agent = Agent(env, input_size, action_size, model_from_memory=True)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.act(env, state, is_eval=True)

        next_state, reward, done, info = env.step(action)
        sleep(0.01)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Aborted!')
