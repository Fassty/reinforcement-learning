import numpy as np
import multiprocessing as mp
import gym
import cart_pole_pixels_environment
import wrappers

_BATCH_SIZE = 128
_NUM_BATCHES = 64
_RENDER = False

def generate_action():
    action = np.random.randint(2)

    return action

def simulate_batch(batch_num):
    env = gym.make('CartPolePixels-v0')

    obs_data = []
    for i_episode in range(_BATCH_SIZE):
        observation, done = env.reset(), False

        while not done:
            if _RENDER:
                env.render()

            action = generate_action()

            observation, reward, done, _ = env.step(action)

            obs_data.append(observation)

    print("Saving dataset for batch {}".format(batch_num))
    np.save('./data/obs_data_VAE_{}'.format(batch_num), obs_data)

    env.close()

def main():
    print("Generating data for env CartPolePixels-v0")

    with mp.Pool(mp.cpu_count()) as p:
       p.map(simulate_batch, range(_NUM_BATCHES))


if __name__ == "__main__":
    main()
