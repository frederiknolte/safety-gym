import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import imageio
import gym

from safety_gym.envs.suite import vision_env_base


def make_movie(frames, path, filename, fps=24):
    path = os.path.join(path, 'mp4')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename + '.mp4')
    writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def get_action(env, magnitude=60.):
    selected_ball_id = np.random.randint(low=0, high=env.balls_num)
    ball_pos = env.layout['ball'+str(selected_ball_id)]
    robot_pos = env.layout['robot']
    pos_diff = ball_pos - robot_pos
    action_vector = pos_diff / np.sqrt(np.sum(pos_diff ** 2))
    action = action_vector * magnitude * np.array([-1., 1.])
    action = np.flip(action)
    return action


# Create environment
vision_env_base.register('', {'camera_name': 'topdown',
                              # 'robot_locations': [[1., 0.02]],
                              'vision_size': (64, 64),
                              'balls_num': 1,
                              # 'balls_locations': [[0., 0.]]
                              })

env = gym.make('Safexp-Ball-v0')

# Seed
np.random.seed(42)
env.seed(42)

# Settings for Dataset
start_index = 0
num_samples = 1000
path = './data/train'
os.makedirs(path, exist_ok=True)

for s in tqdm(range(start_index, start_index+num_samples)):
    frames = []
    obs = env.reset()
    action = get_action(env)

    for i in range(100):
        if i % 2 == 0:
            frames.append((np.transpose(obs['vision'], (1, 2, 0)) * 255).astype('uint8'))
        action = np.array([0., 0.]) if i > 0 else action
        obs, reward, done, info = env.step(action)

    # Save frames
    frames = np.stack(frames)
    np.save(os.path.join(path, str(s)), frames)

    # Make video
    make_movie(frames, path, str(s), fps=5)

pass
