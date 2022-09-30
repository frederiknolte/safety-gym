import os
import numpy as np
from tqdm import tqdm
import torch
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


def get_contacts(env):
    return env.contacts


def remove_duplicates(contacts):
    return [list(set(tuple(sorted(contact)) for contact in time_step)) for time_step in contacts]


# Create environment
balls_num = 3
vision_env_base.register('', {'num_steps': 10000,
                              'camera_name': 'topdown',
                              'vision_size': (64, 64),
                              'balls_num': balls_num,
                              'observe_vision': True,
                              'observe_pos': True,
                              'observe_size': True,
                              'observe_color': True,
                              'sensors_obs': ['accelerometer', 'velocimeter'] + [f'accelerometer_ball{i}' for i in range(balls_num)] +
                                                [f'velocimeter_ball{i}' for i in range(balls_num)],
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
    states = []
    contacts = [[]]
    obs = env.reset()
    action = get_action(env)
    action = np.tile(action, (balls_num+1,))

    for i in range(500):
        contacts[-1] += get_contacts(env)

        if i % 2 == 0:
            state_i = np.empty((env.balls_num + 1, 0))

            if env.observe_vision:
                frames.append((np.transpose(obs['vision'], (1, 2, 0)) * 255).astype('uint8'))
                # plt.imshow(frames[-1])
                # plt.show()
            if env.observe_pos:
                positions = np.concatenate([obs['robot_pos'][:, :-1], obs['balls_pos'][:, :-1]], axis=0)
                state_i = np.concatenate([state_i, positions], axis=-1)
            if env.observe_size:
                sizes = np.concatenate([obs['robot_size'], obs['balls_size']], axis=0)[:, None]
                state_i = np.concatenate([state_i, sizes], axis=-1)
            if env.observe_color:
                colors = np.concatenate([obs['robot_color'][:, :-1], obs['balls_color'][:, :-1]], axis=0)
                state_i = np.concatenate([state_i, colors], axis=-1)
            if 'accelerometer' in env.sensors_obs:
                accelerometer = np.concatenate([obs[accelerometer_name][None, :-1] for accelerometer_name in ['accelerometer'] + [f'accelerometer_ball{i}' for i in range(balls_num)]], axis=0)
                state_i = np.concatenate([state_i, accelerometer], axis=-1)
            if 'velocimeter' in env.sensors_obs:
                velocimeter = np.concatenate([obs[velocimeter_name][None, :-1] for velocimeter_name in ['velocimeter'] + [f'velocimeter_ball{i}' for i in range(balls_num)]], axis=0)
                state_i = np.concatenate([state_i, velocimeter], axis=-1)

            states.append(state_i)
            contacts.append([])

        action = np.array([0., 0.]*(balls_num+1)) if i > 0 else action
        obs, reward, done, info = env.step(action)

    contacts = remove_duplicates(contacts)

    # Save frames
    if env.observe_vision:
        frames = np.stack(frames)
        np.save(os.path.join(path, 'vision_' + str(s)), frames)
        # Make video
        make_movie(frames, path, str(s), fps=5)
    if env.observe_pos or env.observe_size or env.observe_color:
        states = np.stack(states)
        np.save(os.path.join(path, 'state_' + str(s)), states)

    torch.save(contacts, os.path.join(path, 'contacts_' + str(s) + '.pt'))
