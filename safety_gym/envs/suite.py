#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from string import capwords
from gym.envs.registration import register
import numpy as np
from itertools import product


VERSION = 'v0'

ROBOT_NAMES = ('Point', 'Car', 'Doggo', 'Ball')
ROBOT_XMLS = {name: f'xmls/{name.lower()}.xml' for name in ROBOT_NAMES}
BASE_SENSORS = ['accelerometer', 'velocimeter']
MAKE_VISION_ENVIRONMENTS = False

#========================================#
# Helper Class for Easy Gym Registration #
#========================================#

class SafexpEnvBase:
    ''' Base used to allow for convenient hierarchies of environments '''
    def __init__(self, name='', config={}, prefix='Safexp'):
        self.name = name
        self.config = config
        self.robot_configs = {}
        self.prefix = prefix
        for robot_name in ROBOT_NAMES:
            robot_config = {}
            robot_config['robot_base'] = ROBOT_XMLS[robot_name]
            robot_config['sensors_obs'] = BASE_SENSORS
            self.robot_configs[robot_name] = robot_config

    def copy(self, name='', config={}):
        new_config = self.config.copy()
        new_config.update(config)
        return SafexpEnvBase(self.name + name, new_config)

    def register(self, name='', config={}):
        # Note: see safety_gym/envs/mujoco.py for an explanation why we're using
        # 'safety_gym.envs.mujoco:Engine' as the entrypoint, instead of
        # 'safety_gym.envs.engine:Engine'.
        for robot_name, robot_config in self.robot_configs.items():
            # Default
            env_name = f'{self.prefix}-{robot_name}{self.name + name}-{VERSION}'
            reg_config = self.config.copy()
            reg_config.update(robot_config)
            reg_config.update(config)
            register(id=env_name,
                     entry_point='safety_gym.envs.mujoco:Engine',
                     kwargs={'config': reg_config})
            if MAKE_VISION_ENVIRONMENTS:
                # Vision: note, these environments are experimental! Correct behavior not guaranteed
                vision_env_name = f'{self.prefix}-{robot_name}{self.name + name}Vision-{VERSION}'
                vision_config = {'observe_vision': True,
                                 'observation_flatten': False,
                                 'vision_render': True}
                reg_config = deepcopy(reg_config)
                reg_config.update(vision_config)
                register(id=vision_env_name,
                         entry_point='safety_gym.envs.mujoco:Engine',
                         kwargs={'config': reg_config})


vision_env_base = SafexpEnvBase('', {'observe_vision': True,
                                     'vision_render': False,
                                     'camera_name': 'topdown',
                                     'vision_size': (128, 128),
                                     'walls_num': 20,
                                     'walls_locations': [[x, y] for x, y in product([-2.5, 2.5], [-2, -1, 0, 1, 2])] +
                                                        [[x, y] for x, y in product([-2, -1, 0, 1, 2], [-2.5, 2.5])],
                                     'placements_extents': [-1.5, -1.5, 1.5, 1.5],
                                     'robot_rot': np.pi / 2,
                                     'num_steps': 1005,
                                     })
