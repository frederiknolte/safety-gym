#!/usr/bin/env python

import gym
import gym.spaces
import numpy as np
from itertools import product
from copy import deepcopy
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import mujoco_py
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

from safety_gym.envs.world import World, Robot

import sys


# Distinct colors for different types of objects.
# For now this is mostly used for visualization.
# This also affects the vision observation, so if training from pixels.
# COLOR_agent = [[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.]]
COLOR_ball = [[1., 0.05, 0.05, 1.], [0.05, 1., 0.05, 1.], [0.05, 0.05, 1., 1.], [0.156, 0.768, 0.56, 1.], [0.6, 0.8, 0.045, 1.]]  # np.array([0, 1, 1, 1])
COLOR_WALL = np.array([.92, .92, .92, 0])
COLOR_plane = [[0.8, 0.8, 0.8], [0.77, 0.56, 0.47], [0.22, 0.27, 0.47]]

# Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
# We use these for raycasting lidar, where there are different lidar types.
# These work by turning "on" the group to see and "off" all the other groups.
# See obs_lidar_natural() for more.
GROUP_WALL = 2
GROUP_ball = 4

# Constant for origin of world
ORIGIN_COORDINATES = np.zeros(3)

# Constant defaults for rendering frames for humans (not used for vision)
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256

class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def quat2mat(quat):
    ''' Convert Quaternion to a 3x3 Rotation Matrix using mujoco '''
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco_py.functions.mju_quat2Mat(m, q)
    return m.reshape((3,3))


def quat2zalign(quat):
    ''' From quaternion, extract z_{ground} dot z_{body} '''
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat
    return a**2 - b**2 - c**2 + d**2


class Engine(gym.Env, gym.utils.EzPickle):

    '''
    Engine: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.

    '''

    # Default configuration (this should not be nested since it gets copied)
    DEFAULT = {
        'num_steps': 1000,  # Maximum number of environment steps in an episode

        'action_noise': 0.0,  # Magnitude of independent per-component gaussian action noise

        'placements_extents': [-2, -2, 2, 2],  # Placement limits (min X, min Y, max X, max Y)
        'placements_margin': 0.0,  # Additional margin added to keepout when placing objects

        # Floor
        'floor_display_mode': False,  # In display mode, the visible part of the floor is cropped

        # Robot
        'robot_placements': None,  # Robot placements list (defaults to full extents)
        'robot_locations': [],  # Explicitly place robot XY coordinate
        'robot_keepout': 0.4,  # Needs to be set to match the robot XML used
        'robot_base': 'xmls/ball.xml',  # Which robot XML to use as the base
        'robot_rot': None,  # Override robot starting angle
        'robot_size': 0.4,

        # Starting position distribution
        'randomize_layout': True,  # If false, set the random seed before layout to constant
        'build_resample': True,  # If true, rejection sample from valid environments
        'terminate_resample_failure': True,  # If true, end episode when resampling fails,
                                             # otherwise, raise a python exception.
        # TODO: randomize starting joint positions

        # Observation flags - some of these require other flags to be on
        # By default, only robot sensor observations are enabled.
        'observation_flatten': False,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        'observe_vision': False,  # Observe vision from the robot
        'observe_pos': False,  # Observe positions of robot and balls
        'observe_size': False,  # Observe sizes of robot and balls
        'observe_color': False,  # Observe colors of robot and balls
        # These next observations are unnormalized, and are only for debugging

        # Render options
        'render_labels': False,

        # Vision observation parameters
        'vision_size': (60, 40),  # Size (width, height) of vision observation; gets flipped internally to (rows, cols) format
        'vision_render': True,  # Render vision observation in the viewer
        'vision_render_size': (300, 200),  # Size to render the vision in the viewer
        'camera_name': 'vision',  # Name of the camera that is used for rendering the observations (!= the rendering for human)

        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        # Walls - barriers in the environment not associated with any constraint
        # NOTE: this is probably best to be auto-generated than manually specified
        'walls_num': 0,  # Number of walls
        'walls_placements': None,  # This should not be used
        'walls_locations': [],  # This should be used and length == walls_num
        'walls_keepout': 0.0,  # This should not be used
        'walls_size': np.array([0.5, 0.5, 1.]),  # Should be fixed at fundamental size of the world
        'walls_color': np.array([.5, .5, .5, 1]),  # Object color

        # balls
        'balls_num': 0,  # Number of balls in the world
        'balls_placements': None,  # balls placements list (defaults to full extents)
        'balls_locations': [],  # Fixed locations to override placements
        'balls_keepout': 0.4,  # Radius of balls keepout for placement
        'balls_size': 0.4,  # Half-size (radius) of ball object
        'balls_density': 1000.,  # Density of balls
        'balls_sink': 0.,  # Experimentally measured, based on size and density,
                             # how far balls "sink" into the floor.
        # Mujoco has soft contacts, so balls slightly sink into the floor,
        # in a way which can be hard to precisely calculate (and varies with time)
        # Ignore some costs below a small threshold, to reduce noise.
        'balls_contact_cost': 1.0,  # Cost (per step) for being in contact with a ball
        'balls_displace_cost': 0.0,  # Cost (per step) per meter of displacement for a ball
        'balls_displace_threshold': 1e-3,  # Threshold for displacement being "real"
        'balls_velocity_cost': 1.0,  # Cost (per step) per m/s of velocity for a ball
        'balls_velocity_threshold': 1e-4,  # Ignore very small velocities

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip)
        'frameskip_binom_p': 1.0,  # Probability of trial return (controls distribution)

        '_seed': None,  # Random state seed (avoid name conflict with self.seed)
    }

    def __init__(self, config={}):
        # First, parse configuration. Important note: LOTS of stuff happens in
        # parse, and many attributes of the class get set through setattr. If you
        # are trying to track down where an attribute gets initially set, and 
        # can't find it anywhere else, it's probably set via the config dict
        # and this parse function.
        self.parse(config)
        gym.utils.EzPickle.__init__(self, config=config)

        # Load up a simulation of the robot, just to figure out observation space
        self.robot = Robot(self.robot_base)

        self.action_space = gym.spaces.Box(-1, 1, (self.robot.nu,), dtype=np.float32)
        self.build_observation_space()
        self.build_placements_dict()

        self.viewer = None
        self.world = None
        self.clear()

        self.seed(self._seed)
        self.done = True

    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    @property
    def sim(self):
        ''' Helper to get the world's simulation instance '''
        return self.world.sim

    @property
    def model(self):
        ''' Helper to get the world's model instance '''
        return self.sim.model

    @property
    def data(self):
        ''' Helper to get the world's simulation data instance '''
        return self.sim.data

    @property
    def robot_pos(self):
        ''' Helper to get current robot position '''
        return [self.data.get_body_xpos('robot').copy()]

    @property
    def balls_pos(self):
        ''' Helper to get the list of ball positions '''
        return [self.data.get_body_xpos(f'ball{p}').copy() for p in range(self.balls_num)]

    @property
    def walls_pos(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.data.get_body_xpos(f'wall{i}').copy() for i in range(self.walls_num)]

    @property
    def balls_color(self):
        ''' Helper to get the hazards positions from layout '''
        return [self.color_balls[f'ball{i}'].copy() for i in range(self.balls_num)]

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()

        if self.observe_sensors:
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                if sensor in self.robot.sensor_dim:
                    dim = self.robot.sensor_dim[sensor]
                else:
                    dim = self.robot.sensor_dim[sensor.split('_')[0]]
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
            # Velocities don't have wraparound effects that rotational positions do
            # Wraparounds are not kind to neural networks
            # Whereas the angle 2*pi is very close to 0, this isn't true in the network
            # In theory the network could learn this, but in practice we simplify it
            # when the sensors_angle_components switch is enabled.
            for sensor in self.robot.hinge_vel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            for sensor in self.robot.ballangvel_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
            # Angular positions have wraparound effects, so output something more friendly
            if self.sensors_angle_components:
                # Single joints are turned into sin(x), cos(x) pairs
                # These should be easier to learn for neural networks,
                # Since for angles, small perturbations in angle give small differences in sin/cos
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
                # Quaternions are turned into 3x3 rotation matrices
                # Quaternions have a wraparound issue in how they are normalized,
                # where the convention is to change the sign so the first element to be positive.
                # If the first element is close to 0, this can mean small differences in rotation
                # lead to large differences in value as the latter elements change sign.
                # This also means that the first element of the quaternion is not expectation zero.
                # The SO(3) rotation representation would be a good replacement here,
                # since it smoothly varies between values in all directions (the property we want),
                # but right now we have very little code to support SO(3) roatations.
                # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32)
            else:
                # Otherwise include the sensor without any processing
                # TODO: comparative study of the performance with and without this feature.
                for sensor in self.robot.hinge_pos_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
                for sensor in self.robot.ballquat_names:
                    obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        if self.observe_vision:
            width, height = self.vision_size
            rows, cols = height, width
            self.vision_size = (rows, cols)
            obs_space_dict['vision'] = gym.spaces.Box(0, 1.0, (3,) + self.vision_size, dtype=np.float32)
        # Flatten it ourselves
        if self.observe_pos:
            obs_space_dict['robot_pos'] = gym.spaces.Box(-np.inf, np.inf, (1, 3,), dtype=np.float32)
            obs_space_dict['balls_pos'] = gym.spaces.Box(-np.inf, np.inf, (self.balls_num, 3,), dtype=np.float32)
        if self.observe_size:
            obs_space_dict['robot_size'] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            obs_space_dict['balls_size'] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        if self.observe_color:
            obs_space_dict['robot_color'] = gym.spaces.Box(-np.inf, np.inf, (1, 4,), dtype=np.float32)
            obs_space_dict['balls_color'] = gym.spaces.Box(-np.inf, np.inf, (self.balls_num, 4,), dtype=np.float32)
        self.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(obs_space_dict)

    def toggle_observation_space(self):
        self.observation_flatten = not(self.observation_flatten)
        self.build_observation_space()

    def placements_from_location(self, location, keepout):
        ''' Helper to get a placements list from a given location and keepout '''
        x, y = location
        return [(x - keepout, y - keepout, x + keepout, y + keepout)]

    def placements_dict_from_object(self, object_name):
        ''' Get the placements dict subset just for a given object name '''
        placements_dict = {}
        if hasattr(self, object_name + 's_num'):  # Objects with multiplicity
            plural_name = object_name + 's'
            object_fmt = object_name + '{i}'
            object_num = getattr(self, plural_name + '_num', None)
            object_locations = getattr(self, plural_name + '_locations', [])
            object_placements = getattr(self, plural_name + '_placements', None)
            object_keepout = getattr(self, plural_name + '_keepout')
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(self, object_name + '_locations', [])
            object_placements = getattr(self, object_name + '_placements', None)
            object_keepout = getattr(self, object_name + '_keepout')
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def build_placements_dict(self):
        ''' Build a dict of placements.  Happens once during __init__. '''
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))
        placements.update(self.placements_dict_from_object('wall'))

        if self.balls_num: #self.constrain_balls:
            placements.update(self.placements_dict_from_object('ball'))

        self.placements = placements

    def seed(self, seed=None):
        ''' Set internal random state seeds '''
        self._seed = np.random.randint(2**32) if seed is None else seed

    def build_layout(self):
        ''' Rejection sample a placement of objects to find a layout. '''
        if not self.randomize_layout:
            self.rs = np.random.RandomState(0)

        for _ in range(10000):
            if self.sample_layout():
                break
        else:
            raise ResamplingError('Failed to sample layout of objects')

    def sample_layout(self):
        ''' Sample a single layout, returning True if successful, else False. '''

        def placement_is_valid(xy, layout):
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                xy = self.draw_placement(placements, keepout)
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def constrain_placement(self, placement, keepout):
        ''' Helper function to constrain a single placement by the keepout radius '''
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    def draw_placement(self, placements, keepout):
        ''' 
        Sample an (x,y) location, based on potential placement areas.

        Summary of behavior: 

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify 
        rectangles in the XY-plane where an object could be placed. 

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y) 
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally, 
        randomly draw a uniform point within the selected rectangle.

        '''
        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert len(constrained), 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.rs.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        return np.array([self.rs.uniform(xmin, xmax), self.rs.uniform(ymin, ymax)])

    def random_rot(self):
        ''' Use internal random state to get a random rotation in radians '''
        return self.rs.uniform(0, 2 * np.pi)

    def build_world_config(self):
        ''' Create a world_config from our own config '''
        # TODO: parse into only the pieces we want/need
        world_config = {}

        world_config['robot_base'] = self.robot_base
        world_config['robot_xy'] = self.layout['robot']

        ball_colors_tmp = deepcopy(COLOR_ball)
        color_id = self.rs.randint(low=0, high=len(ball_colors_tmp))
        world_config['robot_rgba'] = ball_colors_tmp[color_id]
        self.robot_color = [ball_colors_tmp[color_id]]
        del ball_colors_tmp[color_id]

        world_config['robot_size'] = self.robot_size

        if self.robot_rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot_rot)

        if self.floor_display_mode:
            floor_size = max(self.placements_extents)
            world_config['floor_size'] = [floor_size + .1, floor_size + .1, 1]

        #if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        world_config['observe_vision'] = self.observe_vision

        # Extra objects to add to the scene
        world_config['objects'] = {}
        self.color_balls = {}
        if self.balls_num:
            for i in range(self.balls_num):
                color_id = self.rs.randint(low=0, high=len(ball_colors_tmp))
                name = f'ball{i}'
                object = {'name': name,
                          'size': np.ones(3) * self.balls_size,
                          'type': 'sphere',
                          'density': self.balls_density,
                          'mass': 0.01,
                          'pos': np.r_[self.layout[name], self.balls_size - self.balls_sink],
                          'rot': self.random_rot(),
                          'group': GROUP_ball,
                          'rgba': ball_colors_tmp[color_id]}
                world_config['objects'][name] = object
                self.color_balls[name] = ball_colors_tmp[color_id]
                del ball_colors_tmp[color_id]

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        if self.walls_num:
            for i in range(self.walls_num):
                name = f'wall{i}'
                geom = {'name': name,
                        'size': self.walls_size,
                        'pos': np.r_[self.layout[name], self.walls_size[-1]],
                        'rot': 0,
                        'type': 'box',
                        'group': GROUP_WALL,
                        'rgba': COLOR_WALL}
                world_config['geoms'][name] = geom

        # Extra mocap bodies used for control (equality to object of same name)
        world_config['mocaps'] = {}

        world_config['texplane'] = {'rgb_1': COLOR_plane[self.rs.randint(low=0, high=len(COLOR_plane))],
                                    'rgb_2': COLOR_plane[self.rs.randint(low=0, high=len(COLOR_plane))]}

        return world_config

    def clear(self):
        ''' Reset internal state for building '''
        self.layout = None

    def build(self):
        ''' Build a new physics simulation environment '''
        # Sample object positions
        self.build_layout()

        # Build the underlying physics world
        self.world_config_dict = self.build_world_config()

        if self.world is None:
            self.world = World(self.world_config_dict)
            self.world.reset(build=False)
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_config_dict, state=False)
        # Redo a small amount of work, and setup initial goal state

        # Save last action
        self.last_action = np.zeros(self.action_space.shape)

        # Save last subtree center of mass
        self.last_subtreecom = self.world.get_sensor('subtreecom')

    def reset(self):
        ''' Reset the physics simulation and return observation '''
        self._seed += 1  # Increment seed
        self.rs = np.random.RandomState(self._seed)
        self.done = False
        self.steps = 0  # Count of steps taken in this episode
        # Set the button timer to zero (so button is immediately visible)
        self.buttons_timer = 0

        self.clear()
        self.build()
        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)

        # Reset stateful parts of the environment
        self.first_reset = False  # Built our first world successfully

        # Return an observation
        return self.obs()

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position '''
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        robot_pos = self.world.robot_pos()
        return np.sqrt(np.sum(np.square(pos - robot_pos[:2])))

    def world_xy(self, pos):
        ''' Return the world XY vector to a position from the robot '''
        assert pos.shape == (2,)
        return pos - self.world.robot_pos()[:2]

    def ego_xy(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates

    def obs_vision(self):
        ''' Return pixels from the robot camera '''
        # Get a render context so we can
        rows, cols = self.vision_size
        width, height = cols, rows
        vision = self.sim.render(width, height, camera_name=self.camera_name, mode='offscreen')
        vision = np.array(vision, dtype='float32')[::-1, :, :] / 255
        return np.transpose(vision, (2, 0, 1))

    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_sensors:
            # Sensors which can be read directly, without processing
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.hinge_vel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballangvel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            # Process angular position sensors
            if self.sensors_angle_components:
                for sensor in self.robot.hinge_pos_names:
                    theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                    obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
                for sensor in self.robot.ballquat_names:
                    quat = self.world.get_sensor(sensor)
                    obs[sensor] = quat2mat(quat)
            else:  # Otherwise read sensors directly
                for sensor in self.robot.hinge_pos_names:
                    obs[sensor] = self.world.get_sensor(sensor)
                for sensor in self.robot.ballquat_names:
                    obs[sensor] = self.world.get_sensor(sensor)
        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        if self.observe_pos:
            obs['robot_pos'] = np.array(self.robot_pos)
            obs['balls_pos'] = np.array(self.balls_pos)
        if self.observe_size:
            obs['robot_size'] = np.array([self.robot_size,])
            obs['balls_size'] = np.array([self.balls_size,] * self.balls_num)
        if self.observe_color:
            obs['robot_color'] = np.array(self.robot_color)
            obs['balls_color'] = np.array(self.balls_color)
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
        return obs

    def update_layout(self):
        ''' Update layout dictionary with new places of objects '''
        self.sim.forward()
        for k in list(self.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k:
                continue
            self.layout[k] = self.data.get_body_xpos(k)[:2].copy()

    def buttons_timer_tick(self):
        ''' Tick the buttons resampling timer '''
        self.buttons_timer = max(0, self.buttons_timer - 1)

    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        action = np.array(action, copy=False)  # Cast to ndarray
        assert not self.done, 'Environment must be reset before stepping'

        info = {}

        # Set action
        action_range = self.model.actuator_ctrlrange
        # action_scale = action_range[:,1] - action_range[:, 0]
        self.data.ctrl[:] = np.clip(action, action_range[:,0], action_range[:,1]) #np.clip(action * 2 / action_scale, -1, 1)
        if self.action_noise:
            self.data.ctrl[:] += self.action_noise * self.rs.randn(self.model.nu)

        # Simulate physics forward
        exception = False
        for _ in range(self.rs.binomial(self.frameskip_binom_n, self.frameskip_binom_p)):
            try:
                self.sim.step()  # Physics simulation step
            except MujocoException as me:
                print('MujocoException', me)
                exception = True
                break
        if exception:
            self.done = True
            info['cost_exception'] = 1.0
        else:
            self.sim.forward()  # Needed to get sensor readings correct!

            # Button timer (used to delay button resampling)
            self.buttons_timer_tick()

        # Timeout
        self.steps += 1
        if self.steps >= self.num_steps:
            self.done = True  # Maximum number of steps in an episode reached

        return self.obs(), 0, self.done, info

    def render_area(self, pos, size, color, label='', alpha=0.1):
        ''' Render a radial area in the environment '''
        z_size = min(size, 0.3)
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(pos=pos,
                               size=[size, size, z_size],
                               type=const.GEOM_CYLINDER,
                               rgba=np.array(color) * alpha,
                               label=label if self.render_labels else '')

    def render_sphere(self, pos, size, color, label='', alpha=0.1):
        ''' Render a radial area in the environment '''
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(pos=pos,
                               size=size * np.ones(3),
                               type=const.GEOM_SPHERE,
                               rgba=np.array(color) * alpha,
                               label=label if self.render_labels else '')

    def render_swap_callback(self):
        ''' Callback between mujoco render and swapping GL buffers '''
        if self.observe_vision and self.vision_render:
            self.viewer.draw_pixels(self.save_obs_vision, 0, 0)

    def render(self,
               mode='human', 
               camera_id=None,
               width=DEFAULT_WIDTH,
               height=DEFAULT_HEIGHT
               ):
        ''' Render the environment to the screen '''

        if self.viewer is None or mode!=self._old_render_mode:
            # Set camera if specified
            if mode == 'human':
                self.viewer = MjViewer(self.sim)
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.type = const.CAMERA_FREE
            else:
                self.viewer = MjRenderContextOffscreen(self.sim)
                self.viewer._hide_overlay = True
                self.viewer.cam.fixedcamid = camera_id #self.model.camera_name2id(mode)
                self.viewer.cam.type = const.CAMERA_FIXED
            self.viewer.render_swap_callback = self.render_swap_callback
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            self._old_render_mode = mode
        self.viewer.update_sim(self.sim)

        if camera_id is not None:
            # Update camera if desired
            self.viewer.cam.fixedcamid = camera_id

        # Draw vision pixels
        if self.observe_vision and self.vision_render:
            vision = self.obs_vision()
            vision = np.array(vision * 255, dtype='uint8')
            vision = Image.fromarray(vision).resize(self.vision_render_size)
            vision = np.array(vision, dtype='uint8')
            self.save_obs_vision = vision

        if mode=='human':
            self.viewer.render()
        elif mode=='rgb_array':
            self.viewer.render(width, height)
            data = self.viewer.read_pixels(width, height, depth=False)
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()
            return data[::-1, :, :]