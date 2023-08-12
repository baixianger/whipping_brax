"""End to end training a whip to hit a flying target.

Based on the Xiang Bai's dm_control Whipping environment.
"""

from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp

class Reacher(PipelineEnv):
    # pyformat: disable
    """
    ### Description

    "Whipping" is based on franka panda and a lumped-­parameter whip. 
    The goal is to wield the whip to hit the flying target with a 0.05 radius under the position control. 
    The primary stage is to his a fixed trajectory. Future we might try to spawn the trajectory at a random setting

    ### Action Space
    The action space is a 7-dimensional continuous space that
    represents the target joint position applied at the joints of franka panda.
    | Num | Action         | Control Min | Control Max | Name   | Joint | Unit |
    |-----|----------------|-------------|-------------|--------|-------|------|
    | 0   | Position of A1 | -2.8973     | 2.8973      | joint0 | hinge |  rad |
    | 1   | Position of A2 | -1.7628     | 1.7628      | joint1 | hinge |  rad |
    | 2   | Position of A3 | -2.8973     | 2.8973      | joint2 | hinge |  rad |
    | 3   | Position of A4 | -3.0718     | -0.0698     | joint3 | hinge |  rad |
    | 4   | Position of A5 | -2.8973     | 2.8973      | joint4 | hinge |  rad |
    | 5   | Position of A2 | -­0.0175     | 3.7525      | joint5 | hinge |  rad |
    | 6   | Position of A7 | -2.8973     | 2.8973      | joint6 | hinge |  rad |

    ### Observation Space
    Observations consist of:
    1. The observed position series of the flying target
    2. The observed position series of the end of the whip
    3. The observed position series of the end of the whip

    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
    |-----|------------------------------------------------------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
    | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                    | hinge | unitless                 |
    | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                    | hinge | unitless                 |
    | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | cos(joint0)                    | hinge | unitless                 |
    | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | cos(joint1)                    | hinge | unitless                 |
    | 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                       | slide | position (m)             |
    | 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                       | slide | position (m)             |
    | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                         | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                         | hinge | angular velocity (rad/s) |
    | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                             | slide | position (m)             |
    | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                             | slide | position (m)             |
    | 10  | z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both) | -Inf | Inf | NA                             | slide | position (m)             |

    ### Rewards

    The reward consists of two parts:

    - *reward_dist*: This reward is a measure of how far the *whip-end* is from the target, 
    with a more negative value assigned for when the *whip-end* is further away from the target. 
    It is calculated as the negative vector norm of (position of the whipend - position of target), or *-norm("whipend" - "target")*.
    - *reward_ctrl*: 能量惩罚项目

    Unlike other environments, Reacher does not allow you to specify weights for
    the individual reward terms. However, `info` does contain the keys
    *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms, you
    should create a wrapper that computes the weighted reward from `info`.

    ### Starting State

    Frank panda and the whip's joints all start in the state descriped in the keyframe from the sepecified xml file
    with a noise added for stochasticity.  A uniform noise in the range [-0.01, 0.01] is added to the joint, 
    while the target position is selected uniformly at random in a sphere of radius 0.01 around the origin. 

    Independent, uniform noise in the range of [-0.005, 0.005] is added to the velocities, and the last element 
    ("whipend" - "target") is calculated at the end once everything is set.

    The control timestep is 0.02s and the physical timestep is 0.001.

    ### Episode Termination
    The episode terminates when any of the following happens:
    1. The episode duration reaches maximum duration of 1.5 seconds
    2. The *whip-end* is within 0.06 meters of the target

    ### Arguments
    No additional arguments are currently supported for this environment.
    """
    # pyformat: enable

    def __init__(self, backend='generalized', **kwargs):
        path = 'env/whip.xml'
        sys = mjcf.load_model(path)
        dt_sys = 0.001 # simulation step in brax system
        dt_ctrl = 0.02 # control step in this tailored env
        if backend in ['spring', 'positional']:
            sys = sys.replace(dt=dt_sys)
            sys = sys.replace(actuator=sys.actuator.replace(gear=jp.array([25.0, 25.0])))
        n_frames = dt_ctrl / dt_sys
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        self.mocap = jp.load('env/mocap.npy')

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state with some randomization."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # randomize the arm
        q_arm = self.sys.init_q[:7] + jax.random.uniform(rng1, (7,), minval=-0.01, maxval=0.01)

        # set the target's inital q_pos, q_vel
        _, q_pos, q_vel = self._random_target(rng)

        # get the initial q
        q = self.sys.init_q.at[:7].set(q_arm)
        q = q.at[-7:-4].set(q_pos)

        # get the initial qd
        qd = jp.zeros(self.sys.qd_size())
        qd = qd.at[-6:-3].set(q_vel)

        pipeline_state = self.pipeline_init(q, qd) # 会调用brax.generalized或者brax.positional或者brax.spring中的init函数

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_dist': zero,
            'reward_ctrl': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        # vector from tip to target is last 3 entries of obs vector
        reward_dist = -math.safe_norm(obs[-3:])
        reward_ctrl = -jp.square(action).sum()
        reward = reward_dist + reward_ctrl

        state.metrics.update(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Returns: Target's position and velocity.
                    Arm's position and velocity.
                    Whip's position and velocity.
                    Vector from tip to target.
                    """
        # Arm
        a_q = pipeline_state.q[:7]
        a_qd = pipeline_state.qd[:7]
        # Whip
        w_q = pipeline_state.q[-8:-7]
        w_qd = pipeline_state.qd[-7:-6]

        theta = pipeline_state.q[:2]
        target_pos = pipeline_state.x.pos[2]
        tip_pos = (
            pipeline_state.x.take(1)
            .do(base.Transform.create(pos=jp.array([0.11, 0, 0])))
            .pos
        )
        # tip_vel, instead of pipeline_state.qd[:2], leads to more sensible policies
        # for a randomly initialized policy network
        tip_vel = (
            base.Transform.create(pos=jp.array([0.11, 0, 0]))
            .do(pipeline_state.xd.take(1))
            .vel
        )
        tip_to_target = tip_pos - target_pos

        return jp.concatenate([
            jp.cos(theta),
            jp.sin(theta),
            pipeline_state.q[2:],  # target x, y
            tip_vel[:2],
            tip_to_target,
        ])

    def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns a target location and velocity."""
        rngs = jax.random.split(rng, 5)
        # target position and velocity
        mp_idx = jax.random.randint(rngs[1], 1, 0, len(self.mocap)-250)
        pos_offset = 0.05 * jax.random.uniform(rngs[2], (3,), minval=-1, maxval=1)    
        vel_offset = 0.05 * jax.random.uniform(rngs[3], (3,), minval=-1, maxval=1)
        ang_offset = jp.pi / 36 * jax.random.uniform(rngs[3], (1,), minval=-1, maxval=1)

        q_pos = self.mocap[mp_idx][1:4] + pos_offset
        _, y_vel, z_vel = self.mocap[mp_idx][4:] + vel_offset
        x_vel, y_vel = jp.sin(ang_offset) * y_vel, jp.cos(ang_offset) * y_vel
        q_vel = jp.array([x_vel, y_vel, z_vel])

        return rng, q_pos, q_vel