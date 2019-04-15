import numpy as np
import sys
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {
    "4x12": [
             "RRRRRRRRRRRR",
             "RRRRRRRRRRRR",
             "RRRRRRRRRRRR",
             "SCCCCCCCCCCG",
            ],
}


# Adapted from https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
class WindyCliffWalkingEnv(discrete.DiscreteEnv):
    """
    This is a simple implementation of the Gridworld Cliff reinforcement learning task.

    The cliff is windy, however, so the agent is sometimes pushed down

    Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/the-book-2nd.html

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py

    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center

    Step, fall, and goal rewards are configurable. Stepping or falling onto the cliff results in
    a reset to the start. An episode terminates when the agent reaches the goal.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc = None, map_name = '4x12', wind_prob=0.1, step_rew=-1, fall_rew=-100, goal_rew=100):

        self.map_name = map_name
        if desc is None and self.map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[self.map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.shape = (len(self.desc), len(self.desc[1]))
        self.wind_prob = wind_prob
        self.step_rew = step_rew
        self.fall_rew = fall_rew
        self.goal_rew = goal_rew

        nS = np.prod(self.shape)
        nA = 4

        # Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/windy_gridworld.py
        # Wind probabilities & strengths
        winds = np.zeros(self.shape)
        # Winds that move the agend down 1 position
        winds[:, [3, 4, 5, 8]] = 1 * (np.random.uniform(0.0, 1.0) <= min(self.wind_prob, 1.0))
        # Winds that move the agend down 2 positions
        winds[:, [6, 7]] = 2 * (np.random.uniform(0.0, 1.0) <= min(self.wind_prob, 1.0))

        # Start Location
        self._start_state_index = 0
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            if self.desc[position] == b'S':
                self._start_state_index = np.ravel_multi_index(position, self.shape)

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            if self.desc[position] == b'C':
                self._cliff[position] = True

        # Terminal States
        self._terminal_state = np.zeros(self.shape, dtype=np.bool)
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            if self.desc[position] == b'G':
                self._terminal_state[position] = True

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # Calculate initial state distribution
        isd = np.zeros(nS)
        isd[self._start_state_index] = 1.0

        super(WindyCliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """

        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """

        # Get the new position
        new_position = np.array(current) + np.array(delta) + (np.array([1, 0]) * winds[tuple(current)])
        new_position = self._limit_coordinates(new_position).astype(int)

        # Get the new state
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        # Check if new position is a terminal state
        if self._terminal_state[new_position[0], new_position[1]]:
            return [(1.0, new_state, self.goal_rew, True)]

        # Check if new position is a cliff position; if so, the agent is moved to the start state
        if self._cliff[tuple(new_position)]:
            return [(1.0, self._start_state_index, self.fall_rew, False)]

        # Not a terminal state or a cliff position
        return [(1.0, new_state, self.step_rew, False)]

    def render(self, mode='human'):

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // 12, self.s % 4
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Up", "Right", "Down", "Left"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def colors(self):

        return {
            b'S': 'black',
            b'R': 'blue',
            b'C': 'darkred',
            b'G': 'green',
        }

    def directions(self):

        return {
            0: '⬆',
            1: '➡',
            2: '⬇',
            3: '⬅',
            4: '',
        }

    def new_instance(self):
        return WindyCliffWalkingEnv(desc=self.desc, map_name=self.map_name, wind_prob=self.wind_prob, \
                                    step_rew=self.step_rew, fall_rew=self.fall_rew, goal_rew=self.goal_rew)

