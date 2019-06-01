import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    [
        "                  ",
        "                  ",
        "                  ",
        "                  ",
        "               XX ",
        "               XX ",
        "               XX ",
        "      XX          ",
        "      XX          ",
        "      XX          ",
        "                  ",
        "   XX             ",
        "   XX             ",
        "   XX             ",
        "   XX             ",
        "                  ",
        "                  ",
        "G                 ",
    ],
    [
        "                  ",
        "                  ",
        "                  ",
        "                  ",
        "               XX ",
        "               XX ",
        "     XXXX   XXXXX ",
        "     XXXX   XXXX  ",
        "     XXXX   XXXX  ",
        "      XX          ",
        "                  ",
        "   XX             ",
        "   XXXXXX         ",
        "   XXXXXX         ",
        "   XXXXXX         ",
        "                  ",
        "                  ",
        "G                 ",
    ],
    [
        "                  ",
        "                  ",
        "                  ",
        "                  ",
        "               XX ",
        "       XX     XXX ",
        "     XXXX   XXXXX ",
        "     XXXX   XXXX  ",
        "     XXXX   XXXXX ",
        "      XX          ",
        "           XX     ",
        "   XX   XX XX     ",
        "   XXXXXXX XX     ",
        "   XXXXXXX        ",
        "   XXXXXX         ",
        "                  ",
        "                  ",
        "G                 ",
    ],
]


class GridWorldEnv(discrete.DiscreteEnv):
    """
    The Grid World Problem

    Using a simple 18×18 grid-world environment. The red circle represents a fixed goal state which the agent
    must reach. Furthermore, there are several 'slippery' rectangular patches (shaded in the left panel) which span
    several states. At every state, the agent must choose from four deterministic actions: up, down, right, left. Each
    transition from a normal state causes the agent to receive a reward of -1. If the agent transits out of a `slippery'
    state, the reward is uniformly distributed in the interval [-12, 10]. If the agent attempts to move out of the
    grid-world, the state stays unchanged and it receives a reward of -10. An episode starts with the agent randomly
    placed in the grid and stops when either the agent reaches the goal, or 150 steps have been taken without success.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.locs = locs = [(17, 0)]

        self.size = 18

        num_rows = self.size
        num_columns = self.size
        num_states = num_rows * num_columns
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 4
        P = [{state: {action: [] for action in range(num_actions)} for state in range(num_states)} for fidelity in
             range(3)]
        for fidelity in range(3):
            for row in range(num_rows):
                for col in range(num_columns):
                    state = self.encode(row, col)
                    initial_state_distrib[state] += 1
                    for action in range(num_actions):
                        import random
                        reward = (lambda: random.uniform(-12, 10)) if MAP[fidelity][row][col] == "X" else (lambda: -1)
                        if action == 0:
                            new_state = self.encode(min(num_rows - 1, row + 1), col)
                        elif action == 1:
                            new_state = self.encode(max(0, row - 1), col)
                        elif action == 2:
                            new_state = self.encode(row, min(num_columns - 1, col + 1))
                        else:
                            new_state = self.encode(row, max(0, col - 1))
                        # border
                        if new_state == state:
                            reward = lambda: -10
                        if state == self.encode(locs[0][0], locs[0][1]):
                            done = True
                            reward = lambda: 120
                            new_state = None
                        else:
                            done = False
                        P[fidelity][state][action].append((1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

        self.fidelity_supported = True

    def encode(self, row, col):
        # (5) 5, 5, 4
        i = row
        i *= self.size
        i += col
        return i

    def decode(self, i):
        return reversed([i % self.size, i // self.size])

    def render(self, mode='human', fidelity=0):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = np.asarray(MAP[fidelity], dtype='c').tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]

        def ul(x):
            return "_" if x == " " else x

        if self.s is not None:
            row, col = self.decode(self.s)
            out[row][col] = utils.colorize(ul(out[row][col]), 'green', highlight=True)
        outfile.write('____________________\n')
        outfile.write("\n".join(['|' + "".join(row) + '|' for row in out]) + "\n")
        outfile.write('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n')
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West"][self.lastaction]))
        else:
            outfile.write("\n")
        if self.lastfidelity is not None:
            outfile.write("  (Fidelity: {})\n".format(self.lastfidelity))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
