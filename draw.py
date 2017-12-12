import numpy as np
import random

from model import Model


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    res = []
    while n != 0:
        remainder = n % k
        n = n // k
        res += [remainder]
    return res

def from_base_k_to_decimal(k, nonDec):
    dec = 0
    result = 0
    for i in range(len(nonDec)):
        result += nonDec[i] * (k ** i)
    return int(result)

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 500)
        self.make_param('height', 300)
        self.make_param('density', 0.2)
        self.make_param('rule', 184, setter=self.setter_rule)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""

        res = decimal_to_base_k(self.rule,self.k)
        lengthOfRule = self.k ** (2*self.r+1)
        self.rule_set = [0 for i in range(lengthOfRule - len(res))]
        self.rule_set += res[::-1]

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        newState = []
        formedElement = from_base_k_to_decimal(self.k, inp[::-1])
        self.rr = decimal_to_base_k(formedElement, self.k)
        newState = self.rule_set[-formedElement - 1]
        return newState

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""

        initial = []
        carsNumber = round(self.width*self.density)
        # TODO: select random indexes
        initial = np.zeros(self.width)
        #initial[len(initial)//2] = random.randrange(1,self.k)
        for i in range(self.width):
            initial[i] += [random.randrange(0,self.k)]
        return initial

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('t = %d for rule = %d' % (self.t, self.rule))
        # save into file if max is reached


    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)


if __name__ == '__main__':
    sim = CASim()
    from gui import GUI
    cx = GUI(sim)
    cx.start()
