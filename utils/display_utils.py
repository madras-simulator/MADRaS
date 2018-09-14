"""Display Utils."""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def plot_heatmap(x, y, z, nbins=100):
    """
    Heatmap Plotter.

    x: list
    y: list
    z: list
    """
    x_min, y_min = np.min(x), np.min(y)
    x_max, y_max = np.max(x), np.max(y)

    x_interval = (x_max - x_min) / nbins
    y_interval = (y_max - y_min) / nbins

    x_quantized = [np.int32(np.floor((x_i - x_min) / x_interval)) for x_i in x]
    y_quantized = [np.int32(np.floor((y_i - y_min) / y_interval)) for y_i in y]

    z_matrix = np.ones((nbins + 1, nbins + 1)) * (-50)

    idx = 0
    for x_i, y_i in zip(x_quantized, y_quantized):
        z_matrix[y_i, x_i] = z[idx]
        idx += 1

    plt.figure()
    plt.imshow(z_matrix, cmap="gnuplot", origin="lower", vmin=-50, vmax=150)
    plt.colorbar()


class bcolors:
    """Color Info."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKYELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class StructuredPrinter():
    """accumulates different entities and their values.

        prints them in a nice structured way"""

    def __init__(self, mode="episode"):
        """Init Method."""
        self.mode = mode  # "episode" or "step"
        self.data = {}  # dict in <field: value> format
        self.explore_mode = None

    def print_episode(self, episode_num=None):
        """Getting data ready."""
        fields = [key for key in self.data.keys()]
        values = [self.data[field] for field in fields]  # this makes values appear in the same sequence as the keys
        fields.insert(0, "EPISODE NO.")
        values.insert(0, episode_num)

        # Preparing table
        if episode_num == 0:
            self.table = PrettyTable(fields)  # Intialize table
        self.table.add_row(values)

        # Printing table
        os.system("clear")
        print(self.table)

    def print_step(self, step_num=None):
        """Step Printer."""
        fields = [key for key in self.data.keys()]
        values = [self.data[field] for field in fields]  # this makes values appear in the same sequence as the keys
        fields.insert(0, "STEP NO.")
        values.insert(0, step_num)
        row_format = "{:>25}" * (len(fields))
        if step_num == 0:
            print(bcolors.HEADER + "\n" + row_format.format(*fields), end="\n")
        if self.explore_mode == -1:
            print(bcolors.OKYELLOW + row_format.format(*values), end="\r")
        elif self.explore_mode == 0:
            print(bcolors.OKGREEN + row_format.format(*values), end="\r")
        elif self.explore_mode == 1:
            print(bcolors.OKBLUE + row_format.format(*values), end="\r")
        else:
            print(row_format.format(*values), end="\r")


if __name__ == "__main__":

    # # plot_heatmap() test case:
    # x = np.arange(10)/10.
    # y = np.arange(10)/10.
    # z = x+y
    # # print x
    # # print y
    # print z
    # plot_heatmap(x,y,z)
    # # =======================

    printer = StructuredPrinter()
    printer.data = {"A": 1, "B": 2, "C": 3}
    printer.print_episode(0)
    printer.data = {"A": 2, "B": 3, "C": 4}
    printer.print_episode(1)
    for i in range(10000000):
        printer.print_step(i)
