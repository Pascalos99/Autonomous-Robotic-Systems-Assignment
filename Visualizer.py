
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        self.xy_delta = 0.025

    def show_rosenbrock_function(self):
        a = 0
        b = 100

        x = np.arange(-1.25, 1.25, self.xy_delta)
        y = np.arange(-3.00, 3.00, self.xy_delta)
        X, Y = np.meshgrid(x, y)

        Z = (a - x)**2 + b * (Y - X**2)**2

        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, 200, vmin=0, vmax=1000, cmap=mpl.colormaps['jet'])
        plt.show()

    def show_rastrigin_function(self):
        x = np.arange(-5.12, 5.12, self.xy_delta)
        y = np.arange(-5.12, 5.12, self.xy_delta)
        X, Y = np.meshgrid(x, y)

        Z = [self.rastrigin_formula(X_i, Y_i) for (X_i, Y_i) in zip(X, Y)]

        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, 200, vmin=0, vmax=80, cmap=mpl.colormaps['jet'])
        plt.show()

    def rastrigin_formula(self, x, y):
        n = 2 # number of dimensions
        return 10 * n + ((x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)))


if __name__ == '__main__':
    viz = Visualizer()
    viz.show_rastrigin_function()