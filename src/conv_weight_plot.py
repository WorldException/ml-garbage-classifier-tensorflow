import matplotlib.pyplot as plt
import math

class WeightPlotter:
    def __init__(self, nrows, ncols):
        plt.ion()
        self.ax = None
        self.nrows = nrows
        self.ncols = ncols

    def plot(self, w):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols)

        for r in range(0, self.nrows):
            for c in range(0, self.ncols):
                self.ax[r, c].axis('off')
                self.ax[r, c].imshow(w[:, :, 0, (c * self.nrows) + r])

        self.fig.canvas.draw()