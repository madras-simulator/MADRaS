"""Noise Function."""
import numpy as np


class OU(object):
    """Ornstein Uhlenbeck Process."""

    def function(self, x, mu, theta, sigma):
        """The stochastic equation."""
        return theta * (mu - x) + sigma * np.random.randn(1)


if __name__ == "__main__":
    NUM_SAMPLES = 1000
    x = np.linspace(0, 10, NUM_SAMPLES)
    noise = OU()
    dx = np.asarray([noise.function(x_i, 5, 0.15, 0.1)
                    for x_i in x]).reshape((NUM_SAMPLES,))
    y = x + dx
    import matplotlib.pyplot as plt
    plt.plot(x, y, c='r')
    plt.plot(x, x)
    plt.show()
