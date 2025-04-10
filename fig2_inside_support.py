"""Figure 2.

"""

import sys
import numpy as np
sys.path.append("./CrossMI")
from functions import calc_and_plot

np.random.seed(111)

N = np.random.normal

n = 2000  # Number of baseline samples
m = 500   # Number of test samples
W = 1     # Block length for stats testing

X = N(1, 0.25, size=m); Y = N(0.5*X, 0.05, size=m)

x = N(size=n); y = N(size=n)
calc_and_plot(x, y, X, Y, W=W, filename="plots/fig2a.png")

x = N(size=n); y = N(0.5*x, 0.5, size=n)
calc_and_plot(x, y, X, Y, W=W, filename="plots/fig2b.png")

x = N(size=n); y = N(0.5*x, 0.1, size=n)
calc_and_plot(x, y, X, Y, W=W, filename="plots/fig2c.png")
