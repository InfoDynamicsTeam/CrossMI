"""Figure 3.

"""

import sys
import numpy as np
sys.path.append("./CrossMI")
from functions import calc_and_plot, calc_cross_mi

np.random.seed(123)

N = np.random.normal
Unif = np.random.uniform

n = 2000  # Number of baseline samples
m = 500   # Number of test samples
W = 1     # Block length for stats testing

# Test data outside of the range of the reference data
x = N(size=n); y = N(0.5*x, 0.1, size=n)
X = N(5, 0.1, size=m); Y = N(0.5*X, 0.05, size=m)

calc_and_plot(x, y, X, Y, W=W, filename="plots/fig3a.png")

# Get varability in the estimation
CI = []
for _ in range(10):
    x = N(size=n); y = N(0.5*x, 0.1, size=n)
    X = N(5, 0.1, size=m); Y = N(0.5*X, 0.05, size=m)
    CI.append(calc_cross_mi(x, y, X, Y))
std = np.std(CI)
print(f"Standard deviation: {std:.2f}\n")

# Relationship in the test data different to the reference data
p = Unif(0, 2*np.pi)
x = N(size=n); y = N(0.5*x, 0.1, size=n)
X = N(1, 0.5, size=m); Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=m)

calc_and_plot(x, y, X, Y, W=W, filename="plots/fig3b.png")
