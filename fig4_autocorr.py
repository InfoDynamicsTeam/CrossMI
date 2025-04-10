"""Stats testing with autocorrelated data.

"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from cycler import cycler
sys.path.append("./CrossMI")
from functions import calc_and_plot

np.random.seed(555)

N = np.random.normal

n = 100
ex = N(0, 0.5, n)
ey = N(0, 0.5, n)
t = np.arange(n)
x = np.zeros(n)
y = np.zeros(n)
for t_ in t[1:]:
    x[t_] = 0.8 * x[t_ - 1] + ex[t_]
    y[t_] = 0.3 * y[t_ - 1] + 0.9 * x[t_ - 1] + ey[t_]

ex = N(0, 0.5, n)
ey = N(0, 0.5, n)
t = np.arange(n)
X = np.zeros(n)
Y = np.zeros(n)
for t_ in t[1:]:
    X[t_] = 0.8 * X[t_ - 1] + ex[t_]
    Y[t_] = 0.9 * X[t_ - 1] + ey[t_]

calc_and_plot(x, y, X, Y, W=5, filename="plots/sfig1c.png")

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
ax[0].scatter(t, x)
ax[0].plot(t, x, linestyle="--")
ax[0].set_title("Data")
ax[0].set_xlabel("$t$", fontsize=18)
ax[0].set_ylabel("$x$", fontsize=18)
ax[0].tick_params("both", labelsize=17)
tsaplots.plot_acf(x, ax=ax[1])
ax[1].set_xlabel("Lag", fontsize=18)
plt.tight_layout()
plt.savefig("plots/sfig1ai.png")
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
ax[0].scatter(t, y)
ax[0].plot(t, y, linestyle="--")
ax[0].set_title("Data")
ax[0].set_xlabel("$t$", fontsize=18)
ax[0].set_ylabel("$y$", fontsize=18)
ax[0].tick_params("both", labelsize=17)
tsaplots.plot_acf(y, ax=ax[1])
ax[1].set_xlabel("Lag", fontsize=17)
plt.tight_layout()
plt.savefig("plots/sfig1aii.png")

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:orange'])

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
ax[0].scatter(t, X)
ax[0].plot(t, X, linestyle="--")
ax[0].set_title("Data")
ax[0].set_xlabel("$t$", fontsize=18)
ax[0].set_ylabel("$x$", fontsize=18)
ax[0].tick_params("both", labelsize=17)
tsaplots.plot_acf(X, ax=ax[1])
ax[1].set_xlabel("Lag", fontsize=18)
plt.tight_layout()
plt.savefig("plots/sfig1bi.png")
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
ax[0].scatter(t, Y)
ax[0].plot(t, Y, linestyle="--")
ax[0].set_title("Data")
ax[0].set_xlabel("$t$", fontsize=18)
ax[0].set_ylabel("$y$", fontsize=18)
ax[0].tick_params("both", labelsize=17)
tsaplots.plot_acf(Y, ax=ax[1])
ax[1].set_xlabel("Lag", fontsize=18)
plt.tight_layout()
plt.savefig("plots/sfig1bii.png")

