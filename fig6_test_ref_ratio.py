"""Generate simulation results.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.append("./CrossMI")
from functions import calc

np.random.seed(123)

N = np.random.normal
Unif = np.random.uniform

plt.rc('text', usetex=True)


def plot_distribution(x, y, X, Y, filename):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Reference")
    ax.scatter(X, Y, label="Test")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-5, 4)
    ax.set_xlabel("$x$", fontsize=26)
    ax.set_ylabel("$y$", fontsize=26)
    ax.legend(loc=3, fontsize=22)
    ax.tick_params("both", labelsize=26)
    plt.tight_layout()
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()


def plot_scatter(M, I1, I2, DI, CI, filename):
    fig, ax = plt.subplots()
    ax.scatter(M, I1, label="$I_q$")
    ax.plot(M, I1, linestyle="--")
    ax.scatter(M, I2, label="$I_p$")
    ax.plot(M, I2, linestyle="--")
    ax.scatter(M, DI, label="$\Delta I_{pq}$")
    ax.plot(M, DI, linestyle="--")
    ax.scatter(M, CI, label="$CI_{pq}$")
    ax.plot(M, CI, linestyle="--")
    ax.set_xlabel("Number of test data samples", fontsize=22)
    ax.set_ylabel("Measure (nats)", fontsize=22)
    ax.legend(loc=3, fontsize=22)
    ax.tick_params("both", labelsize=24)
    plt.tight_layout()
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()

x = N(size=2000); y = N(size=2000)
X = N(1.5, 0.4, size=200); Y = N(0.5*X, 0.1, size=200)
plot_distribution(x, y, X, Y, filename="plots/sfig2ai.png")

M = [50, 100, 200, 500, 1000]
I1, I2, DI, CI = [], [], [], []
for m in M:
    X = N(1.5, 0.4, size=m)
    Y = N(X, 0.1, size=m)

    i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
    print(m, i1, i2, di, ci)
    I1.append(i1)
    I2.append(i2)
    DI.append(di)
    CI.append(ci)

plot_scatter(M, I1, I2, DI, CI, filename="plots/sfig2aii.png")

p = Unif(0, 2*np.pi)
X = N(1, 0.5, size=200); Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=200)
plot_distribution(x, y, X, Y, filename="plots/sfig2bi.png")

I1, I2, DI, CI = [], [], [], []
for m in M:
    X = N(1, 0.5, size=m)
    Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=m)

    i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
    print(m, i1, i2, di, ci)
    I1.append(i1)
    I2.append(i2)
    DI.append(di)
    CI.append(ci)

plot_scatter(M, I1, I2, DI, CI, filename="plots/sfig2bii.png")

x = N(size=2000); y = N(0.5*x, 0.1, size=2000)
X = N(1.5, 0.4, size=200); Y = N(0.5*X, 0.1, size=200)
plot_distribution(x, y, X, Y, filename="plots/sfig2ci.png")

M = [50, 100, 200, 500, 1000]
I1, I2, DI, CI = [], [], [], []
for m in M:
    X = N(1.5, 0.4, size=m)
    Y = N(X, 0.1, size=m)

    i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
    print(m, i1, i2, di, ci)
    I1.append(i1)
    I2.append(i2)
    DI.append(di)
    CI.append(ci)

plot_scatter(M, I1, I2, DI, CI, filename="plots/sfig2cii.png")

p = Unif(0, 2*np.pi)
X = N(1, 0.5, size=200); Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=200)
plot_distribution(x, y, X, Y, filename="plots/sfig2di.png")

I1, I2, DI, CI = [], [], [], []
for m in M:
    X = N(1, 0.5, size=m)
    Y = N(np.sin(2*np.pi*X + p) + 1, 0.05, size=m)

    i1, i2, di, ci = calc(x, y, X, Y, W=1, concat=True)
    print(m, i1, i2, di, ci)
    I1.append(i1)
    I2.append(i2)
    DI.append(di)
    CI.append(ci)

plot_scatter(M, I1, I2, DI, CI, filename="plots/sfig2dii.png")
