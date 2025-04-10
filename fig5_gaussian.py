"""Gaussian cross MI.

"""

import numpy as np
import jpype as jp
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import SymLogNorm

plt.rc("text", usetex=True)

np.random.seed(111)

N = np.random.normal


def calc_gauss_local_cross_mi(x, y, mq, sq):
    mqx = mq[0]
    mqy = mq[1]
    sqxx2 = sq[0, 0]
    sqyy2 = sq[1, 1]
    sqxy = sq[0, 1]
    sqxx = np.sqrt(sqxx2)
    sqyy = np.sqrt(sqyy2)
    detsq = np.linalg.det(sq)
    dx = x - mqx
    dy = y - mqy
    return (
        np.log(sqxx * sqyy / np.sqrt(detsq))
        + 0.5 * (1 / sqxx2 - sqyy2 / detsq) * dx**2
        + 0.5 * (1 / sqyy2 - sqxx2 / detsq) * dy**2
        + (sqxy / detsq) * dx * dy
    )


def plot_heatmap(cmi, x, y, filename):
    plt.scatter(x, y, c="black", label="Reference", zorder=1)
    plt.legend(fontsize=18)
    vmax = np.max([abs(np.min(cmi)), abs(np.max(cmi))])
    im = plt.imshow(
        cmi.T,
        origin="lower",
        cmap="coolwarm",
        interpolation="nearest",
        extent=[-5, 5, -5, 5],
        norm=SymLogNorm(linthresh=1, linscale=1, vmin=-vmax, vmax=vmax),
        #vmin=-vmax,
        #vmax=vmax,
        alpha=0.85,
        zorder=2,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("$CI_{pq}$", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("$x$", fontsize=22)
    plt.ylabel("$y$", fontsize=22)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


x = N(size=2000)
y = N(size=2000)
# mq = [np.mean(x), np.mean(y)]
# sq = np.cov([x, y])
mq = np.array([0, 0])
sq = np.array([[1, 0], [0, 1]])

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)

cmi = np.zeros([X.shape[0], Y.shape[0]])
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        cmi[i, j] = calc_gauss_local_cross_mi(X[i], Y[j], mq, sq)

plot_heatmap(cmi, x, y, filename="plots/sfig3a.png")

x = N(size=2000)
y = N(0.5 * x, 0.5, size=2000)
mq = [np.mean(x), np.mean(y)]
sq = np.cov([x, y])

cmi = np.zeros([X.shape[0], Y.shape[0]])
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        cmi[i, j] = calc_gauss_local_cross_mi(X[i], Y[j], mq, sq)

plot_heatmap(cmi, x, y, filename="plots/sfig3b.png")

x = N(size=2000)
y = N(0.5 * x, 0.1, size=2000)
mq = [np.mean(x), np.mean(y)]
sq = np.cov([x, y])

cmi = np.zeros([X.shape[0], Y.shape[0]])
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        cmi[i, j] = calc_gauss_local_cross_mi(X[i], Y[j], mq, sq)

plot_heatmap(cmi, x, y, filename="plots/sfig3c.png")
