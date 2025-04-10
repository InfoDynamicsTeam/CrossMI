"""Figure 1.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("./CrossMI")
from functions import calc_mi_and_do_stats, calc_cross_mi_and_do_stats

np.random.seed(444)

def save(filename):
    print("Saving", filename)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Simulate data
n = 1000

x1 = np.random.normal(0, 0.1, size=n)
y1 = np.random.normal(x1, 0.1, size=n)

x2 = np.random.normal(0.2, 0.1, size=n)
y2 = np.random.normal(x2, 0.1, size=n)

x3 = np.random.normal(0, 0.1, size=n)
y3 = np.random.normal(0, 0.1, size=n)

x4 = np.random.normal(0.2, 0.1, size=n)
y4 = np.random.normal(-x4, 0.1, size=n)

x = np.concatenate([x1, x2, x3, x4])
y = np.concatenate([y1, y2, y3, y4])

X = np.array([0.25])
Y = np.array([0.25])

# Plot time series
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,2))
ax[0].plot(x, lw=0.5, c="tab:blue")
ax[0].axis("off")
ax[1].axis("off")
save(filename="plots/fig1a_x.png")

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,2))
ax[1].plot(y, lw=0.5, c="tab:blue")
ax[0].axis("off")
ax[1].axis("off")
save(filename="plots/fig1a_y.png")

# Plot distributions
def plot_distribution(x, y, X, Y, filename):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Reference ($q$)")
    ax.scatter(X, Y, s=120, label="Test ($p$)")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("$x$", fontsize=30)
    ax.set_ylabel("$y$", fontsize=30)
    ax.tick_params("both", labelsize=26)
    ax.legend(fontsize=20, loc=3)
    I, p_I = calc_mi_and_do_stats(x, y, W=1)
    CI, p_CI = calc_cross_mi_and_do_stats(x, y, X, Y, W=1, shuffle_ref=True)
    table = f"\\begin{{tabular}}{{ c c c }} & Value & p-value \\\\ \hline $I_q$ & {I:.2f} & {p_I:.1g} \\\\ $CI_{{pq}}$ & {CI:.2f} & {p_CI:.1g} \\end{{tabular}}"
    plt.text(-1.1, 0.7, table, size=24)
    save(filename)

for i, (x_, y_) in enumerate(zip([x1, x2, x3, x4], [y1, y2, y3, y4])):
    plot_distribution(x_, y_, X, Y, f"plots/fig1b_state{i+1}.png")

plot_distribution(x, y, X, Y, f"plots/fig1c.png")
