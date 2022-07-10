"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Luan Marc Suquet Camargo      (nUSP: 11809090)
         Francisco Barbosa             (nUSP: 6672901)

Computacao IV (CCM): Chemistry project
"""
from numerical.chem.solver import ssa, elmaru
import matplotlib.pyplot as plt
import numpy as np


font = { "family": "monospace", "size": 20 }


def dimkin():
    ls = ["P", "$P_2$"]
    x0 = [301, 0]
    N = 2
    vol = 1e-15
    r = [
        [-2, 1], # Dimerisation
        [2, -1] # Dissociation
    ]
    c1, c2 = 1.66e-3, 0.2

    def dim(x):
        """Dimerisation propensity"""
        return c1 * x[0] * (x[0] - 1)/2

    def diss(x):
        """Dissociation propensity"""
        return c2 * x[1]

    ssa(x0, r, [dim, diss], vol, title="Dimerisation kinetics: SSA", tspan=10, labels=ls)
    elmaru(x0, r, [dim, diss], vol,
           title="Dimerisation kinetics: Euler-Maruyama", tspan=10, L=500, labels=ls)

    print("Plotting the overlay of 20 runs for the protein...")
    for _ in range(20):
        ts, xs, _ = ssa(x0, r, [dim, diss], vol, tspan=10, _plot=False)
        plt.plot(ts, [x[0] for x in xs], color="black", label="P")
    plt.title("Protein evolution: 20 runs")
    plt.ylim(0, x0[0])
    plt.xlabel("Time")
    plt.ylabel("Nr. molecules")
    plt.show()

    runs = 10000
    print(f"Running {runs} stochastic simulations...")
    ps = []
    for _ in range(runs):
        _, xs, _ = ssa(x0, r, [dim, diss], vol, tspan=10, _plot=False)
        ps.append(xs[-1][0])

    print("Plotting the density histogram of the protein at time t = 10...")
    plt.hist(ps, bins=int(185/5), density=True,
             edgecolor="black", color="orange")
    plt.title("Density histogram of P: SSA")
    plt.xlabel("P(10)", font)
    plt.ylabel("Density", font)
    plt.show()

    runs = 1000
    print(f"Running {runs} Euler-Maruyama simulations...")
    _L = 500
    _tspan = 10
    tau = _tspan / _L
    ts = np.array([n * tau for n in range(_L)])
    pem = np.zeros(_L)
    for _ in range(runs):
        _, xs, _ = elmaru(x0, r, [dim, diss], vol, tspan=_tspan, L=_L, _plot=False)
        xs = np.array(xs)
        for t in range(_L):
            pem[t] += xs[t, 0]
    pem *= 1/runs # Take mean

    # Calculate standard deviation
    std_run = 3
    rec = np.zeros((std_run, _L))
    for run in range(std_run):
        _, xs, _ = elmaru(x0, r, [dim, diss], vol, tspan=_tspan, L=_L, _plot=False)
        xs = np.array(xs)
        for t in range(_L):
            rec[run, t] = xs[t, 0]
    std = np.zeros(_L)
    for t in range(_L):
        std[t] = np.std(rec[:, t])

    print("Plotting sample mean for the protein evolution...")
    plt.plot(ts, pem, color="orange")
    plt.plot(ts, pem + std, color="magenta")
    plt.plot(ts, pem - std, color="green")
    plt.ylim(0, x0[0])
    plt.legend(
        ["Mean of P", f"Mean of P + {std_run} samples std. deviation",
         f"Mean of P - {std_run} samples std. deviation"])
    plt.show()


def michmen():
    ls = ["Substrate", "Enzyme", "Complex", "Product"]
    x0 = [301, 120, 0, 0] # Initial number of molecules
    r = [
        [-1, -1, 1, 0], # Binding
        [1, 1, -1, 0], # Dissociation
        [0, 1, -1, 1] # Conversion
    ]
    vol = 1e-15 # Volume
    c1, c2, c3 = 1.66e-3, 1e-4, 0.1 # Reaction constants

    def b(x):
        """Binding propensity"""
        return c1 * x[0] * x[1]

    def d(x):
        """Dissociation propensity"""
        return c2 * x[2]

    def c(x):
        """Conversion propensity"""
        return c3 * x[2]

    ssa(x0, r, [b, d, c], vol, title="Michael-Menten: SSA", labels=ls)

    elmaru(x0, r, [b, d, c], vol, title="Michael-Menten: Euler-Maruyama", labels=ls)


def main():
    dimkin()
    # michmen()

if __name__ == "__main__":
    main()
