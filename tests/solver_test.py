"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Luan Marc Suquet Camargo      (nUSP: 11809090)
         Francisco Barbosa             (nUSP: 6672901)

Computacao IV (CCM): Chemistry project
"""
from numerical.chem.solver import ssa, elmaru
import matplotlib.pyplot as plt
import numpy as np


# Font configuration
font = { "family": "monospace", "size": 15 }
_font = { "family": "monospace", "size": 10 }
title_font = { "family" : "monospace", "size": 20, "weight": "bold" }

# Colors for CLI text
RUN = "\033[1;32m::\033[0m"
QST = "\033[1;35m=>\033[0m"
PLOT = "\033[1;34m->\033[0m"
TITL = "\033[1;35m"
RES = "\033[0m"


def dimkin():
    """Dimerisation kinetics: example 1"""
    print(f"\n{TITL}{' Dimerisation kinetics ':-^79}{RES}\n")
    ls = ["P", "$P_2$"]
    x0 = [301, 0]
    vol = 1e-15
    r = [
        [-2,  1], # Dimerisation
        [ 2, -1] # Dissociation
    ]
    c1, c2 = 1.66e-3, 0.2

    def dim(x):
        """Dimerisation propensity"""
        return c1 * x[0] * (x[0] - 1)/2

    def diss(x):
        """Dissociation propensity"""
        return c2 * x[1]

    print(f"{RUN} Calculating and plotting the evolution over 10s (SSA)\n")
    ssa(x0, r, [dim, diss], vol, title="Dimerisation kinetics: SSA", tspan=10,
        labels=ls, conc=True)

    print(f"{RUN} Calculating and plotting the evolution"
          " over 10s (Euler-Maruyama)\n")
    elmaru(x0, r, [dim, diss], vol, tspan=10, L=500, labels=ls,
           title="Dimerisation kinetics: Euler-Maruyama")


    print(f"{RUN} Running 20 simulation with SSA...")
    for _ in range(20):
        ts, xs, _ = ssa(x0, r, [dim, diss], vol, tspan=10, _plot=False)
        plt.plot(ts, [x[0] for x in xs], color="black", label="P")
    print(f"{PLOT} Plotting the overlay of 20 runs for the protein...\n")
    plt.title("Protein evolution: 20 runs", title_font)
    plt.ylim(0, x0[0])
    plt.xlabel("Time", font)
    plt.ylabel("Nr. molecules", font)
    plt.legend(["P"], prop=_font)
    plt.show()

    runs = 10000
    print(f"{RUN} Running {runs} stochastic simulations...")
    ps = []
    for _ in range(runs):
        _, xs, _ = ssa(x0, r, [dim, diss], vol, tspan=10, _plot=False)
        ps.append(xs[-1][0])

    print(f"{PLOT} Plotting the density histogram of the protein"
          " at time t = 10...\n")
    plt.hist(ps, bins=int(185/5), density=True,
             edgecolor="black", color="orange")
    plt.title("Density histogram of P: SSA", title_font)
    plt.xlabel("P(10)", font)
    plt.ylabel("Density", font)
    plt.show()

    runs = 1000
    print(f"{RUN} Running {runs} Euler-Maruyama simulations...")
    _L = 500
    _tspan = 10
    tau = _tspan / _L
    ts = np.array([n * tau for n in range(_L)])
    pem = np.zeros(_L)
    for _ in range(runs):
        _, xs, _ = elmaru(x0, r, [dim, diss], vol,
                          tspan=_tspan, L=_L, _plot=False)
        xs = np.array(xs)
        for t in range(_L):
            pem[t] += xs[t, 0]
    pem *= 1/runs # Take mean

    # Calculate standard deviation
    std_run = 3
    rec = np.zeros((std_run, _L))
    for run in range(std_run):
        _, xs, _ = elmaru(x0, r, [dim, diss], vol,
                          tspan=_tspan, L=_L, _plot=False)
        xs = np.array(xs)
        for t in range(_L):
            rec[run, t] = xs[t, 0]
    std = np.zeros(_L)
    for t in range(_L):
        std[t] = np.std(rec[:, t])

    print(f"{PLOT} Plotting sample mean for the protein evolution...")
    plt.plot(ts, pem, color="orange")
    plt.plot(ts, pem + std, color="magenta")
    plt.plot(ts, pem - std, color="green")
    plt.ylim(0, x0[0])
    plt.ylabel("Nr. molecules", font)
    plt.xlabel("Time", font)
    plt.legend(
        ["Mean of P",
         f"Mean of P + {std_run} samples std. deviation",
         f"Mean of P - {std_run} samples std. deviation"],
        prop=_font
    )
    plt.show()


def michmen():
    """Michael-Menten: example 2"""
    print(f"\n{TITL}{' Michael-Menten ':-^79}{RES}\n")
    ls = ["Substrate", "Enzyme", "Complex", "Product"]
    x0 = [301, 120, 0, 0] # Initial number of molecules
    r = [
        [-1, -1,  1, 0], # Binding
        [ 1,  1, -1, 0], # Dissociation
        [ 0,  1, -1, 1]  # Conversion
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

    print(f"{RUN} Calculating SSA simulation and plotting...\n")
    ssa(x0, r, [b, d, c], vol, labels=ls, conc=True,
        title="Michael-Menten: SSA")

    print(f"{RUN} Calculating Euler-Maruyama simulation and plotting...")
    elmaru(x0, r, [b, d, c], vol, labels=ls,
           title="Michael-Menten: Euler-Maruyama")


def arn():
    """Auto-regulatory network"""
    print(f"\n{TITL}{' Auto-regulatory network ':-^79}{RES}\n")
    ls = ["Gene", "P2.Gene", "RNA", "P", "P2"]
    x0 = [10, 0, 0, 0, 0]
    c1, c1r, c2, c3, c4, c4r, c5, c6 = 1, 10, 0.02, 10, 1, 1, 0.1, 0.01
    r = [
        [-1,  1,  0,  0, -1], # Repression binding
        [ 1, -1,  0,  0,  1], # Reverse repression binding
        [ 0,  0,  1,  0,  0], # Transcription
        [ 0,  0,  0,  1,  0], # Translation
        [ 0,  0,  0, -2,  1], # Dimerisation
        [ 0,  0,  0,  2, -1], # Dissociation
        [ 0,  0, -1,  0,  0], # RNA degeneration
        [ 0,  0,  0, -1,  0]  # Protein degeneration
    ]

    def rb(x):
        """Repression binding:
        Gene + P2 -> P2.Gene
        """
        return c1 * x[0] * x[-1]

    def rrb(x):
        """Reverse repression binding:
        P2.Gene -> Gene + P2"""
        return c1r * x[1]

    def trsc(x):
        """Transcription:
        Gene -> Gene + RNA
        """
        return c2 * x[0]

    def trans(x):
        """Translation:
        RNA -> RNA + P
        """
        return c3 * x[2]

    def dim(x):
        """Dimerisation:
        P + P -> P2
        """
        return c4 * 0.5 * x[-2] * (x[-2] - 1)

    def diss(x):
        """Dissociation:
        P2 -> P + P
        """
        return c4r * x[-1]

    def rnadeg(x):
        """RNA degeneration:
        RNA -> nothing
        """
        return c5 * x[2]

    def pdeg(x):
        """Protein degeneration:
        P -> nothing
        """
        return c6 * x[-2]

    def constraint_time(ts: list[float], tfinal: float):
        """Auxiliary function for contraining time arrays:
        Returns the array containing all elments of `ts` less than or equal to
        `tfinal`. Assumes that `ts` is sorted.
        """
        _ts = []
        i = 0
        t = ts[i]
        while t <= tfinal:
            _ts.append(t)
            i += 1
            t = ts[i]
        return _ts


    a = [rb, rrb, trsc, trans, dim, diss, rnadeg, pdeg]

    print(f"{RUN} Calculating evolution of the network through"
          " a time span of 5000s (SSA)...")
    ts, xs, _ = ssa(x0, r, a, tspan=5000, labels=ls, _plot=False)
    xs = np.array(xs)
    plt.figure()

    print(f"{PLOT} Plotting evolution for time in [0, 5000]...")
    plt.subplot(321)
    plt.plot(ts, xs[:, 2], color="orange")
    plt.xlabel("Time", _font)
    plt.ylabel("RNA", _font)

    plt.subplot(323)
    plt.plot(ts, xs[:, 3], color="purple")
    plt.xlabel("Time", _font)
    plt.ylabel("P", _font)

    plt.subplot(325)
    plt.plot(ts, xs[:, 4], color="blue")
    plt.xlabel("Time", _font)
    plt.ylabel("P2", _font)

    print(f"{PLOT} Plotting evolution for time in [0, 250]...")
    _ts = constraint_time(ts, 250)

    plt.subplot(322)
    plt.plot(_ts, xs[:len(_ts), 2], color="orange")
    plt.xlabel("Time", _font)
    plt.ylabel("RNA", _font)

    plt.subplot(324)
    plt.plot(_ts, xs[:len(_ts), 3], color="purple")
    plt.xlabel("Time", _font)
    plt.ylabel("P", _font)

    plt.subplot(326)
    plt.plot(_ts, xs[:len(_ts), 4], color="blue")
    plt.xlabel("Time", _font)
    plt.ylabel("P2", _font)

    plt.suptitle("Auto-regulatory genetic network over 5000s (SSA)",
                 fontproperties=title_font)
    plt.show()

    print(f"{PLOT} Plotting evolution of P over 10s...\n")
    _ts = constraint_time(ts, 10)
    plt.plot(_ts, xs[:len(_ts), 3], color="purple")
    plt.title("Evolution of P over 10s", title_font)
    plt.ylabel("Nr. molecules", font)
    plt.xlabel("Time", font)
    plt.show()

    runs = 10000
    print(f"{RUN} Running {runs} simulations over 10s (SSA)...")
    ps = []
    for _ in range(runs):
        _, xs, _ = ssa(x0, r, a, tspan=10, labels=ls, _plot=False)
        ps.append(xs[-1][-2])

    print(f"{PLOT} Plotting density histogram of P over 10s...\n")
    plt.hist(ps, bins=int(185/5), density=True,
             edgecolor="black", color="orange")
    plt.title("Density histogram of P over 10s", title_font)
    plt.xlabel("P(10)", font)
    plt.ylabel("Density", font)
    plt.show()


def main():
    dimkin()
    michmen()
    arn()


if __name__ == "__main__":
    main()
