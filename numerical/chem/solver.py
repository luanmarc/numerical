"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Luan Marc Suquet Camargo      (nUSP: 11809090)
         Francisco Barbosa             (nUSP: 6672901)

Computacao IV (CCM): Chemistry project
"""
from typing import Callable
import matplotlib.pyplot as plt
from random import random
import math


def plot(ts, xs, ytitle: str, title: str):
    c = ["green", "brown", "blue", "orange"]
    font = { "family": "monospace", "size": 20 }

    # Plot the evolution of each species
    kinds = len(xs[0])
    for k in range(kinds):
       plt.plot(ts, [x[k] for x in xs], color=c[k])

    plt.title(title, font)
    plt.xlabel("time", font)
    plt.ylabel(ytitle, font)
    plt.show()


def ssa(
    x0: list[int],
    r: list[list[int]],
    a: list[Callable[[list[int]], float]],
    title: str,
    tspan: float=50,
    t0: float=0
):
    """Stochastic simulation algorithm:
    `x0`: initial state of the system
    `r`: list of state-changing reactions
    `a`: list of callable propensity functions
    `tspan`: time span for the simulation (optional, defaults to 50)
    `t0`: starting time (optional, defaults to 0)
    """
    N = len(x0) # Number of species
    M = len(r) # Number of possible reactions
    ts = [t0] # Time recording
    xs = [x0] # State recording
    def _ssa(x: list[int], t: float):
        """Auxiliar recursive function for ssa"""
        ax = [_a(x) for _a in a]
        e1, e2 = (random(), random())
        sum_ax = sum(ax)

        # j0: index of the occured reaction
        j0, acc = 0, 0
        for j in range(M):
            # Calculates the least index satisfying acc > e1 * sumprop
            acc = acc + ax[j]
            if acc > e1 * sum_ax:
                j0 = j
                break

        # Time taken for the reaction j0 to occur
        tau = math.log(1/e2) / sum_ax

        # Update both the state and the current time
        x = [x[j] + r[j0][j] for j in range(N)]
        t = t + tau
        xs.append(x)
        ts.append(t)

        if t - t0 < tspan:
            _ssa(x, t)

    print("Calculating stochastic interactions over time...")
    _ssa(x0, t0)
    print("Plotting data-points for stochastic model...")
    plot(ts, xs, "Nr. molecules", title)


def elmaru(
    x0: list[int],
    r: list[list[int]],
    a: list[Callable[[list[int]], float]],
    vol: float,
    title: str,
    tspan: float=50,
    L: int=250
):
    """Euler-Maruyama method for Chemical Langevin Equation
    `x0`: initial state of the system
    `r`: list of state-changing reactions
    `a`: list of callable propensity functions
    `vol`: volume of the system
    `tspan`: time span for the simulation (optional, defaults to 50)
    `L`: number of time steps (optional, defaults to 250)
    """
    C = 6.023e23 * vol # Used to convert from #molecules -> concentration
    N = len(x0) # Number of species
    M = len(r) # Number of reactions
    tau = tspan / L # Fixed time step

    # Initial setup
    xs = [x0] # State recording
    ys = [[x0[j]/C for j in range(N)]] # Records concentrations
    ts = [n * tau for n in range(L + 1)] # Time records

    # Simulations
    for _ in range(L):
        # Propensities for the current realisation
        axs = [_a(xs[-1]) for _a in a]

        # Coefficients for each type of reaction
        d = [
            tau * axs[j] + math.sqrt(abs(tau * axs[j])) * random()
            for j in range(M)
        ]

        rsum = []
        for i in range(N):
            si = 0
            for j in range(M):
                si = si + d[j] * r[j][i]
            rsum.append(si)

        last = xs[-1]
        xs.append([last[j] + rsum[j] for j in range(N)])
        ys.append([xs[-1][j]/C for j in range(N)])

    print("Plotting the evolution of the Euler-Maruyama model...")
    plot(ts, xs, "Nr. molecules", title)
    plot(ts, ys, "concentration", title)
