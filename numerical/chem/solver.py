# TODO: Numero usp francisco
"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Luan Marc Suquet Camargo      (nUSP: 11809090)
         Francisco Barbosa             (nUSP: ?)

Computacao IV (CCM): Chemistry project
"""
from typing import Callable
import random
import math

State = list[int | float]
Reaction = list[int]
Time = float
Propensity = Callable[[State], float]
StopCondition = Callable[[State, Time, int], bool]

def ssa(x0: State, r: list[Reaction], a: list[Propensity],
        cond: StopCondition, t0: Time=0) -> State:
    """Stochastic simulation algorithm:
    `x0`: initial state of the system
    `r`: list of state-changing reactions
    `a`: list of callable propensity functions
    `cond`: callable stopping condition function
    `t0`: starting time (optional, if not set, we let `t0 = 0`)
    """
    def _ssa(x: State, t: Time) -> State:
        """Auxiliar recursive function for ssa"""
        ax = [_a(x) for _a in a]
        e1, e2 = (random.random(), random.random())
        sum_ax = sum(ax)

        # j0: index of the occured reaction
        j0, acc = 0, 0
        for j in range(len(ax)):
            # Calculates the least index satisfying acc > e1 * sumprop
            acc = acc + ax[j]
            if acc > e1 * sum_ax:
                j0 = j
                break

        # Time taken for the reaction j0 to occur
        tau = math.log(1/e2) / sum_ax

        # Update both the state and the current time
        x, t = [x[j] + r[j0][j] for j in range(len(x))], t + tau
        if not cond(x, t, j0):
            # If the stopping condition was not satisfied, continue
            _ssa(x, t)
        return x
    return _ssa(x0, t0)


def elmaru(x0: State, r: list[Reaction], a: list[Propensity],
           cond: StopCondition, L: int) -> State:
    """Euler-Maruyama method for Chemical Langevin Equation
    `x0`: initial state of the system
    `r`: list of state-changing reactions
    `a`: list of callable propensity functions
    `cond`: callable stopping condition function
    `N`: number of time steps
    `t0`: starting time (optional, if not set, we let `t0 = 0`)
    """
    N = len(x0)
    M = len(r)
    dt = 1/L # time step
    def _elmaru(x: State, n: int=0) -> State:
        z = [random.random() for _ in range(M)]
        tau = n * dt
        sqrt_tau = math.sqrt(tau)
        ax = [_a(x) for _a in a]
        sqrt_ax = [math.sqrt(_ax) for _ax in ax]

        y = [
            sum([tau * ax[j] * r[i][j] for j in range(M)])
            for i in range(N)
        ]
        sqy = [
            sum([sqrt_tau * sqrt_ax[j] * z[j] * r[i][j] for j in range(M)])
            for i in range(N)
        ]

        n = n + 1
        x = x + y + sqy
        if not cond(x, tau, n):
            _elmaru(x, n)
        return x

    return _elmaru(x0)
