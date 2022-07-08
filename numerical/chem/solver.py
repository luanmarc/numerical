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

State = list[int]
Reaction = list[int]
Realisation = list[float]
Time = float
Propensity = Callable[[State], float]

def ssa(x0: State, r: list[Reaction], a: list[Propensity],
        cond: Callable[[State, Time, int], bool], t0: float=0) -> State:
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
