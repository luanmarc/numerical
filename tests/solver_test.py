"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Luan Marc Suquet Camargo      (nUSP: 11809090)
         Francisco Barbosa             (nUSP: 6672901)

Computacao IV (CCM): Chemistry project
"""
from numerical.chem.solver import ssa, elmaru
import numpy as np

def dimkin():
    y0 = [5e-7, 0]
    vol = 1e-15
    def dimerisation():
        k1 = 5e5 # 2 P -> P2

    def dissociation():
        k2 = 0.2 # P2 -> 2 P

def michmen():
    y0 = [5e-7, 2e-7, 0, 0] # Initial concentrations
    x0 = [301, 120, 0, 0] # Initial number of molecules
    r = [
        [-1, -1, 1, 0], # Binding
        [1, 1, -1, 0], # Dissociation
        [0, 1, -1, 1] # Conversion
    ]
    vol = 1e-15 # Volume
    r1, r2, r3 = 1.66e-3, 1e-4, 0.1 # Reaction constants

    def b(x):
        return r1 * x[0] * x[1]
    def d(x):
        return r2 * x[2]
    def c(x):
        return r3 * x[2]

    ssa(x0, r, [b, d, c], "Michael-Menten: SSA")

    elmaru(x0, r, [b, d, c], vol, "Michael-Menten: Euler-Maruyama")


def main():
    michmen()

if __name__ == "__main__":
    main()
