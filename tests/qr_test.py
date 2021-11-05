"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc Suquet Camargo      (nUSP: 11809090)

Computacao III (CCM): EP 3 QR Factorization test
"""
from numerical.matrix.linear_sys import Qr
import numpy as np


def example1():
    print(f"{' Example 1 ':-^79}")
    mat = np.array([
        [1., 2, -1, 3],
        [2, 3, 9, 16],
        [-2, 0, 4, -1],
        [2, 11, 5, 3],
        [0, 2, -1, 4],
        [1, -3, 1, 7],
    ])
    vec = np.array([1., 0, 1, 0, -1, 1])
    sol = Qr.solve(mat, vec)
    npsol, a, b, c = np.linalg.lstsq(mat, vec, rcond=None)
    print(f"solution:\n{sol}\n")
    print(f"numpy solution:\n{npsol}\n\n\n")


def main():
    example1()


if __name__ == "__main__":
    main()
