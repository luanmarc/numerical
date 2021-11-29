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


def example2():
    print(f"{' Example 2 ':-^79}")

    year = np.array(
        [1900.0, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
    )
    pop = np.array([
        75995.0, 91972, 105711, 123203, 131669, 150697,
        179323, 203212, 226505, 249633, 281422,
    ])

    mat = np.ndarray((11, 4))

    for i in range(11):
        mat[i, 0] = 1
        mat[i, 1] = (year[i] - 1950) / 50
        mat[i, 2] = ((year[i] - 1950) / 50) ** 2
        mat[i, 3] = ((year[i] - 1950) / 50) ** 3

    sol = Qr.solve(mat, pop)
    npsol, a, b, c = np.linalg.lstsq(mat, pop, rcond=None)
    print(f"solution:\n{sol}\n")
    print(f"numpy solution:\n{npsol}\n\n\n")

    value_at_2010 = sol[0]
    for i in range(1, 4):
        value_at_2010 += sol[i] * ((6 / 5) ** i)

    print(f"Approximation for the population in 2010:\n{value_at_2010}\n\n\n")


def example3():
    print(f"{' Example 3 ':-^79}")

    x_coor = np.array(
        [1.02, 0.95, 0.87, 0.77, 0.67, 0.56, 0.44, 0.3, 0.16, 0.01]
    )
    y_coor = np.array(
        [0.39, 0.32, 0.27, 0.22, 0.18, 0.15, 0.13, 0.12, 0.13, 0.15]
    )

    mat = np.ndarray((10, 5))
    f = np.array([-1., -1, -1, -1, -1, -1, -1, -1, -1, -1])

    for i in range(10):
        mat[i, 0] = x_coor[i] ** 2
        mat[i, 1] = x_coor[i] * y_coor[i]
        mat[i, 2] = y_coor[i] ** 2
        mat[i, 3] = x_coor[i]
        mat[i, 4] = y_coor[i]

    sol = Qr.solve(mat, f)
    npsol, a, b, c = np.linalg.lstsq(mat, f, rcond=None)
    print(f"solution:\n{sol}\n")
    print(f"numpy solution:\n{npsol}\n\n\n")


# TODO: find some interesting application of qr factorization algorithm
def example4():
    print(f"{' Example 4 ':-^79}")
    print("TODO: implement")


def main():
    example1()
    example2()
    example3()
    example4()


if __name__ == "__main__":
    main()
