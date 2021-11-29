"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc Suquet Camargo      (nUSP: 11809090)

Computacao III (CCM): EP 3 QR Factorization test
"""
from numerical.matrix.linear_sys import Qr
import numpy as np

class Col:
    QST = "\033[1;32m::\033[0m"
    SOL = "\033[1;35m=>\033[0m"
    INF = "\033[1;34m->\033[0m"
    WAR = "\033[4;33m"
    ERR = "\033[1;31m"
    TITL = "\033[1;35m"
    RES = "\033[0m"

def example1(path: str):
    print(f"\n{Col.TITL}{' Example 1 ':-^79}{Col.RES}\n")
    if not path:
        path = "qr_data/ex1.txt"
    file = open(path, "r")
    ls = file.readlines()

    # Read data from `file`
    shape = ls[0]
    m, n = map(int, shape.replace("\n", "").split(" "))
    data = np.zeros((m, n + 1))
    for i in range(1, m + 1):
        data[i - 1, :] = list(map(float, ls[i].replace("\n", "").split(" ")))

    # Solve system
    mat, vec = data[:, :-1], data[:, -1]
    sol = Qr.solve(mat, vec)
    res = vec - np.matmul(mat, sol)
    print(
        f"{Col.INF} Given the system (A b):\n{data}\n\n"
        f"{Col.SOL} The obtained solution is:\n{sol}\n\n"
        f"{Col.INF} Residue norm is: {np.linalg.norm(res)}"
    )


def example2(path: str):
    print(f"\n{Col.TITL}{' Example 2 ':-^79}{Col.RES}\n")
    if not path:
        path = "qr_data/ex2.txt"
    file = open(path, "r")
    ls = file.readlines()

    # Read input data from `file`
    m = int(ls[0].replace("\n", ""))
    data = np.zeros((m, 2))
    for i in range(1, m + 1):
        data[i - 1, :] = list(map(float, ls[i].replace("\n", "").split(" ")))
    year = data[:, 0]
    pop = data[:, 1]
    # TODO: find yy_i and insert it into the zip for printing!!
    print(
        f"{Col.INF} From the input we got the following data:\n"
        f"[(t_i, y_i, yy_i)] =\n{[x for x in zip(year, pop)]}\n\n"
    )

    mat = np.ndarray((m, 4))
    for i in range(m):
        mat[i, 0] = 1
        mat[i, 1] = (year[i] - 1950) / 50
        mat[i, 2] = ((year[i] - 1950) / 50) ** 2
        mat[i, 3] = ((year[i] - 1950) / 50) ** 3

    sol = Qr.solve(mat, pop)
    value_at_2010 = sol[0]
    for i in range(1, 4):
        value_at_2010 += sol[i] * ((6 / 5) ** i)
    res = pop - np.matmul(mat, sol)

    print(
        f"{Col.INF} Given the system (A b):\n{np.column_stack((mat, pop))}\n\n"
        f"{Col.INF} The obtained solution is:\n{sol}\n\n"
        f"{Col.INF} Residue norm is: {np.linalg.norm(res)}\n\n"
        f"{Col.SOL} Approximate population in 2010: {value_at_2010}"
    )



def example3(path: str):
    print(f"\n{Col.TITL}{' Example 3 ':-^79}{Col.RES}\n")
    if not path:
        path = "qr_data/ex3.txt"
    file = open(path, "r")
    ls = file.readlines()

    # Read input data from `file`
    m = int(ls[0].replace("\n", ""))
    data = np.zeros((m, 2))
    for i in range(1, m + 1):
        data[i - 1, :] = list(map(float, ls[i].replace("\n", "").split(" ")))
    x_coor = data[:, 0]
    y_coor = data[:, 1]

    mat = np.ndarray((m, 5))

    f = np.full(shape=m, fill_value=-1.0, dtype=float)
    for i in range(10):
        mat[i, 0] = x_coor[i] ** 2
        mat[i, 1] = x_coor[i] * y_coor[i]
        mat[i, 2] = y_coor[i] ** 2
        mat[i, 3] = x_coor[i]
        mat[i, 4] = y_coor[i]

    sol = Qr.solve(mat, f)
    res = f - np.matmul(mat, sol)
    print(
        f"{Col.INF} Given the system (A b):\n{np.column_stack((mat, f))}\n\n"
        f"{Col.SOL} The obtained solution is:\n{sol}\n\n"
        f"{Col.INF} Residue norm is: {np.linalg.norm(res)}"
    )


# TODO: find some interesting application of qr factorization algorithm
def example4(path: str):
    print(f"\n{Col.TITL}{' Example 4 ':-^79}{Col.RES}\n")
    if not path:
        path = "qr_data/ex4.txt"
    file = open(path, "r")
    ls = file.readlines()
    print(f"{Col.ERR}[*] TODO: implement{Col.RES}")


def main():
    print(f"{Col.TITL}{' QR factorization ':-^79}{Col.RES}")
    stop = False
    while not stop:
        ans = input(
            f"\n{Col.QST} Examples:\n"
            "   [1] Linear System;\n"
            "   [2] Populational growth;\n"
            "   [3] Planetary orbit;\n"
            f"   [4] {Col.WAR}****TODO****{Col.RES}\n"
            f"{Col.QST} Which one would you like to run? [1, 2, 3, 4] "
        )
        examples = {"1": example1, "2": example2, "3": example3, "4": example4}

        if ans in examples:
            arg = input(
                f"\n{Col.QST} Please provide the path for the input file\n"
                f"{Col.INF} "
            )
            examples[ans](arg)
        else:
            ans_err = input(
                f"\n{Col.ERR}[*] Unfortunately such example {ans} "
                f"is not available.{Col.RES}\n"
                f"{Col.QST} Would you like to quit? [Y/n] "
            )
            if not ans_err or ans_err in ["y", "Y"]:
                stop = True
            else:
                continue

        ans_ex = input(
            f"\n{Col.QST} Would you like to test another example? [Y/n] "
        )
        if ans_ex in ["n", "N"]:
            stop = True


if __name__ == "__main__":
    main()
