'''
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Computacao III (CCM): EP 2 Cubic interpolating splines
'''
import numpy as np
from numerical.interpolate.matrix import Tridiagonal, SquareMatrix


def decomposition(mat, vec):
    print('initial matrix =\n{}\n'.format(mat))
    SquareMatrix.gaussian_elim(mat, vec, True)
    print('Resulting mixed matrix =\n{}\n'.format(mat))
    lower = np.tril(mat)
    np.fill_diagonal(lower, 1)
    upper = np.triu(mat)
    print('upper =\n{}\nlower =\n{}\n'.format(upper, lower))
    prod = np.matmul(lower, upper)
    print('product = lower * upper =\n{}\n'.format(prod))


def main():
    print('>>> Testing tridiagonal gaussian elimination:\n')
    coeff = np.array([[ 4.,  1,  0,  0,  0,  0],
                      [-3,  10, -1,  0,  0,  0],
                      [ 0,  7,  30,  1,  0,  0],
                      [ 0,  0, -6, 90,  2,  0],
                      [ 0,  0,  0,  2,  5,  15],
                      [0,   0,  0,  0,  3.,  40]])
    vec = np.array([5., 7, 9, 10, 7, 8])
    sol = Tridiagonal.solve(coeff, vec)
    print('The solution for the system coeff * sol = vec, where\ncoeff '
        '=\n{}\nvec =\n{}\nis given by\nsol =\n{}\n'.format(coeff, vec, sol))

    print('\n\n>>> Gaussian elimination test:\n')
    y = np.array([
        [7., 8., 9., 19., 6],
        [6, 3, 2, 20, 9],
        [10, 7, 11, 22, 8],
        [3, 7, 8, 33, 2],
        [4, 5, 6, 8, 3]])
    t = np.array([5., 7., 8., 8, 10])
    print('Given a matrix\n{}\nand a vector\n{}\n'.format(y, t))
    SquareMatrix.gaussian_elim(y, t)
    print('The gaussian elimination yield matrix\n{}\nand'
        'vector\n{}\n'.format(y, t))

    print('\n\n>>> LU decomposition test:\n')
    y = np.array([
        [7., 8., 9., 19., 6],
        [6., 3, 2, 20, 9],
        [10., 7, 11, 22, 8],
        [3., 7, 8, 33, 2],
        [4., 5, 6, 8, 3]])
    t = np.array([5., 7., 8., 8, 10])
    decomposition(y, t)


if __name__ == '__main__':
    main()
