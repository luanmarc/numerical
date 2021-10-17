import numpy as np
from numerical.interpolate.matrix import Tridiagonal, SquareMatrix


def main():
    # Some idiot tests for the module
    print('testing `Matrix`:')
    c = np.array([[ 4.,  1,  0,  0,  0,  0],
                  [-3,  10, -1,  0,  0,  0],
                  [ 0,  7,  30,  1,  0,  0],
                  [ 0,  0, -6, 90,  2,  0],
                  [ 0,  0,  0,  2,  5,  15],
                  [0,   0,  0,  0,  3.,  40]])
    v = np.array([5., 7, 9, 10, 7, 8])
    sol = Tridiagonal.solve(c, v)
    print(sol)
    x = np.array([[2., 6., 8.], [4., 11., 10.], [5., 8., 7.]])
    v = np.array([5., 7., 8.])
    SquareMatrix.gaussian_elim(x, v)
    print('DONE!!')
    print('matrix x =\n{}'.format(x))
    print('vector v =\n{}'.format(v))
    y = np.array([
        [7., 8., 9., 19., 6],
        [6, 3, 2, 20, 9],
        [10, 7, 11, 22, 8],
        [3, 7, 8, 33, 2],
        [4, 5, 6, 8, 3]])
    t = np.array([5., 7., 8., 8, 10])
    SquareMatrix.gaussian_elim(y, t)
    print('DONE!!')
    print('matrix y =\n{}'.format(y))
    print('vector t =\n{}'.format(t))


if __name__ == '__main__':
    main()
