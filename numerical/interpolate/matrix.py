'''
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Computacao III (CCM): EP 2 Cubic interpolating splines 
'''
import numpy as np


class Tridiagonal:
    '''A class for methods concerning tridiagonal matrices'''

    @classmethod
    def gaussian_elim(cls, mat, vec):
        '''In-place gaussian elimination algorithm.
        Supposes `mat` is diagonal dominant
        '''
        # Check user input
        n = len(vec)
        if mat.shape[1] != n:
            raise Exception('Lengths do not match')

        for i in range(1, n):
            mult = mat[i, i - 1] / mat[i - 1, i - 1]
            mat[i, i] -= mult * mat[i - 1, i]
            mat[i, i - 1] = 0
            vec[i] -= mult * vec[i - 1]


    @classmethod
    def back_substitution(cls, mat, vec):
        '''Back substitution method.
        Assumes `mat` is upper triangular.
        '''
        n = len(vec)
        sol = np.zeros(n)
        sol[-1] = vec[-1] / mat[-1, -1]
        for i in range(n - 2, -1, -1):
            sol[i] = (vec[i] - mat[i, i + 1] * sol[i + 1]) / mat[i, i]

        return sol


    @classmethod
    def solve(cls, coeff, res):
        '''Solves the system `coeff * X = res` for `X`.
        The algorithm is stable for diagonal dominant `coeff` matrices.
        '''
        # Triangularization of `coeff` and `res`
        cls.gaussian_elim(coeff, res)
        return cls.back_substitution(coeff, res)

def main():
    print('testing `Matrix`:')
    c = np.array([[ 4,  1,  0,  0,  0,  0],
                  [-3,  10, -1,  0,  0,  0],
                  [ 0,  7,  30,  1,  0,  0],
                  [ 0,  0, -6, 90,  2,  0],
                  [ 0,  0,  0,  2,  5,  15],
                  [0,   0,  0,  0,  3,  40]])
    v = np.array([5, 7, 9, 10, 7, 8])
    sol = Tridiagonal.solve(c, v)
    print(sol)

if __name__ == '__main__':
    main()
