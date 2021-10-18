"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Computacao III (CCM): EP 2 Cubic interpolating splines
"""
import numpy as np


class SquareMatrix:
    """Methods for square matrices"""

    @classmethod
    def gaussian_elim(cls, mat, vec, lu_decomp: bool = False):
        """General, in-place, gaussian elimination.
        If `lu_decomp` is set as `True`, the method will use the upper
        triangular part of `mat` for U and the lower part for L"""
        if mat.shape[0] != mat.shape[1]:
            raise Exception("Matrix not square")

        def pivot_selection(mat, col: int) -> int:
            """Partial pivot selection:
            Returns the row index of the pivot given a specific column."""
            pivot_row = 0
            for row in range(1, mat.shape[0]):
                if abs(mat[row, col]) > abs(mat[pivot_row, col]):
                    pivot_row = row
            if mat[pivot_row, col] == 0:
                raise Exception("The matrix is singular!")
            return pivot_row

        def switch_rows(mat, row0, row1):
            """In-place switch rows: `row0` and `row1`"""
            if row0 == row1:
                return
            for col in range(mat.shape[1]):
                aux = mat[row0, col]
                mat[row0, col] = mat[row1, col]
                mat[row1, col] = aux

        # For each column, select the `pivot`, switch rows if need be. For each
        # row below the `pivot_row`, subtract element-wise the multiple `mult`
        # in order to make the pivot the only non-zero element in the column.
        # Pivot selection is done only for the first diagonal element,
        # otherwise we simply use the current diagonal. This is acceptable if
        # `mat` is equilibrated.
        for diag in range(mat.shape[1]):
            if diag == 0:
                pivot_row = pivot_selection(mat, diag)
                pivot = mat[pivot_row, diag]
                switch_rows(mat, diag, pivot_row)
            else:
                pivot = mat[diag, diag]

            for row in range(diag + 1, mat.shape[0]):
                mult = mat[row, diag] / pivot
                vec[row] -= mult * vec[diag]
                for col in range(diag, mat.shape[1]):
                    mat[row, col] -= mult * mat[diag, col]
                # If LU decomposition is wanted, store the multipliers in the
                # lower matrix (this creates the lower tridiagonal matrix)
                if lu_decomp:
                    mat[row, diag] = mult

    @classmethod
    def back_substitution(cls, mat, vec):
        """Back substitution method. Assumes `mat` is upper triangular."""
        sol = np.zeros(len(vec))
        sol[-1] = vec[-1] / mat[-1, -1]
        for i in range(len(vec) - 2, -1, -1):
            sol[i] = (vec[i] - mat[i, i + 1] * sol[i + 1]) / mat[i, i]
        return sol

    @classmethod
    def solve(cls, coeff, res):
        """Solves the system `coeff * X = res` for `X`.
        The algorithm is stable for diagonal dominant `coeff` matrices.
        """
        # Triangularization of `coeff` and `res`
        cls.gaussian_elim(coeff, res)
        return cls.back_substitution(coeff, res)


class Tridiagonal(SquareMatrix):
    """A class for methods concerning tridiagonal matrices"""

    @classmethod
    def tri_gaussian_elim(cls, mat, vec):
        """In-place gaussian elimination algorithm."""
        if mat.shape[1] != len(vec):
            raise Exception("Lengths do not match")

        for i in range(1, len(vec)):
            mult = mat[i, i - 1] / mat[i - 1, i - 1]
            mat[i, i] -= mult * mat[i - 1, i]
            mat[i, i - 1] = 0
            vec[i] -= mult * vec[i - 1]