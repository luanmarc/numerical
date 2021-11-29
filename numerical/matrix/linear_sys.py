"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc Suquet Camargo      (nUSP: 11809090)

Computacao III (CCM):
EP 2 Cubic interpolating splines: SquareMatrix, Tridiagonal, Periodic
EP 3 QR Factorization: Qr
"""
from numerical.matrix.linear_space import RealSpace
import numpy as np


class SquareMatrix:
    """Methods for square matrices"""

    @classmethod
    def gaussian_elim(
        cls, mat: np.ndarray, vec: np.ndarray, lu_decomp: bool = False
    ) -> None:
        """General, in-place, gaussian elimination.
        If `lu_decomp` is set as `True`, the method will use the upper
        triangular part of `mat` for U and the lower part for L"""
        if mat.shape[0] != mat.shape[1]:
            raise Exception("Matrix not square")

        def pivot_selection(mat: np.ndarray, col: int) -> int:
            """Partial pivot selection:
            Returns the row index of the pivot given a specific column."""
            pivot_row = 0
            for row in range(1, mat.shape[0]):
                if abs(mat[row, col]) > abs(mat[pivot_row, col]):
                    pivot_row = row
            if mat[pivot_row, col] == 0:
                raise Exception("The matrix is singular!")
            return pivot_row

        def switch_rows(mat: np.ndarray, row0: int, row1: int) -> None:
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
    def back_substitution(cls, mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Back substitution method."""
        if mat.shape[0] != mat.shape[1]:
            raise Exception("Matrix not square")
        n = len(vec)
        sol = np.zeros(n)
        sol[-1] = vec[-1] / mat[-1, -1]

        for i in range(n - 2, -1, -1):
            acc = 0
            for j in range(i, n):
                acc += sol[j] * mat[i, j]
            sol[i] = (vec[i] - acc) / mat[i, i]

        return sol

    @classmethod
    def solve(cls, mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Solves the system `mat * X = vec` for `X`.
        The algorithm is stable for diagonal dominant `mat` matrices.
        """
        if mat.shape[0] != mat.shape[1]:
            raise Exception("Matrix not square")

        # Triangularization of `mat` and `vec`
        cls.gaussian_elim(mat, vec)
        return cls.back_substitution(mat, vec)


class Tridiagonal(SquareMatrix):
    """A class for methods concerning tridiagonal matrices"""

    @classmethod
    def gaussian_elim(
        cls, mat: np.ndarray, vec: np.ndarray, lu_decomp: bool = False
    ) -> None:
        """In-place gaussian elimination algorithm.
        If `lu_decomp` is set as `True`, the method will use the upper
        triangular part of `mat` for U and the lower part for L"""
        if mat.shape[1] != len(vec):
            raise Exception("Lengths do not match")

        for i in range(1, len(vec)):
            mult = mat[i, i - 1] / mat[i - 1, i - 1]
            mat[i, i] -= mult * mat[i - 1, i]
            # If LU decomposition is wanted, store the multipliers in the
            # lower matrix (this creates the lower tridiagonal matrix)
            if lu_decomp:
                mat[i, i - 1] = mult
            else:
                mat[i, i - 1] = 0
            vec[i] -= mult * vec[i - 1]


class Periodic(SquareMatrix):
    """A class for methods concerning periodic matrices"""

    @classmethod
    def solve(cls, mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """In-place solve a periodic linear system and returns the solution"""
        # Before in-place operations, store needed information
        v_tilde = np.copy(mat[-1, :-1])

        # Solve for the y solution and find the multipliers
        SquareMatrix.gaussian_elim(mat[:-1, :-1], mat[:-1, -1], True)
        y_sol = SquareMatrix.back_substitution(mat[:-1, :-1], mat[:-1, -1])

        # Use the multipliers on `z_vec`
        z_vec = vec[:-1]
        for i in range(1, mat.shape[1] - 1):
            z_vec[i] -= z_vec[i - 1] * mat[i, i - 1]

        # Now that `vec` is corrected, we can simply find the z solution by
        # doing the back substitution
        z_sol = SquareMatrix.back_substitution(mat[:-1, :-1], z_vec)

        last = (vec[-1] - RealSpace.inner_product(v_tilde, z_sol)) / (
            mat[-1, -1] - RealSpace.inner_product(v_tilde, y_sol)
        )

        # `sol0` contains the inner solutions, insert the first and last (equal)
        # elements (`last`) when returning
        sol0 = np.zeros(len(vec))
        sol0 = z_sol - last * y_sol
        return np.insert(sol0, [0, len(sol0)], last)


class Qr:
    """
    The goal is to solve the system A x = b where A is an m by n matrix. If A
    has linearly independent columns, the solution is unique.
    """

    @classmethod
    def factorization(cls, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """QR factorization algorithm"""
        m, n = mat.shape
        mn = min(m, n)
        q = np.column_stack([mat[:, j] for j in range(mn)]).reshape((m, mn))
        r = np.zeros((mn, n))

        for j in range(mn):
            for i in range(j):
                x = RealSpace.inner_product(q[:, i], mat[:, j])
                q[:, j] -= x * q[:, i]
                r[i, j] = x
            norm = RealSpace.norm(q[:, j])
            if norm == 0:
                raise Exception("The matrix contains a set of LD columns")
            q[:, j] /= norm
            r[j, j] = RealSpace.inner_product(q[:, j], mat[:, j])

        # remaining columns for the case m < n
        if m < n:
            for j in range(mn, n):
                for i in range(j):
                    if i < r.shape[0]:
                        r[i, j] = RealSpace.inner_product(q[:, i], mat[:, j])
                    else:
                        break

        return q, r

    @classmethod
    def solve(cls, mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Solve the linear system ``mat @ sol = vec``"""
        q, r = cls.factorization(mat)
        m, n = mat.shape

        # Assert the correctness of the arguments
        if m < n:
            raise Exception("The system has no solution")

        # solve the system ``r @ sol = z``
        # `r` is a triangular square matrix of order `n`
        z = np.zeros(n)
        b = np.copy(vec)
        for i in range(n):
            z[i] = RealSpace.inner_product(q[:, i], b)
            b -= z[i] * q[:, i]
        return SquareMatrix.back_substitution(r, z)
