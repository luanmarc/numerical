"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc Suquet Camargo      (nUSP: 11809090)

Computacao III (CCM): Ep 3 QR factorization
"""
from numerical.matrix.linear_space import RealSpace
import numpy as np

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
        r = np.zeros((mn, n)) # TODO: correct this shape

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
