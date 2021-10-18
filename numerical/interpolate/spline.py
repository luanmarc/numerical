"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Computacao III (CCM): EP 2 Cubic interpolating splines
"""
from numerical.interpolate.matrix import Tridiagonal, SquareMatrix
import numpy as np


class Spline:
    """Cubic interpolating spline\n
    Real valued twice continuously differentiable function. For each interval of
    consequent knots the spline coincides with a polynomial of degree at most 3.
    """

    def __init__(self, knots: list[float], values: list[float]):
        """Base of `Spline` variables:
        `knots`: list of the partition for the interval.\n
        `values`: list of the values of the spline at the knots.
        """
        if len(knots) != len(values):
            raise Exception("Args must have the same length")
        self.knots = knots
        self.values = values

    def get_knots(self) -> list[float]:
        """Gets the list of knots"""
        return self.knots

    def get_values(self) -> list[float]:
        """Get the list of values for the knots"""
        return self.values

    def which_interval(self, x: float) -> int:
        """Finds the interval that contains `x`.\n
        For instance, if `x` belongs to the interval
        [`knots[i]`, `knots[i+1]`], the method returns `i`.
        """
        if x < self.knots[0] or x > self.knots[-1]:
            raise Exception("Argument not in the interval")

        index = len(self.knots) - 2
        for i in range(len(self.knots) - 3):
            if self.knots[i] <= x and x <= self.knots[i + 1]:
                index = i
                break
        return index

    def interval_length(self, index: int) -> float:
        """Returns the length of the `index` interval.\n
        Note that `index` should belong to the interval `[0..len(knots) - 1]`
        """
        return self.knots[index + 1] - self.knots[index]


class NaturalSpline(Spline):
    """Natural cubic interpolating spline.\n
    Characterized by first and last moments being equal to zero.
    """

    def __init__(self, knots: list[float], values: list[float]):
        Spline.__init__(self, knots, values)
        self.moments: list[float] = self.find_moments()

    def find_moments(self) -> list[float]:
        # Construct matrices
        coeff = 2 * np.identity(len(self.knots), dtype=float)
        res = np.zeros(len(self.knots), dtype=float)
        for i in range(1, len(self.knots) - 1):
            # Coefficient matrix: tridiagonal
            h_1 = self.interval_length(i - 1)
            h0 = self.interval_length(i)
            upper = h0 / (h0 + h_1)
            coeff[i, i + 1] = upper
            coeff[i, i - 1] = 1 - upper

            # Result matrix
            if i < len(self.knots) - 1:
                diff0 = self.values[i] - self.values[i - 1]
                diff1 = self.values[i + 1] - self.values[i]
                res[i] = 6 / (h_1 + h0) * (diff1 / h0 - diff0 / h_1)

        # Solution to the system `coeff` * `moments` = `res`
        moments = Tridiagonal.solve(coeff, res)
        return moments.tolist()

    def get_moments(self) -> list[float]:
        """Get the moments of the spline"""
        return self.moments

    def value_at(self, x: float) -> float:
        """Returns the value of the spline function at `x`"""
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2 * m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)


class CompleteSpline(Spline):
    """Complete cubic interpolating spline"""

    def __init__(
        self,
        knots: list[float],
        values: list[float],
        derivatives: tuple[float, float]
    ):
        """Base of `Spline` variables:
        `knots`: list of the partition for the interval.\n
        `values`: list of the values of the spline at the knots.\n
        The `CompleteSpline` requires additionally:
        `derivatives`: tuple of derivatives of the spline at the end points of
        the knot list.
        """
        Spline.__init__(self, knots, values)
        self.derivatives: tuple[float, float] = derivatives
        self.moments: list[float] = self.find_moments()

    def find_moments(self) -> list[float]:
        # Construct matrices
        coeff = 2 * np.identity(len(self.knots), dtype=float)
        res = np.zeros(len(self.knots), dtype=float)
        for i in range(1, len(self.knots) - 1):  # Sets the matrices
            # Coefficient matrix: tridiagonal
            h_1 = self.interval_length(i - 1)
            h0 = self.interval_length(i)
            upper = h0 / (h0 + h_1)
            coeff[i, i + 1] = upper
            coeff[i, i - 1] = 1 - upper

            # Result matrix
            if i < len(self.knots) - 1:
                diff0 = self.values[i] - self.values[i - 1]
                diff1 = self.values[i + 1] - self.values[i]
                res[i] = 6 / (h_1 + h0) * (diff1 / h0 - diff0 / h_1)

        # Complete spline special conditions:
        coeff[0, 1], coeff[-1, -2] = 1, 1
        h0 = self.interval_length(0)
        diff0 = self.values[1] - self.values[0]
        res[0] = 6 / h0 * (diff0 / h0 - self.derivatives[0])
        hn = self.interval_length(len(self.knots) - 2)
        diffn = self.values[-1] - self.values[-2]
        res[-1] = 6 / hn * (self.derivatives[1] - diffn / hn)

        moments = Tridiagonal.solve(coeff, res)
        return moments.tolist()

    def get_moments(self) -> list[float]:
        """Get the moments of the spline"""
        return self.moments

    def value_at(self, x: float) -> float:
        """Returns the value of the spline function at `x`"""
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2 * m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)


class PeridiodicSpline(Spline):
    """
    `PeriodicSplines` requires the `k`th derivative (`k` in `[0, 1, 2]`) at the
    end point knots to be equal for the given spline.
    """

    def __init__(self, knots: list[float], values: list[float]):
        Spline.__init__(self, knots, values)
        self.moments: list[float] = self.find_moments()

    def find_moments(self) -> list[float]:
        """Finds moments of the spline at the given knots"""
        # We'll have the two end point moments equal, so we`ll shorten our
        # matrix by one row and column, making it a square `n - 1` matrix,
        # where `n` is the number of knots
        coeff = 2 * np.identity(len(self.knots) - 1, dtype=float)
        res = np.zeros(len(self.knots) - 1, dtype=float)
        for i in range(2, len(self.knots) - 1):
            # Since the range starts at 2, we subtract 1 from the actual matrix
            # coordinates in order to maintain the algorithm concise

            # Coefficient matrix:
            h_1 = self.interval_length(i - 1)
            h0 = self.interval_length(i)
            upper = h0 / (h0 + h_1)
            coeff[i - 1, i] = upper
            coeff[i - 1, i - 2] = 1 - upper

            # Result matrix
            if i < len(self.knots) - 1:
                diff0 = self.values[i] - self.values[i - 1]
                diff1 = self.values[i + 1] - self.values[i]
                res[i - 1] = 6 / (h_1 + h0) * (diff1 / h0 - diff0 / h_1)

        # Complete spline special conditions:
        h0 = self.interval_length(0)
        hn = self.interval_length(len(self.knots) - 2)
        coeff[0, -1] = h0 / (h0 + self.interval_length(1))
        coeff[-1, 0] = h0 / (h0 + hn)
        coeff[-1, -2] = h0 / (hn + h0)

        diff1 = self.values[1] - self.values[-1]
        diffn = self.values[-1] - self.values[-2]
        res[-1] = (6 / (h0 + hn)) * (diff1 / h0 - diffn / hn)

        moments = SquareMatrix.solve(coeff, res).tolist()
        moments.insert(0, moments[-1])  # Repeated moment
        return moments

    def get_moments(self) -> list[float]:
        """Get the moments of the spline"""
        return self.moments

    def value_at(self, x: float) -> float:
        """Returns the value of the spline function at `x`"""
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2 * m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)


class NotKnotSpline(Spline):
    """Not a Knot spline.
    In the intervals `knots[0]` to `knots[2]` and `knots[n - 3]` to `knots[-1]`
    the spline corresponds to a polynomial of degree at most 3.
    """

    def __init__(self, knots: list[float], values: list[float]):
        Spline.__init__(self, knots, values)
        self.moments: list[float] = self.find_moments()

    def find_moments(self) -> list[float]:
        """Finds moments of the spline at the given knots"""
        n_knots = len(self.knots) - 1
        # Construct matrices
        coeff = 2 * np.identity(n_knots, dtype=float)
        res = np.zeros(n_knots, dtype=float)
        for i in range(1, n_knots - 1):  # Sets the matrices
            # Coefficient matrix: tridiagonal
            if i == 1:
                h_1 = self.interval_length(i - 1)
                h0 = self.interval_length(i)
                upper = h_1 / h0
                coeff[i, i + 1] = 1 - upper
                coeff[i, i] = 2 + upper
            if i == (n_knots - 2):
                h_1 = self.interval_length(i - 1)
                h0 = self.interval_length(i)
                upper = h_1 / h0
                coeff[i, i] = 2 + upper
                coeff[i, i - 1] = 1 - upper
            else:
                h_1 = self.interval_length(i - 1)
                h0 = self.interval_length(i)
                upper = h0 / (h0 + h_1)
                coeff[i, i + 1] = upper
                coeff[i, i - 1] = 1 - upper

            # Result matrix
            if i < n_knots - 1:
                diff0 = self.values[i] - self.values[i - 1]
                diff1 = self.values[i + 1] - self.values[i]
                res[i] = 6 / (h_1 + h0) * (diff1 / h0 - diff0 / h_1)

        # First set of moments without m_0 and m_n
        moments_0 = Tridiagonal.solve(coeff, res)

        h1 = self.interval_length(0)
        h2 = self.interval_length(1)
        m_0 = ((h1 + h2) / h2) * moments_0[0] - (h1 / h2) * moments_0[1]
        h_n = self.interval_length(len(self.knots) - 2)
        h_n_1 = self.interval_length(len(self.knots) - 3)
        m_n = ((h_n + h_n_1) / h_n_1) * moments_0[n_knots - 1] - (
            h_n / h_n_1
        ) * moments_0[n_knots - 2]
        moments = np.zeros(len(self.knots), dtype=float)
        moments[0] = m_0
        moments[len(self.knots) - 1] = m_n
        for i in range(1, len(self.knots) - 1):
            moments[i] = moments_0[i - 1]

        return moments.tolist()

    def get_moments(self) -> list[float]:
        """Get the moments of the spline"""
        return self.moments

    def value_at(self, x: float) -> float:
        """Returns the value of the spline function at `x`"""
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2 * m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)