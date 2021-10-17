'''
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Computacao III (CCM): EP 2 Cubic interpolating splines
'''
from numerical.interpolate.matrix import Tridiagonal
import numpy as np


class Spline(object):
    '''Cubic interpolating spline: real valued twice continuously differentiable
    function. For each interval of consequent knots the spline coincides with a
    polynomial of degree at most 3.
    '''

    def __init__(self, knots: list[float], values: list[float]):
        '''Base of spline variables:\n
        `knots`: list of the partition for the interval.\n
        `values`: list of the values of the spline at the knots.
        '''
        Spline.confirm_args(knots, values)
        self.knots = knots 
        self.values = values


    @staticmethod
    def confirm_args(knots, values):
        '''Confirs if the the args length match'''
        if len(knots) == len(values): return
        else:
            raise Exception('Args must have the same length')


    def get_knots(self) -> list[float]:
        '''Gets the list of knots'''
        return self.knots


    def get_values(self) -> list[float]:
        '''Get the list of values for the knots'''
        return self.values


    def which_interval(self, x: float) -> int:
        '''Finds the interval that contains `x`.
        For instance, if `x` belongs to the interval [`knots[i]`, `knots[i+1]`],
        the method returns `i`. 
        '''
        # Check user input
        if x < self.knots[0] or x > self.knots[-1]:
            raise Exception('Argument not in the interval')

        index = len(self.knots) // 2
        while not (self.knots[index] <= x) and (x <= self.knots[index + 1]):
            if x > self.knots[index]:
                index += index // 2
            else:
                index -= index // 2
        return index


    def interval_length(self, index: int) -> float:
        '''Returns the length of the `index` interval.\n
        Note that `index` should belong to the interval `[0..len(knots) - 1]`
        '''
        return self.knots[index + 1] - self.knots[index]



class NaturalSpline(Spline):
    '''Natural cubic interpolating spline.
    These are characterized by first and last moments being equal to zero.
    '''

    def __init__(self, knots: list[float], values: list[float]):
        Spline.__init__(self, knots, values)
        self.moments: list[float] = self.find_moments()


    def find_moments(self) -> list[float]:
        # Construct matrices
        coeff = 2 * np.identity(len(self.knots), dtype = float)
        res = np.zeros(len(self.knots), dtype = float)
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

        # Solution to the system `coeff` * `moments` = `res`
        moments = Tridiagonal.solve(coeff, res)
        return moments.tolist()


    def get_moments(self) -> list[float]:
        '''Get the moments of the spline'''
        return self.moments


    def value_at(self, x: float) -> float:
        '''Returns the value of the spline function at `x`'''
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2* m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)



class CompleteSpline(Spline):
    '''Complete cubic interpolating spline'''

    def __init__(self, knots: list[float], values: list[float],
                 derivatives: tuple[float, float]):
        '''The `CompleteSpline` requires additionally:\n
        `derivatives`: tuple of derivatives of the spline at the end points of
        the knot list.
        '''
        Spline.__init__(self, knots, values)
        self.derivatives: tuple[float, float] = derivatives
        self.moments: list[float] = self.find_moments()


    def find_moments(self) -> list[float]:
        # Construct matrices
        coeff = 2 * np.identity(len(self.knots), dtype = float)
        res = np.zeros(len(self.knots), dtype = float)
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

        moments = Tridiagonal.solve(coeff, res) # `coeff * moments = res`
        return moments.tolist()


    def get_moments(self) -> list[float]:
        '''Get the moments of the spline'''
        return self.moments


    def value_at(self, x: float) -> float:
        '''Returns the value of the spline function at `x`'''
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2* m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)



class PeridiodicSpline(Spline):
    '''
    `PeriodicSplines` requires the `k`th derivative (`k` in `[0, 1, 2]`) at the
    end point knots to be equal for the given spline.
    '''

    def __init__(self, knots: list[float], values: list[float]):
        Spline.__init__(self, knots, values)
        self.moments: list[float] = self.find_moments()


    def find_moments(self) -> list[float]:
        '''Finds moments of the spline at the given knots'''
        # We'll have the two end point moments equal, so we`ll shorten our
        # matrix by one row and column, making it a square `n - 1` matrix, where
        # `n` is the number of knots
        coeff = 2 * np.identity(len(self.knots) - 1, dtype = float)
        res = np.zeros(len(self.knots) - 1, dtype = float)
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

        # TODO: notice that `coeff` is not quite tridiagonal
        moments = Tridiagonal.solve(coeff, res).tolist()  # `coeff * moments = res`
        moments.insert(0, moments[-1])  # Repeated moment
        return moments


    def get_moments(self) -> list[float]:
        '''Get the moments of the spline'''
        return self.moments


    def value_at(self, x: float) -> float:
        '''Returns the value of the spline function at `x`'''
        # Data about the interval of `x`
        i = self.which_interval(x)
        y0, y1 = self.values[i], self.values[i + 1]
        m0, m1 = self.moments[i], self.moments[i + 1]
        h = self.interval_length(i)

        diff = x - self.knots[i]
        beta = (y1 - y0) / h - (2* m0 + m1) * h / 6
        delta = (m1 - m0) / (6 * h)
        gamma = m0 / 2
        return y0 + beta * diff + gamma * (diff ** 2) + delta * (diff ** 3)
        


