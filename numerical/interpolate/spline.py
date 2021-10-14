'''
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Computacao III (CCM): EP 2 Cubic interpolating splines 
'''
from numerical.interpolate.matrix import Tridiagonal
import numpy as np


class Spline(object):
    '''
    Cubic interpolating spline: real valued twice continuously differentiable
    function. For each interval of consequent knots the spline coincides with a
    polynomial of degree at most 3.
    '''

    def __init__(self, knots: list[float] = [], values: list[float] = []):
        '''Spline variables'''
        Spline.confirm_args(knots, values)
        self.knots = knots 
        self.values = values


    @staticmethod
    def confirm_args(knots, values):
        '''Confirs if the the args length match'''
        if len(knots) == len(values):
            return
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
        '''Returns the length of the `index` interval'''
        return self.knots[index + 1] - self.knots[index]



class NaturalSpline(Spline):
    '''Natural cubic interpolating spline.
    These are characterized by first and last moments being equal to zero.
    '''

    def __init__(self, knots: list[float], values: list[float]):
        Spline.__init__(self, knots, values)
        self.moments: list[float] = self.find_moments()


    def find_moments(self) -> list[float]:
        n_knots = len(self.knots)
        # Construct matrices
        coeff = 2 * np.identity(n_knots, dtype = float)
        res = np.zeros(n_knots, dtype = float)
        for i in range(1, n_knots - 1):  # Sets the matrices
            # Coefficient matrix: tridiagonal
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



class CompleteSpline(Spline):
    '''Complete cubic interpolating spline'''

    def __init__(self, knots: list[float], values: list[float],
                 deriv: tuple[float, float]):
        Spline.__init__(self, knots, values)
        # Derivatives at knot 0 and n - 1
        self.deriv: tuple[float, float] = deriv
        self.moments: list[float] = self.find_moments()

    def find_moments(self) -> list[float]:
        n_knots = len(self.knots)
        # Construct matrices
        coeff = 2 * np.identity(n_knots, dtype = float)
        res = np.zeros(n_knots, dtype = float)
        for i in range(1, n_knots - 1):  # Sets the matrices
            # Coefficient matrix: tridiagonal
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

        # Complete spline special conditions:
        coeff[0, 1], coeff[-1, -2] = 1, 1
        h0 = self.interval_length(0)
        diff0 = self.values[1] - self.values[0]
        res[0] = 6 / h0 * (diff0 / h0 - self.deriv[0])
        hn = self.interval_length(n_knots - 2)
        diffn = self.values[-1] - self.values[-2]
        res[-1] = 6 / hn * (self.deriv[1] - diffn / hn)

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


# TODO: class NotKnotSpline


def main():
    # Some idiot tests just to see if stuff works
    knots = [0, 0.5, 0.8, 1, 1.2]
    moments = [54., 70., 66., 20., 10.]
    print('testing natural spline')
    s0 = NaturalSpline(knots, moments)
    print('moments = {}'.format(s0.moments))
    print()
    x = 0.99
    i = s0.which_interval(x)
    print('{} contained in interval {}: [{}, {}]'.format(x, i, knots[i], knots[i + 1]))
    print()


    print('testing complete spline')
    knots = [0, 0.5, 0.8, 1, 1.2]
    moments = [54., 70., 66., 20., 10.]
    derivs = (3., -6.)
    s1 = CompleteSpline(knots, moments, derivs)
    print('moments = {}'.format(s1.moments))


if __name__ == '__main__':
    main()
