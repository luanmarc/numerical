"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc                     (nUSP: 11809090)

Tests: module numerical.interpolate.spline
"""
import math
import matplotlib.pyplot as plt
from numerical.interpolate.spline import (
    NaturalSpline,
    CompleteSpline,
    NotKnotSpline,
    PeridiodicSpline,
)


class Curves:
    """Bidimensional smooth curves"""

    def __init__(self, coord1: list[float], coord2: list[float], _range: int):
        if len(coord1) != len(coord2):
            raise Exception("Coordinates must have the same length")
        self.coord1 = coord1
        self.coord2 = coord2
        self.param = [self.calc_param(i) for i in range(len(coord1))]
        self.spline1 = PeridiodicSpline(self.param, coord1)
        self.spline2 = PeridiodicSpline(self.param, coord2)
        self.points = self.calc_points(_range)

    def calc_param(self, index: int) -> float:
        """Calculates the parameter at `index`"""
        if index == 0:
            return 0
        return self.calc_param(index - 1) + math.sqrt(
            (self.coord1[index] - self.coord1[index - 1]) ** 2
            + (self.coord2[index] - self.coord2[index - 1]) ** 2
        )

    def get_param_at(self, index: int) -> float:
        """Returns the parameter at `index`"""
        return self.param[index]

    def get_param(self) -> list[float]:
        """Returns the list of parameters"""
        return self.param

    def calc_points(self, _range: int) -> list[tuple[float, float]]:
        """Calculates a list of length `_range` of curve points"""
        points = []
        for i in range(_range):
            epsilon = i / _range
            t = (1 - epsilon) * self.param[0] + self.param[-1] * epsilon
            x = self.spline1.value_at(t)
            y = self.spline2.value_at(t)
            points.append((x, y))
        return points

    def get_points(self) -> list[tuple[float, float]]:
        """Return the list of curve points"""
        return self.points

    def plot_curve(self):
        """Plots the curve"""
        plt.plot(*zip(*self.points))
        plt.show()


def ep_test(cls, func, deriv=None, verbose=True):
    for n in [10, 20, 30, 40, 80, 160]:
        # Spline construction
        knots = [i / n for i in range(n)]
        values = [func(k) for k in knots]

        spline = None
        if cls.__name__ == "CompleteSpline":
            # The complete spline requires derivative values
            derivatives = [deriv(k) for k in knots]
            spline = cls(knots, values, derivatives)
        else:
            spline = cls(knots, values)

        if verbose:
            print(
                "-> {} constructed for {} for {} "
                "knots...".format(cls.__name__, func.__name__, n)
            )

        # Error estimate
        error = 0
        _range = 1000
        for i in range(_range):
            point = i / _range
            if point <= knots[-1]:
                oscillation = abs(func(point) - spline.value_at(point))
                error = max(error, oscillation)
            else:
                break

        print(
            "The error for the {} of {} for {} knots is "
            "{}\n".format(cls.__name__, func.__name__, n, error)
        )


def main():
    """Tests"""
    def func(x: float) -> float:
        return 1 / (2 - x)

    def func_derivative(x: float) -> float:
        return 1 / (2 - x) ** 2

    print(">>> Error tests:\n")
    ep_test(NaturalSpline, func)
    ep_test(CompleteSpline, func, func_derivative)
    ep_test(NotKnotSpline, func)

    print("\n\n>>> Curves test:\n")
    coord1 = [25, 19, 13, 9, 5, 2.2, 1, 3, 8, 13, 18, 25]
    coord2 = [5, 7.5, 9.1, 9.4, 9, 7.5, 5, 2.1, 2, 3.5, 4.5, 5]
    curve = Curves(coord1, coord2, 400)
    points = curve.get_points()
    acc = 0
    for point in points:
        print("Point {}: {}".format(acc, point))
        acc += 1
    print("Plotting...")
    curve.plot_curve()


if __name__ == "__main__":
    main()
