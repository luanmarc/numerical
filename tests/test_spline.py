from numerical.interpolate.spline import (
    NaturalSpline,
    CompleteSpline,
    NotKnotSpline,
    PeridiodicSpline
)


def func(x):
    return 1 / (2 - x)


def func_derivative(x):
    return 1 / (2 - x) ** 2


def ep_test(cls, func, deriv=None, verbose=True):
    for n in [10, 20, 30, 40, 80, 160]:
        # Spline construction
        knots = [i / n for i in range(n)]
        values = [func(k) for k in knots]

        spline = None
        if cls.__name__ == 'CompleteSpline':
            derivatives = [deriv(k) for k in knots]
            spline = cls(knots, values, derivatives)
        else:
            spline = cls(knots, values)

        if verbose:
            print('-> {} constructed for {} for {} '
                'knots...'.format(cls.__name__, func.__name__, n))

        # Error estimate
        error = 0
        _range = 1000
        for i in range(_range):
            point = i / _range
            if point <= knots[-1]:
                oscillation = abs(func(point) - spline.value_at(point))
                if error < oscillation:
                    error = oscillation
            else:
                break

        print('The error for the {} of {} for {} knots is '
            '{}\n'.format(cls.__name__, func.__name__, n, error))

def main():
    print('>>> Error tests:\n')
    ep_test(NaturalSpline, func)
    ep_test(CompleteSpline, func, func_derivative)
    ep_test(NotKnotSpline, func)

if __name__ == '__main__':
    main()
