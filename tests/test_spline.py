from numerical.interpolate.spline import (
        NaturalSpline,
        CompleteSpline,
        PeridiodicSpline,
        )

def main():
    # Some idiot tests just to see if stuff works
    print('testing natural spline')
    knots = [0, 0.5, 0.8, 1, 1.2]
    moments = [54., 70., 66., 20., 10.]
    s0 = NaturalSpline(knots, moments)
    print('moments = {}'.format(s0.moments))
    print()
    x = 0.99
    i = s0.which_interval(x)
    print('{} contained in interval '
          '{}: [{}, {}]'.format(x, i, knots[i], knots[i + 1]))
    print()


    print('testing complete spline')
    knots = [0, 0.5, 0.8, 1, 1.2]
    moments = [54., 70., 66., 20., 10.]
    derivatives = (3., -6.)
    s1 = CompleteSpline(knots, moments, derivatives)
    print('moments = {}'.format(s1.moments))


if __name__ == '__main__':
    main()
