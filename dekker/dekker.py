################################################################################
# Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
#          Victor Manuel Dias Saliba     (nUSP: 11807702)
#          Luan Marc                     (nUSP: 11809090)
#
# Computacao III (CCM): EP 1
# Dekker method for finding roots of a given function.
################################################################################

def dekker(f, a, b, abs_error, rel_error, verbose = True):
    """
    Dekker method for finding approximations of zeros of a given function f
    in the interval [a, b] with an absolute error and relative error
    of abs_error and rel_error, respectively. You can toggle the verbose if
    you don't wish any printing.
    """
    if f(a) * f(b) > 0:
        e = 'The function must change sign in the interval [{}, {}]'.format(a, b)
        raise Exception(e)

    # Variable initializations
    ant         = a      # Antipode point
    approx      = b      # Current approximation
    last_approx = b      # Last approximation
    fsize       = abs(approx - ant) # Interval size in iteration 0 (mod 4)
    do_dicotomy = False  # Toggle the use of the dicotomy method
    dicotomy_counter = 0 # Counts the number of dicotomy uses

    iteration = 0
    while not (is_approx(approx, ant, abs_error, rel_error) or f(approx) == 0):
        if dicotomy_counter == 3:
            # Stop dicotomy and reset the counter
            do_dicotomy      = False
            dicotomy_counter = 0

        # Checks if the interval decreased sufficiently in the last 4 iterations
        if iteration != 0 and iteration % 4 == 0:
            last_fsize = fsize
            fsize      = abs(approx - ant) # Updates 4th interval size
            if abs(approx - ant) > last_fsize / 8:
                # Apply dicotomy 3 times in a row
                do_dicotomy = True

        if do_dicotomy:
            next_approx = (approx + ant) / 2
        else:
            next_approx = secant_dicotomy(f, approx, ant, last_approx, abs_error, rel_error)

        # Calculates the next antipode
        next_ant = approx
        if f(ant) * f(next_approx) < 0:
            next_ant = ant

        # Swaps antipode and approximation if needed
        if abs(f(ant)) < abs(f(approx)):
            aux    = ant
            ant    = approx
            approx = aux

        # Change of variables to next iteration
        last_approx = approx
        approx      = next_approx
        ant         = next_ant

        if do_dicotomy:
            dicotomy_counter += 1
        iteration += 1

    if verbose:
        print('{} is the approximation, obtained in {} iterations'.format(approx, iteration))
    return approx

def is_approx(approx, ant, abs_error, rel_error):
    """
    Returns true if the approximation satisfies the
    wanted error conditions, false otherwise.
    """
    tol = max(abs_error, abs(approx) * rel_error)
    abs_diff = abs(approx - ant)
    if abs_diff / 2 < tol:
        return True
    return False

def secant_dicotomy(f, approx, ant, last_approx, abs_error, rel_error):
    """
    Implements a mix of the secant and dicotomy methods, returning
    the next approximation for the Dekker method.
    """
    # Mix of dicotomy and secant method approximations
    m = (approx + ant) / 2
    s = m
    if f(approx) != f(last_approx):
        delta = f(approx) * (approx - last_approx) / (f(approx) - f(last_approx))
        s     = approx - delta

    # Returns the next approximation
    tol = max(abs_error, abs(approx) * rel_error)
    if abs(s - approx) < tol:
        return approx + tol * (approx - ant) / abs(approx - ant)
    elif min(approx, m) < s and s < max(approx, m):
        return s
    return m

def main():
    ''' Some tests for the Dekker method '''
    st = input("Which function would you like to use? (Ex: x ** 2 - 3) ")
    f = lambda x: eval(st)
    a = float(input("Head of the interval: "))
    b = float(input("Tail of the interval: "))
    abs_error = float(input("Absolute error: "))
    rel_error = float(input("Relative error: "))
    dekker(f, a, b, abs_error, rel_error)

def g(x):
    return x ** 2 - 2

if __name__ == '__main__':
    main()
