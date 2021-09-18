import math
import dekker as dek

def func1(k):
    """
    Função cuja raiz é solução da Tarefa 3.1
    """
    return 10 - math.exp(-2 * k) * (10 - 3 * k) - 20 * k


def func2(x):
    """
    Função cuja raiz é solução da Tarefa 3.2
    """
    return x * (math.exp(10 / x) + math.exp(-10 / x) - 2) - 1


def butterfly(theta):
    '''
    Butterfly function
    '''
    return math.e ** math.sin(theta) - 2 * math.cos(4 * theta)


def card(theta):
    '''
    Cardioid function
    '''
    return 1 - math.sin(theta)


def m_wave(theta):
    '''
    Function that looks like an 'M'
    '''
    return math.sin(4 * theta) ** 2 + math.cos(4 * theta)


def s_wave(theta):
    '''
    Cool s wave function
    '''
    return 1 + 2 * math.sin(theta / 2)

def main():
    '''
    Solutions to the given homework problems
    '''
    print('Test for the first homework: ')
    dek.dekker(func1, 0.1, 1, 0.000001, 0.000001)
    print()

    print('Test for the second homework: ')
    dek.dekker(func2, 90, 110, 0.000001, 0.000001)
    print()

    print('Tests for the third homework: ')
    print('Intersections between butterfly and ')
    head = 0
    for n in range(1, 101):
        tail = 2 * n * math.pi / 100
        dek.dekker(lambda x: butterfly(x) - card(x), head, tail, 0.00001, 0.00001)
        head = tail

if __name__ == '__main__':
    main()
