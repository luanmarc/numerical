import math
import dekker as d
# func1: função cuja raiz é solução da Tarefa 3.1
#=================================================================
def func1(k):
    return 10-math.exp(-2*k)*(10-3*k)-20*k
#=================================================================

# func2: função cuja raiz é solução da Tarefa 3.2
#=================================================================
def func2(x):
	return(x*(math.exp(10 / x) + math.exp(-(10 / x)) - 2) - 1)
#=================================================================
