#=============================================================================
"""
	Dekker method for finding roots of a given function.

----------------------------------AUTHORS-------------------------------------

	Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
	Victor Manuel Dias Saliba (nUSP: 11807702)
	Luan Marc Suquet Camargo (nUSP: 11809090)

"""
#=============================================================================


# Método f: calcula o valor da função f no ponto x
#=============================================================================
import math

def func(x):
	return(x*(math.exp(10 / x) + math.exp(-(10 / x)) - 2) - 1)
#=============================================================================


# Método dekker: calcula uma aproximação para a raiz da função f
#=============================================================================
def dekker(f, a1, b1, b0, ERROABS, ERROREL, i, I_0, j):

	# ATUALIZA A QUANTIDADE DE ITERAÇÕES REALIZADAS
	i = i + 1

	# VERIFICA SE A APROXIMAÇÃO É ACEITÁVEL
	TOL = max(ERROABS, abs(b1) * ERROREL)
	if abs((b1 - a1) / 2) < TOL or f(b1) == 0:
		return b1

	# CALCULA O PONTO MÉDIO E O VALOR PROVISÓRIO (s)
	m = (a1 + b1) / 2
	s = m # Inicializa a variável s
	# Método da dicotomia (caso em que o fator de redução não é satisfatório)
	if (i == 4 and (abs(b1 - a1) / I_0) > 0.125) or (0 < j and j < 3):
		s = m
		j = j + 1
	# Método da secante
	elif f(b1) != f(b0):
		delta = f(b1) * ((b1 - b0)/(f(b1) - f(b0)))
		s = b1 - delta
	# Método da dicotomia
	else:
		s = m

	# CALCULA A NOVA APROXIMAÇÃO (b2)
	b2 = m # Inicializa a variável b2

	if abs(s - b1) < TOL:
		b2 = b1 + TOL * (b1 - a1)/abs(b1 - a1)
	elif m < s and s < b1:
		b2 = s
	else:
		b2 = m

	# CALCULA O NOVO PONTO ANTÍPODA (a2)
	a2 = b1 # Inicializa a variável a2
	if f(a1) * f(b2) < 0:
		a2 = a1
	else:
		a2 = b1

	# PERMUTA OS VALORES DE a2 E b2 CASO FOR NECESSÁRIO
	if abs(f(a2)) < abs(f(b2)):
		aux = b2
		b2 = a2
		a2 = aux

	return dekker(f, a2, b2, b1, ERROABS, ERROREL, i, I_0, j)
#=============================================================================


# Método principal
#=============================================================================
def main():

	# Define o intervalo inicial [a_0, b_0]
	a = float(input("a_0: "))
	b = float(input("b_0: "))

	# Define os erros
	ERROABS = float(input("Absolute error: "))
	ERROREL = float(input("Relative error: "))

	if func(a) * func(b) >= 0:
		raise Exception("The function must change sign in [a_0, b_0]")
		pass

	# Calcula a aproximação para a raiz da função f, bar_x
	bar_x = dekker(func, a, b, a, ERROABS, ERROREL, 0, b - a, 0)

	print(bar_x)
 
if __name__ == "__main__":
    main()
#=============================================================================
