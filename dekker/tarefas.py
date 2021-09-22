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


def clover(theta):
    '''
    Function that looks like an 'M'
    '''
    return math.sin(4 * theta) ** 2 + math.cos(4 * theta)


def curly(theta):
    '''
    Cool s wave function
    '''
    return 1 + 2 * math.sin(theta / 2)


# checks whether the function reaches zero and changes signal at least once within the given interval
def there_is_zero(f,head,tail,N):
    length = tail -head
    step = length/N
    t = head
    a = f(head)
    for i in range (1,N+1): 
        t += step
        if a*f(t)<=0:
            return True
    return False


def dekpol(f1,f2,head=0,tail=2*math.pi,abs_error=0.000001,rel_error=0.000001,N=400,rneg=True) :
    
    if there_is_zero(f1,head,tail,N) and there_is_zero(f2,head,tail,N) :
        print("The curves intersect at the origin and at the following points (r,theta):")
    else:
        print("The curves intersect at the following points (r,theta):")
    
    inter = []
    length = tail -head
    step = length/N
    for i in range (N):
        h = head + i*step
        t = h + step
        if (f1(h)-f2(h))*(f1(t)-f2(t))<=0:
            theta = dek.dekker(lambda x: f1(x)-f2(x),h,t,abs_error, rel_error,False)
            r=f1(theta)
            if r<0:
                r=-r
                theta += math.pi
            inter.append((r,theta%(2*math.pi)))
    
        
    if not rneg:
        for j in inter:
            print(j)
        return
    
    
    for i in range (N):
        h = head + i*step
        t = h + step
        if (f1(h)+f2(h+math.pi))*(f1(t)+f2(t+math.pi))<=0:
            theta = dek.dekker(lambda x: f1(x)+f2(x+math.pi),h,t,abs_error, rel_error,False)
            if f1(theta)>0:
                inter.append((f1(theta),theta%(2*math.pi)))
        if (f2(h)+f1(h+math.pi))*(f2(t)+f1(t+math.pi))<=0:
            theta = dek.dekker(lambda x: f2(x)+f1(x+math.pi),h,t,abs_error, rel_error,False)
            if f2(theta)>0:
                inter.append((f2(theta),theta%(2*math.pi)))
        
    for j in inter:
        print(j)

def main():
    '''
    Solutions to the given homework problems
    '''
    print("----------------------------------------------------------------")
    print('Test for the first homework: ',end="\n\n")
    dek.dekker(func1, 0.1, 1, 0.000001, 0.000001)
    print()
    print()
    print("----------------------------------------------------------------")

    print('Test for the second homework: ',end="\n\n")
    dek.dekker(func2, 90, 110, 0.000001, 0.000001)
    print()
    print()
    print("----------------------------------------------------------------")

    print('Tests for the third homework: ',end="\n\n\n")
    print('Intersections between butterfly and cardioid',end="\n\n")
    dekpol(butterfly,card)
    print(end="\n\n")
    print('Intersections between butterfly and clover',end="\n\n")
    dekpol(butterfly,clover)
    print(end="\n\n")
    print('Intersections between butterfly and curly',end="\n\n")
    dekpol(butterfly,curly,0,4*math.pi)
    print(end="\n\n")
    print('Intersections between cardioid and curly',end="\n\n")
    dekpol(card,curly,0,4*math.pi)
    print(end="\n\n")
    print('Intersections between clover and curly',end="\n\n")
    dekpol(clover,curly,0,4*math.pi)
    print(end="\n\n")
    print('Intersections between cardioid and clover',end="\n\n")
    dekpol(card,clover)
    print(end="\n\n")
    

if __name__ == '__main__':
    main()
