import numpy as np

def bisection(func, interval,tol):
    #function to obtain root of func(x) = 0 with bisection algorithm for a given interval
    iter = 0
    
    a = interval[0]
    b = interval[1]
    c = (a+b) / 2
    if func(a) * func(b) >= 0:
        print("error: interval is not approproately set. check the value at each end")
    else:
        iter = 0
        while abs(func(c)) > tol:
            iter += 1
            if func(a) * func(c)  < 0:
                b = c
            else:
                a = c
            c = (a+b)/2

    return c,iter

def regula_falsi(func, interval,tol):
    #function to obtain root of func(x) = 0 with regula falsi algorithm for a given interval
    a = interval[0]
    b = interval[1]
    c = (a * func(b) - b * func(a)) / (func(b) - func(a))
    if func(a) * func(b) >= 0:
        print("error: interval is not approproately set. check the value at each end")
    else:
        iter = 0
        while abs(func(c))  > tol:
            iter += 1
            if func(a) * func(c)  < 0:
                b = c
            else:
                a = c
            c = (a * func(b) - b * func(a)) / (func(b) - func(a))

    return c,iter

def illinois(func, interval,tol):
    #function to obtain root of func(x) = 0 with illinois algorithm for a given interval
    a = interval[0]
    b = interval[1]
    
    if func(a) * func(b) >= 0:
        print("error: interval is not approproately set. check the value at each end")
    else:
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        bias = [0,0]
        iter = 0
        while abs(func(c)) > tol:
            iter +=1
            if func(a) * func(c)  < 0:
                b = c #moving to right
                bias[1] += 1
            else:
                a = c #moving to left
                bias[0] += 1
            
            
            if bias[1] == 2:
                corr = [1,0.5]
                bias = [0,1]
            elif bias[0] == 2:
                corr = [0.5,1]
                bias = [1,0]
            else:
                corr = [1,1]
            c = (a * func(b) * corr[0] - b * func(a) * corr[1]) / (func(b) * corr[0] - func(a) * corr[1])
            

    return c,iter

def newton_raphson(func,deriv,interval=[-1* np.infty,np.infty],tol=10^(-6),init_guess=None):
    if init_guess == None:
        x = interval[0]
    while abs(func(x)) > tol:
        x = x - func(x) / deriv(x)
    return x