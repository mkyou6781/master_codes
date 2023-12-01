import numpy as np
import matplotlib.pyplot as plt

class Trapezium:

    def __init__(self,func, grid):
        self.func = func
        self.grid = grid

    def main(self):
        m = len(self.grid) - 1
        h = (self.grid[-1] - self.grid[0] )/m
        
        integral = (self.func(self.grid[0]) + self.func(self.grid[-1]) ) * (h/2)
        integral += np.sum(self.func(self.grid[1:-1]) * h)

        return integral
    
class Clenshaw_Curtis:

    def __init__(self,func,grid):
        self.func = func
        self.grid = grid
        self.n = len(self.grid) - 1 # since the first value is assigned the index of 0

    def convert_interval(self):
        # function to calculate the normalisation factor so that the inteval becomes (-1,1)
        # and generate the new grid
        self.a = self.grid[0]
        self.b = self.grid[-1]
        norm_factor = (self.b-self.a) / 2 # value to be multiplied to the integral

        self.norm_factor = norm_factor
        self.grid = (norm_factor) **(-1) * (self.grid - self.a) - 1
        self.grid = np.cos(np.pi * (self.grid+1)/2)

    def gen_new_func(self):
        # generate new function after the change of variable
        # now the input of the new func becomes (-1,1)
        new_func = lambda x : self.func(self.a + (x + 1) * (self.b-self.a) / 2)
        self.new_func = new_func

    def cheb_poly(self,k,x):
        return (np.cos(k * np.arccos(x)))

    def gen_vander_matrix(self):
        k_vector = np.arange(self.n + 1)
        x_vector = self.grid
        vander_matrix = self.cheb_poly(k_vector[:, np.newaxis], x_vector)
        vander_matrix = vander_matrix.T
        
        self.vander_matrix = vander_matrix

    def get_coeff(self):
        coeff = np.linalg.solve(self.vander_matrix, self.new_func(self.grid))
        self.coeff = coeff

    def cheb_integral(self,k):
        if k % 2 == 1:
            val = 0
        if k % 2 == 0:
            val = 2 / (1 - k**2)
        return val

    def integral(self):
        cheb_int_vector = [self.cheb_integral(k) for k in np.arange(self.n + 1)]
        
        integral = np.dot(cheb_int_vector,self.coeff) * self.norm_factor
        return integral 

    def main(self):
        self.convert_interval()
        self.gen_new_func()
        self.gen_vander_matrix()
        self.get_coeff()
        integral = self.integral()
        return integral 
    
class Gauss_Legendre:
    def __init__(self,func,grid):
        self.func = func
        self.grid = grid
        self.n = len(self.grid) - 1 # since the first value is assigned the index of 0

    def convert_interval(self):
        # function to calculate the normalisation factor so that the inteval becomes (-1,1)
        # and generate the new grid
        self.a = self.grid[0]
        self.b = self.grid[-1]
        norm_factor = (self.b-self.a) / 2 # value to be multiplied to the integral

        self.norm_factor = norm_factor
        self.grid = (norm_factor) **(-1) * (self.grid - self.a) - 1
        #print(self.grid)

    def gen_new_func(self):
        # generate new function after the change of variable
        # now the input of the new func becomes (-1,1)
        new_func = lambda x : self.func(self.a + (x + 1) * (self.b-self.a) / 2)
        self.new_func = new_func
    
    def get_nodes(self):
        gamma_list = np.array([0.5 / np.sqrt(1-1/((2*i)**2)) for i in range(1,self.n+1)]) 
        tridiagonal = np.diag(gamma_list, k=1)
        tridiagonal += np.diag(gamma_list, k=-1)
        eigenvalues, eigenvectors = np.linalg.eig(tridiagonal)
        return eigenvalues, eigenvectors
    
    def get_weight(self,eigenvectors):
        weight = [((eigenvectors[0,i])**2)/((1/np.sqrt(2))**2) for i in range(self.n+1)]
        weight = np.array(weight)
        return weight
    
    def main(self):
        self.convert_interval()
        self.gen_new_func()
        eigenvalues, eigenvectors = self.get_nodes()
        weight = self.get_weight(eigenvectors)
        integral = np.dot(weight,self.new_func(eigenvalues))
        normalised_integral = integral * self.norm_factor
        return normalised_integral