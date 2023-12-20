import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

class LinearSwimmer:
    MU = 10**(-3)

    def __init__(self,radius, r12,r23,W):
        self.radius = radius
        self.r12 = r12
        self.r23 = r23
        self.r13 = lambda t: np.absolute(r12(t)) + np.absolute(r23(t))
        self.W = W

    def oseen_tensor(self,t):
        r12 = self.r12(t)
        r23 = self.r23(t)
        r13 = self.r13(t)

        H11 = (1/(6*np.pi* self.MU * self.radius)) 
        H12 = (1/(8*np.pi* self.MU * r12)) * 2
        H13 = (1/(8*np.pi* self.MU * r13)) * 2

        H21 = (1/(8*np.pi* self.MU * r12)) * 2
        H22 = (1/(6*np.pi* self.MU * self.radius))
        H23 = (1/(8*np.pi* self.MU * r23)) * 2

        H31 = (1/(8*np.pi* self.MU * r13)) * 2
        H32 = (1/(8*np.pi* self.MU * r23)) * 2
        H33 = (1/(6*np.pi* self.MU * self.radius))

        self.oseen_matrix = np.array([[H11,H12,H13],[H21,H22,H23],[H31,H32, H33]])
        #print(self.oseen_matrix)

    def matrix_system(self,W):
        self.matrix = np.zeros((4,4))
        self.matrix[0:3,0:3] = self.oseen_matrix
        self.matrix[0,3] = -1
        self.matrix[1,3] = -1
        self.matrix[2,3] = -1
        self.matrix[3,:] = np.array([1,1,1,0])
        self.cond = np.linalg.cond(self.matrix)

        self.vector = np.array([W,0,0,0])

    def solve_system(self):
        sol = np.linalg.solve(self.matrix,self.vector)
        U = sol[-1]
        return U

    def one_step(self,t):
        self.oseen_tensor(t)
        self.matrix_system(self.W)
        U = self.solve_system()
        plt.plot(t,self.cond,".") 
        return U

    def auxilirary_move(self):
        distance_travelled = integrate.quad(lambda x: self.one_step(x), 0, 1)
        plt.show()
        return distance_travelled
    
    
