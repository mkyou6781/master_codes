import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root 

# Note that the multi variable newton method does not work for implicit scheme
# due to the fact that Jacobian has a large condition number
def multi_variable_newton(f, jacob, x_0, iter=100, tol=10 * (-6)):
    # function to execute newton method for multiple variables
    # f (numpy array) unknown function to obtain root
    #   input: x the value
    #          x_prev previous value
    # jacob (numpy array) jacobian matrix of f
    # x_0 (numpy array) initial guess
    i = 0
    x = x_0
    while (i < iter) and (np.linalg.norm(f(x, x_0)) > tol):
        print("iter",i)
        #print("error",np.linalg.norm(f(x,x_0)))
        f_val = f(x, x_0)
        # print("f_val",f_val)
        jacob_val = jacob(x)
        print("jacob_condition_num",np.linalg.cond(jacob_val))
        jacob_inv = np.linalg.inv(jacob_val)
        dx = (-1) * np.matmul(jacob_inv, f_val)
        x = x + dx
        i += 1
        print("dx", dx)
    return x


class Cahn_Hilliard:
    def __init__(
        self,
        N,
        dt,
        time_end,
        init_func,
        epsilon,
        scheme_type="explicit",
        semi_imp_type=None,
    ):
        self.N = int(N)
        self.mesh_num = int(N) + 1  # include the fictitious node (N+1 (0,1,...,N) + 2)
        self.dt = dt
        self.time_end = time_end
        self.init_func = init_func
        self.epsilon = epsilon
        self.scheme_type = scheme_type
        self.semi_imp_type = semi_imp_type

    def get_h_dt(self):
        self.h = 1 / (self.N)
        self.time_mesh_num = int(self.time_end / self.dt)

    def meshing(self):
        self.space_mesh = np.linspace(0, 1, self.N + 1, endpoint=True)

    def initialise(self):
        self.c = np.zeros((self.time_mesh_num, self.mesh_num))
        self.w = np.zeros((self.time_mesh_num, self.mesh_num))

    def disc_init(self):
        # discretising the initial condition
        init_cond_c = [self.init_func(element) for element in self.space_mesh]

        return init_cond_c

    def phi_dash(self, c):
        return np.power(c, 3) - c

    def phi_dash_dash(self, c):
        return 3 * np.power(c, 2) - 1

    def imposing_init_cond(self, init_cond_c):
        self.c[0, :] = init_cond_c

        # note that the calculation of W at i=0 and i=N is different due to the Neumann condition
        self.w[0, 0] = 1 / (self.epsilon) * self.phi_dash(
            self.c[0, 0]
        ) - self.epsilon * (1 / (self.h**2)) * (2 * self.c[0, 1] - 2 * self.c[0, 0])
        for i in range(self.mesh_num - 2):
            self.w[0, i + 1] = 1 / (self.epsilon) * self.phi_dash(self.c[0, i + 1]) 
            -self.epsilon * (1 / (self.h**2)) * (
                self.c[0, i + 2] - 2 * self.c[0, i + 1] + self.c[0, i]
            )
        self.w[0, self.mesh_num - 1] = 1 / (self.epsilon) * self.phi_dash(
            self.c[0, self.mesh_num - 1]
        ) - self.epsilon * (1 / (self.h**2)) * (
            2 * self.c[0, self.mesh_num - 2] - 2 * self.c[0, self.mesh_num - 1]
        )

    def explicit_scheme(self):
        # calculating all the values of c and w after t = 0
        for m in range(1, self.time_mesh_num):
            # solve for c first
            # Note Neumann condition
            self.c[m, 0] = self.c[m - 1, 0] + (self.dt / (self.h) ** 2) * (
                2 * self.w[m - 1, 1] - 2 * self.w[m - 1, 0]
            )
            for i in range(self.mesh_num - 2):
                self.c[m, i + 1] = self.c[m - 1, i + 1] + (self.dt / (self.h) ** 2) * (
                    self.w[m - 1, i + 2] - 2 * self.w[m - 1, i + 1] + self.w[m - 1, i]
                )
            self.c[m, self.mesh_num - 1] = self.c[m - 1, self.mesh_num - 1] + (
                self.dt / (self.h) ** 2
            ) * (
                self.w[m - 1, self.mesh_num - 2] - 2 * self.w[m - 1, self.mesh_num - 1]
            )

            # solve for w
            self.w[m, 0] = 1 / (self.epsilon) * self.phi_dash(
                self.c[m, 0]
            ) - self.epsilon * (1 / (self.h) ** 2) * (
                2 * self.c[m, 1] - 2 * self.c[m, 0]
            )
            for i in range(self.mesh_num - 2):
                self.w[m, i + 1] = 1 / (self.epsilon) * self.phi_dash(
                    self.c[m, i + 1]
                ) - self.epsilon * (1 / (self.h) ** 2) * (
                    self.c[m, i + 2] - 2 * self.c[m, i + 1] + self.c[m, i]
                )
            self.w[m, self.mesh_num - 1] = 1 / (self.epsilon) * self.phi_dash(
                self.c[m, self.mesh_num - 1]
            ) - self.epsilon * (1 / (self.h) ** 2) * (
                2 * self.c[m, self.mesh_num - 2] - 2 * self.c[m, self.mesh_num - 1]
            )

    def implicit_scheme(self):
        # defining all the functions corresponding to the scheme
        def f(unknown, prev_unknown):
            val = np.zeros(2 * self.mesh_num)
            # f (scheme for dc/dt - Dw=0)
            val[0] = (unknown[0] - prev_unknown[0]) / self.dt - (1 / (self.h**2)) * (
                2 * unknown[self.mesh_num+1] - 2 * unknown[self.mesh_num]
            )
            for i in range(self.mesh_num - 2):
                val[i + 1] = (unknown[i + 1] - prev_unknown[i + 1]) / self.dt - (
                    1 / (self.h**2)
                ) * (unknown[self.mesh_num + i] - 2 * unknown[self.mesh_num + i + 1] + unknown[self.mesh_num + i + 2])
            val[self.mesh_num - 1] = (
                unknown[self.mesh_num - 1] - prev_unknown[self.mesh_num - 1]
            ) / self.dt - (1 / (self.h**2)) * (
                2 * unknown[2 * self.mesh_num - 2] - 2 * unknown[2 * self.mesh_num - 1]
            )

            # g (scheme for w - phi_dash/epsilon(c) + epsilon Dc = 0)
            val[self.mesh_num] = (
                unknown[self.mesh_num]
                - (1 / self.epsilon) * self.phi_dash(unknown[0])
                + (self.epsilon/(self.h**2))
                * (2 * unknown[self.mesh_num + 1] - 2 * unknown[self.mesh_num])
            )
            for i in range(self.mesh_num - 2):
                val[self.mesh_num + i + 1] = (
                    unknown[self.mesh_num + i + 1]
                    - (1 / self.epsilon) * self.phi_dash(unknown[i + 1])
                    + (self.epsilon/(self.h**2))
                    * (
                        unknown[i]
                        - 2 * unknown[i + 1]
                        + unknown[i + 2]
                    )
                )
            val[2 * self.mesh_num - 1] = (
                unknown[2 * self.mesh_num - 1]
                - (1 / self.epsilon) * self.phi_dash(unknown[self.mesh_num - 1])
                + (self.epsilon/(self.h**2))
                * (
                    2 * unknown[self.mesh_num - 2]
                    - 2 * unknown[self.mesh_num - 1]
                )
            )
            return val


        for m in range(1, self.time_mesh_num):
            prev_unknown = np.zeros(2 * self.mesh_num)
            prev_unknown[0 : self.mesh_num] = self.c[m - 1, :]
            prev_unknown[self.mesh_num : 2 * self.mesh_num] = self.w[m - 1, :]

            unknown = root(f, prev_unknown, args=prev_unknown)
            #print("unknown",unknown.x)
            unknown = unknown.x
            self.c[m, :] = unknown[0 : self.mesh_num]
            self.w[m, :] = unknown[self.mesh_num : 2 * self.mesh_num]

    
    def semi_implicit_scheme(self):
        pass


    def difference_scheme(self, init_cond_c):
        self.imposing_init_cond(init_cond_c)
        if self.scheme_type == "explicit":
            self.explicit_scheme()

        elif self.scheme_type == "implicit":
            self.implicit_scheme()
        else:
            print("error: chooose the right scheme type")

    def main(self):
        self.get_h_dt()
        self.meshing()
        self.initialise()
        init_cond_c = self.disc_init()
        self.difference_scheme(init_cond_c)
        return self.c, self.w
