import matplotlib.pyplot as plt
import numpy as np

class Plotting:
    def __init__(self,c,w,init_func,mesh_num,time_array,dt):
        self.c = c
        self.w = w
        self.init_func = init_func
        self.mesh_num = int(mesh_num) # excluding the fictitious nodes
        self.time_array = time_array
        self.dt = dt

    def space_mesh(self):
        self.space_mesh = np.linspace(0,1,self.mesh_num,endpoint=True)

    def plot_init_func(self):
        init_cond_c = [self.init_func(element) for element in self.space_mesh]
        plt.plot(self.space_mesh,init_cond_c,label="initial condition of c")

    def plot_at_time_points(self):
        for t in self.time_array:
            plt.plot(self.space_mesh,self.c[int(t/self.dt)-1,1:self.mesh_num+1],label = "at time t = {:.3f}".format(t))

    def main(self):
        self.space_mesh()
        self.plot_init_func()
        self.plot_at_time_points()
        
        plt.title("c over time",fontsize="x-large")
        plt.xlabel("x",fontsize="x-large")
        plt.ylabel("c",fontsize="x-large")
        plt.legend(fontsize="x-large")
        plt.savefig("CAHN_HILLIARD/cahn_hilliard/result_plot/cahn_hil_implicit.png",bbox_inches = "tight")
        plt.show()