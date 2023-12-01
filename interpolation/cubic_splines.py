import numpy as np

class Cubicsplines:
    r"""
    generating the cubic splines interpolation of a function for 
    arbitral degree and for specified degree and basis
    N.B. the implementation is for equispaced grid

    Parameters
    ----------
    func: function
        function to interpolate
    degree: int
        degree of interpolation (number of sections)
    interval: list [start, end]
        interval of the interpolation
    output_points: int
        number of data points in the output

    Returns
    ---------
    x: numpy array 
        array of x in a fine grid
    y: numpy array
        array of interpolated value calculated for fine grid

    """

    def __init__(self,func,degree, interval,output_points = 1024):
        self.func = func
        self.degree = degree
        self.interval = interval
        self.output_points = output_points

    def gen_equispaced_grid(self):
        start = self.interval[0]
        end = self.interval[1]
        self.grid_size = self.degree + 1 #n+1
        
        
        return np.linspace(start, end, self.grid_size, endpoint=True)
    
    def gen_data_points(self,grid):
        return self.func(grid)
    
    def gen_h(self,grid):
        #generate the equispaced grid assuming grid is
        h = ((self.interval[1] - self.interval[0]) / self.degree)
        #h has size n
        self.h = h

    def get_matrix(self): 
        # Define the size of the matrix
        n = self.degree -1

        main_diag = np.ones(n) * 4
        up_diag = np.ones(n-1)
        down_diag = np.ones(n-1)

        tridiag = np.diag(main_diag) + np.diag(up_diag, -1) + np.diag(down_diag, 1)
        matrix = tridiag * self.h
        return matrix
    
    def gen_vector(self,data_points):
        #function to get the right hand side of the system
        # returns the vector (1,n-1) with each value (6 * (f(x_i+1) - f(x_i)/h_i+1 - f(x_i) - f(x_i-1)/h_i))   
        vector = []
        for i in range(1,self.degree):
            vector.append(6 * ((data_points[i+1]-data_points[i])/self.h - (data_points[i]-data_points[i-1])/self.h))
        return np.array(vector)
    
    def get_coeff(self,matrix,vector):
        #function to solve the system of the simultaneous equation to get sigma_i
        #eliminate the first and the last datapoints

        coeff = np.linalg.solve(matrix, vector)

        coeff = np.concatenate(([0], coeff, [0]))
        #coeff has size n+1
        return coeff

    def cubicspline_interpolate(self,grid,data_points,coeff):

        number_in_each_segment = int(self.output_points / (self.degree))
        x = np.array([])
        y = np.array([])
        for i in range(self.degree):
            alpha = data_points[i+1]/self.h - (1/6) * coeff[i+1] * self.h
            beta = data_points[i]/self.h - (1/6) * coeff[i] * self.h

            x_segment = np.linspace(grid[i],grid[i+1],number_in_each_segment)
            y_segment = (grid[i+1]-x_segment)**3 / (6 * self.h) * coeff[i] + (x_segment - grid[i])** 3 / (6 * self.h) * coeff[i+1]+ alpha * (x_segment - grid[i]) + beta * (grid[i+1] - x_segment)
            x = np.concatenate((x,x_segment))
            y = np.concatenate((y,y_segment))

        return x, y
    
    def main(self):
        grid = self.gen_equispaced_grid()
        data_points = self.gen_data_points(grid)
        self.gen_h(grid)
        matrix = self.get_matrix()
        vector = self.gen_vector(data_points)
        coeff = self.get_coeff(matrix,vector)
        x,y =  self.cubicspline_interpolate(grid,data_points,coeff)
        return x, y 



