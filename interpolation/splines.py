import numpy as np

class Splines:
    r"""
    generating the splines interpolation of a function for 
    arbitral degree and for specified degree and basis

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
        array of interpolated values calculated for fine grid

    """

    def __init__(self,func,degree, interval,output_points = 1000):
        self.func = func
        self.degree = degree
        self.interval = interval
        self.output_points = output_points

    def gen_equispaced_grid(self):
        start = self.interval[0]
        end = self.interval[1]
        
        return np.linspace(start, end, self.degree + 1, endpoint=True)
    
    def gen_data_points(self,grid):
        return self.func(grid)
    
    def spline_interpolate(self,grid,data_points):
        #probably there is better way to implement this as it does not work 
        #when the degree is not exact division of the data_points number
        number_in_each_segment = int(self.output_points / (self.degree))
        x = np.array([])
        y = np.array([])
        for i in range(self.degree):
            x_segment = np.linspace(grid[i],grid[i+1],number_in_each_segment)
            
            y_segment = (grid[i+1] - x_segment) / (grid[i+1]-grid[i])*data_points[i] + (x_segment - grid[i]) / (grid[i+1]-grid[i])*data_points[i+1]
            x = np.concatenate((x,x_segment))
            y = np.concatenate((y,y_segment))

        return x, y
    
    def main(self):
        grid = self.gen_equispaced_grid()
        data_points = self.gen_data_points(grid)
        x,y =  self.spline_interpolate(grid,data_points)
        return x, y 


