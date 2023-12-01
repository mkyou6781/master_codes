import numpy as np


class Finite_Difference_Heat:
    r"""
    Class to solve the 1D heat equation using finite difference methods with the theta method.

    Parameters
    ----------
    init_con : function
        Function representing the initial condition. 
        Input: x, Output: initial temperature distribution.
    a_boundary : function
        Function representing the boundary condition at the start (x=0). 
        Input: t, Output: temperature.
    b_boundary : function
        Function representing the boundary condition at the end (x=x_end). 
        Input: t, Output: temperature.
    N : int
        Number of spatial discretization points.
    x_end : float
        Spatial domain endpoint.
    time_end : float
        Temporal domain endpoint.
    mu : float
        Parameter representing the ratio of time step over square of spatial step.
    theta : float, optional
        Parameter for theta methods (default 0, explicit).

    Methods
    -------
    mesh_construction()
        Constructs the spatial and temporal mesh grids.
    init_u()
        Initializes the solution matrix with zeros.
    init_val()
        Sets the initial condition values in the solution matrix.
    boundary_val()
        Applies the boundary conditions at each time step.
    two_dim_theta()
        Solves the heat equation using the theta method.
    main()
        Executes the solution process and returns the computed temperature distribution.

    Returns
    -------
    u : numpy array
        2D array containing the solution of the heat equation at each time and spatial discretization point.
    """

    
    def __init__(
        self, init_con, a_boundary, b_boundary, N, x_end, time_end, mu, theta=0
    ):
        self.init_con = init_con
        self.a_boundary = a_boundary
        self.b_boundary = b_boundary
        self.N = N
        self.x_end = x_end
        self.time_end = time_end
        self.theta = theta
        self.mu = mu

    def mesh_construction(self):
        self.x_mesh = np.linspace(0, self.x_end, self.N + 1, endpoint=True)
        # to set N+1 rather than N is important to impose boundary condition correctly
        dt = self.mu * (self.x_end / self.N) ** 2
        self.t_mesh = np.linspace(0, self.time_end, int(self.time_end / dt), endpoint=True)

    def init_u(self):
        self.u = np.zeros((len(self.t_mesh), len(self.x_mesh)))

    def init_val(self):
        self.u[0, :] = self.init_con(self.x_mesh)

    def boundary_val(self):
        self.u[:, 0] = self.a_boundary(self.t_mesh)
        self.u[:, -1] = self.b_boundary(self.t_mesh)

    def two_dim_theta(self):
        if self.theta == 0:
            for m in range(1, len(self.t_mesh)):
                for j in range(1, len(self.x_mesh) - 1):
                    self.u[m, j] = (
                        self.u[m - 1, j] + (
                            self.mu * (self.u[m - 1, j + 1] 
                            - 2 * self.u[m - 1, j]
                            + self.u[m - 1, j - 1])
                        )
                    )

        else:
            N = self.N + 1

            # obtain laplace matrix, the approximation of second spatial derivative
            diag = np.ones(N) * (-2)
            first_diag = np.ones(N - 1)
            laplace_matrix = np.diag(diag, k=0)
            laplace_matrix += np.diag(first_diag, k=1)
            laplace_matrix += np.diag(first_diag, k=-1)
            laplace_matrix[0, 0:2] = [0, 0]
            laplace_matrix[-1, N - 2 : N] = [0, 0]

            # prepare two identity matrix
            eye = np.identity(N)
            small_eye = np.identity(N)
            small_eye[0, 0] = 0
            small_eye[N - 1, N - 1] = 0

            for m in range(1, len(self.t_mesh)):
                # solve linear equation to obtain the value at t = t_{m}
                u_prev = self.u[m - 1, :]
                boundary_vector = np.zeros(N)
                boundary_vector[0] = self.u[m, 0]
                boundary_vector[-1] = self.u[m, -1]
                b_vector = (
                    np.matmul(
                        small_eye + self.mu * (1 - self.theta) * laplace_matrix
                        ,u_prev
                    )
                    + boundary_vector
                )

                self.u[m, :] = np.linalg.solve(
                    eye - self.mu * self.theta * laplace_matrix, b_vector
                )

    def main(self):
        self.mesh_construction()
        self.init_u()
        self.init_val()
        self.boundary_val()
        self.two_dim_theta()
        return self.u
