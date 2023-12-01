import numpy as np

def newton_raphson(func,deriv,interval=[-1* np.infty,np.infty],tol=10**(-6),init_guess=None):
    if init_guess == None:
        x = interval[0]
    else:
        x = init_guess
        
    iter = 0
    while abs(func(x)) > tol:
        x = x - func(x) / deriv(x)
        iter += 1
    return x

def theta_method(func,init_val,t,dt,theta=0):
    r"""
    function to solve first order ODE with theta method
    theta = 0 corresponds to explicit Euler method
    theta = 1 corresponds to implicit Euler method
    
    Parameters
    ----------
    func: function representing derivative of u
          input: (u,t) output:u'
    init_val: float
          initial value
    t: float
          maximum value of time 
    dt: float(dividing t)
          width of timestep
    
    Returns
    ----------
    u: numpy array
          solution of the differential scheme
    """

    num_step = int(t/dt) + 1
    u = np.zeros(num_step)
    u[0] = init_val

    if theta == 0:
        for i in range(1,num_step):
            u[i] = u[i-1] + dt * func(u[i-1],dt*(i-1))
    else:
        for i in range(1,num_step):
            eq = lambda x: (
                x 
                - u[i-1] 
                - dt * (theta * func(x,i * dt) 
                +(1-theta) * func(u[i-1],(i-1) * dt))
                )
            d = 10 ** (-10)# increment to use in numerical differentiation
            der = lambda y: (eq(y+d) - eq(y))/d
            new_u = newton_raphson(eq,der,init_guess=u[i-1])
            u[i] = new_u

    return u

def improved_euler(func,init_val,t,dt):
    r"""
    function to solve first order ODE with improved Euler method

    Parameters
    ----------
    func : function representing derivative of u
           input: (u,t) output: u'
    init_val : float
           initial value
    t : float
           maximum value of time 
    dt : float (dividing t)
           width of timestep

    Returns
    ----------
    u : numpy array
         solution of the differential scheme
    """
    num_step = int(t/dt) + 1
    u = np.zeros(num_step)
    u[0] = init_val
    for i in range(1,num_step):
        approx_u = u[i-1]+dt *func(u[i-1],dt*(i-1))
        u[i] = (
            u[i-1] 
            + (0.5) * dt * (func(u[i-1],dt*(i-1)) 
            + func(approx_u,i*dt))
            )
    return u

def runge_kutta_four(func,init_val,t,dt):
    r"""
    Function to solve first order ODE with RK4 scheme

    Parameters
    ----------
    func : function
        Function representing the derivative of u. 
        Input: (u, t), Output: u'.
    init_val : float
        Initial value.
    t : float
        Maximum value of time.
    dt : float
        Width of timestep (time interval divided by t).

    Returns
    ----------
    u : numpy array
        Solution of the differential scheme.
    """

    num_step = int(t/dt) + 1
    u = np.zeros(num_step)
    u[0] = init_val
    prior_time = 0
    for i in range(1,num_step):
        k = []
        k.append(func(u[i-1],prior_time)) #k_1
        k.append(func(u[i-1]+dt*0.5*k[0],prior_time+0.5*dt)) #k_2
        k.append(func(u[i-1]+dt*0.5*k[1],prior_time+0.5*dt)) #k_3
        k.append(func(u[i-1]+dt*k[2],prior_time+dt)) #k_4
        u[i] = u[i-1] + dt * (1/6)*(k[0]+2*k[1]+2*k[2]+k[3])
        prior_time += dt
    return u

def adams_bashforth(func,init_val,t,dt,generate_init="RK4"):
    num_step = int(t/dt) + 1
    u = np.zeros(num_step)
    u[0] = init_val
    prior_time = 0

    #obtain the first four terms of the scheme with RK4 or explicit Euler scheme
    if generate_init == "RK4":
        u[0:4] = runge_kutta_four(func,init_val,3*dt,dt)
    elif generate_init == "Euler":
        u[0:4] = theta_method(func,init_val,3*dt,dt,theta=0)
    else:
        print("error: generate_init should be either RK4 or Euler")
    prior_time = 3 * dt

    for i in range(4,num_step):
        u[i] = (
            u[i-1] 
            + (1/24) * dt*(55* func(u[i-1],prior_time) 
            - 59* func(u[i-2],prior_time-dt) 
            + 37* func(u[i-3],prior_time-2*dt) 
            - 9* func(u[i-4],prior_time-3*dt))
            )

        prior_time += dt
    return u
    
def adams_moulton(func,func_der,init_val,t,dt,generate_init = "RK4"):
    num_step = int(t/dt) + 1
    u = np.zeros(num_step)
    u[0] = init_val
    prior_time = 0

    #obtain the first four terms of the scheme with RK4 or explicit Euler scheme
    if generate_init == "RK4":
        u[0:4] = runge_kutta_four(func,init_val,3*dt,dt)
    elif generate_init == "Euler":
        u[0:4] = theta_method(func,init_val,3*dt,dt,theta=0)
    else:
        print("error: generate_init should be either RK4 or Euler")
    prior_time = 3 * dt

    for i in range(4,num_step):
        eq = lambda x: (
            x 
            - u[i-1] 
            - (1/24) * dt*(9* func(x,prior_time+dt) 
            +19* func(u[i-1],prior_time) 
            -5* func(u[i-2],prior_time-dt) 
            + func(u[i-3],prior_time-2*dt))
            )
        #could just enter the explicit derivative
        d = 10 ** (-10)# increment to use in numerical differentiation
        #der = lambda y: (eq(y+d) - eq(y))/d
        der = lambda y: 1 - (1/24) * dt * 9 * func_der(y,prior_time+dt)
        new_u = newton_raphson(eq,der,tol=10**(-10),init_guess=u[i-1])
        u[i] = new_u
        prior_time += dt
    return u



def implicit_euler_second(func,init_val,t,dt):
    r"""
    Function to solve second order ODE with implicit Euler method

    Parameters
    ----------
    func : function
        Function representing the derivative of u.
        Input: ([u, u'], t), Output: [u', u''].
    init_val : numpy array
        Initial values [u_0, u'_0].
    t : float
        Maximum value of time.
    dt : float
        Width of timestep (time interval divided by t).

    Returns
    ----------
    u : numpy array
        Solution of the differential scheme.
    """
    num_step = int(t/dt)
    u = np.zeros(num_step)
    u[0] = init_val[0]
    prev_u = init_val[0]
    prev_u_prime = init_val[1]

    for i in range(1,num_step):
        eq = lambda x: (x 
                        - prev_u_prime 
                        - dt * (func(prev_u + dt * x,x,dt * i))
                        )
        d = 10 ** (-10)# increment to use in numerical differentiation
        der = lambda y: (eq(y+d) - eq(y))/d
        new_u_prime = newton_raphson(eq,der,init_guess=prev_u_prime)
        new_u = prev_u + dt * new_u_prime
        u[i] = new_u

        prev_u = new_u
        prev_u_prime = new_u_prime

    return u