from linear_swimmer.linear_swimmer import LinearSwimmer
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#test
D = 25
radius = 3
epsilon = 0.1 * D
W = epsilon

r12 = lambda t : D - W * t
r23 = lambda t : D

linear_swimmer = LinearSwimmer(radius, r12,r23,W)
distance_travelled_D = linear_swimmer.auxilirary_move()
distance_travelled_D = distance_travelled_D[0]
print(distance_travelled_D)


r12 = lambda t : D - W * t
r23 = lambda t : D - epsilon

linear_swimmer = LinearSwimmer(radius, r12,r23,W)
distance_travelled_D_epsilon = linear_swimmer.auxilirary_move()
distance_travelled_D_epsilon = distance_travelled_D_epsilon[0]

distance_travelled = 2 * (distance_travelled_D - distance_travelled_D_epsilon)
print(distance_travelled)

#plot
# Define the model function
def model_function(epsilon, C):
    return C * (epsilon**2 + epsilon**3)

epsilon_list = np.linspace(0,0.8,20) * D
distance_travelled_list = []

for epsilon in epsilon_list:
    W = epsilon
    r12 = lambda t : D - W * t
    r23 = lambda t : D

    linear_swimmer = LinearSwimmer(radius, r12,r23,W)
    distance_travelled_D = linear_swimmer.auxilirary_move()
    distance_travelled_D = distance_travelled_D[0]
    print(distance_travelled_D)


    r12 = lambda t : D - W * t
    r23 = lambda t : D - epsilon

    linear_swimmer = LinearSwimmer(radius, r12,r23,W)
    distance_travelled_D_epsilon = linear_swimmer.auxilirary_move()
    distance_travelled_D_epsilon = distance_travelled_D_epsilon[0]

    distance_travelled = 2 * (distance_travelled_D - distance_travelled_D_epsilon)
    distance_travelled_list.append(distance_travelled)

distance_travelled_list = np.array(distance_travelled_list)
distance_travelled_normalised = distance_travelled_list * (1/radius)
epsilon_normalised = epsilon_list * (1/D)

# Perform curve fitting
popt, pcov = curve_fit(model_function, epsilon_normalised, distance_travelled_normalised)

# Extract the fitted parameter
C_fitted = popt[0]

# Generate fitted values for plotting
fitted_values = model_function(epsilon_normalised, C_fitted)

plt.plot(epsilon_normalised, distance_travelled_normalised, label='Original Data')
plt.plot(epsilon_normalised, fitted_values, label='Fitted Curve', linestyle='--')
plt.xlabel('Epsilon /D')
plt.ylabel('Traveled Distance/R')
plt.legend()
plt.show()