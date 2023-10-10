import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import Rbf
import scipy.integrate

# Define the ordinary differential equation
def odefunc(t, y, params):
    a, b = params
    dydt = a * y[0] - b * y[0]**2
    return dydt

# Define the objective function to minimize
def objective(params):
    a, b = params
    initial_conditions = [1.0]  # Initial condition for y(0)
    t_span = (0, 10)  # Time span for the ODE solution
    sol = scipy.integrate.solve_ivp(odefunc, t_span, initial_conditions, args=(params,), t_eval=np.linspace(0, 10, 100))
    y_solution = sol.y[0]
    # Define the objective to minimize (e.g., maximize the final value of y)
    return -y_solution[-1]

# Generate sample data for fitting the surrogate model
n_samples = 20
a_values = np.random.uniform(0.5, 2.0, n_samples)
b_values = np.random.uniform(0.01, 0.2, n_samples)
objective_values = np.zeros(n_samples)

for i in range(n_samples):
    objective_values[i] = objective([a_values[i], b_values[i]])

#print(objective_values)

# Fit a response surface model (quadratic polynomial)
rbf = Rbf(a_values, b_values, objective_values, function='quintic')

# Define the surrogate objective function
def surrogate_objective(params):
    return -rbf(params[0], params[1])

# Initial guess for the parameters to optimize
initial_guess = [1.0, 0.1]

# Perform optimization using Response Surface Method (RSM)
result = minimize(surrogate_objective, initial_guess, method='Nelder-Mead')

# Extract the optimized parameters
optimized_params = result.x

# Print the optimized parameters
print("Optimized Parameters:", optimized_params)
