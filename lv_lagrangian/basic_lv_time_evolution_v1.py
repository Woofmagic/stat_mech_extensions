"""
Use this script to make plots of standard Lotka-Volterra
dynamical equation evolution.

Date initialized: 20260312
Last edited: 20260312

Notes:
    1. 20260312:
        - Where do we get the values of the parameters and initial conditions
        that we are using here? We are getting them from this Wikipedia figure:
        https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations#/media/File:Lotka-Volterra_model_(1.1,_0.4,_0.4,_0.1).svg
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# 3rd Party Library | SciPy:
from scipy.integrate import solve_ivp

#################################
# File Information
#################################

# (X): Define a version number so we don't get confused:
_version_number = "1.2"

# (X): Dynamically set the plot title using the version number:
PLOT_TITLE = f"lv_dynamics_generic_params_v{_version_number}"

#################################
# Plotting Configuration
#################################
# (X): We tell rcParams to use LaTeX. [NOTE]: this will *crash* your 
# | version of the code if you do not have TeX distribution installed!
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True    
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True

#################################
# Simulation Parameters
#################################
PARAMETER_ALPHA = 1.1 # prey exponential increase
PARAMETER_BETA = 0.4 # predator hunting "power
PARAMETER_GAMMA = 0.4 # predator exponential decrease
PARAMETER_DELTA = 0.1 # prey presence measurement:

INITIAL_PREY = 10
INITIAL_PREDATORS = 10

TIME_STARTING_VALUE = 0 #simulation's starting "t" value
TIME_ENDING_VALUE = 100 #simulation's ending "t" value
NUMBER_OF_TIME_SLICES = 1000 #simulation's Δt

TIME_INTERVAL = (TIME_STARTING_VALUE, TIME_ENDING_VALUE)
TIME_SLICES = np.linspace(*TIME_INTERVAL, NUMBER_OF_TIME_SLICES)

#################################
# Functions:
#################################

def first_order_ode(t, vector_of_equations):
    """
    ## Description:
    Numerically evaluate a system of first-order, ordinary differential equations (ODE).
    """
    x, y = vector_of_equations
    x_dot = PARAMETER_ALPHA * x - PARAMETER_BETA * x * y
    y_dot = - PARAMETER_GAMMA * y + PARAMETER_DELTA * x * y
    return [x_dot, y_dot]

#################################
# Main program:
#################################

print("[INFO]: Now solving IVP...")

numerical_solution = solve_ivp(
    fun = first_order_ode,  # *the* function we are numerically integrating:
    t_span = TIME_INTERVAL,
    y0 = [INITIAL_PREY, INITIAL_PREDATORS],
    t_eval = TIME_SLICES)

time_axis = numerical_solution.t # extract the time "history"
prey_time_evolution = numerical_solution.y[0]
predators_time_evolution = numerical_solution.y[1]

#################################
# Plotting flow:
#################################

print("[INFO]: Now plotting stuff...")

figure_instance, axis_instance =  plt.subplots(
    nrows = 1, ncols = 1, figsize = (10, 5.5))

axis_instance.plot(time_axis, prey_time_evolution, color = 'blue', label = "Prey Population")
axis_instance.plot(time_axis, predators_time_evolution, color = 'orange', label = "Predator Population")

axis_instance.set_xlabel("Time", rotation = 0, labelpad = 5.0, fontsize = 18)
axis_instance.set_ylabel("Population", rotation = 90, labelpad = 17.0, fontsize = 18)
axis_instance.set_title(
    fr"Lotka-Volterra Dynamical Evolution with $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$", 
    fontsize = 18)

axis_instance.set_ylim(ymin = 0., ymax = np.max(prey_time_evolution) + 5.0, )
axis_instance.set_xlim(xmin = TIME_STARTING_VALUE - 2.0, xmax = TIME_ENDING_VALUE + 2.0)
axis_instance.tick_params(labelsize = 17)

axis_instance.legend(fontsize = 17)

plt.savefig(f"{PLOT_TITLE}.eps", format = "eps")
plt.savefig(f"{PLOT_TITLE}.svg", format = "svg")
plt.close(figure_instance)

print("[INFO]: End of script reached!")
