"""
We are using this script to visualize the LV dynamics in their (i) state-space
represention, (ii) phase-space representation, (iii) rotated phase-space
representation.

Date initialized: 20260508
Last edited: 20260508

Notes:
    1. 20260312:
        - Where do we get the values of the parameters and initial conditions
        that we are using here? We are getting them from this Wikipedia figure:
        https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations#/media/File:Lotka-Volterra_model_(1.1,_0.4,_0.4,_0.1).svg
"""

##################################################
# Import 3rd-party libraries:
##################################################
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

##################################################
# File Information
##################################################

# Define a version number so we don't get confused:
_version_number = "1_2"

##################################################
# Plotting Configuration
##################################################
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

##################################################
# Simulation Parameters
##################################################
PARAMETER_ALPHA = 2.0/3.0 # prey exponential increase
PARAMETER_BETA =  4.0/3.0 # predator hunting "power
PARAMETER_GAMMA = 1.0 # predator exponential decrease
PARAMETER_DELTA = 1.0 # prey presence measurement:

INITIAL_PREY = 0.01
INITIAL_PREDATORS = 0.

TIME_STARTING_VALUE = 0 #simulation's starting "t" value
TIME_ENDING_VALUE = 20 #simulation's ending "t" value
NUMBER_OF_TIME_SLICES = 5000 #simulation's Δt

TIME_INTERVAL = (TIME_STARTING_VALUE, TIME_ENDING_VALUE)
TIME_SLICES = np.linspace(*TIME_INTERVAL, NUMBER_OF_TIME_SLICES)

##################################################
# Functions:
##################################################

def lotka_volterra_equations_2d(t, vector_of_equations):
    """
    ## Description:
    Numerically evaluate a system of first-order, ordinary differential equations (ODE).
    """
    x, y = vector_of_equations
    x_dot = PARAMETER_ALPHA * x - PARAMETER_BETA * x * y
    y_dot = - PARAMETER_GAMMA * y + PARAMETER_DELTA * x * y
    return [x_dot, y_dot]

##################################################
# Numerical integration
##################################################

numerical_solutions_list = []

number_of_initial_conditions = 30

print("[INFO]: Now solving IVP...")

list_of_initial_conditions = [
    [INITIAL_PREY + i * 0.1, INITIAL_PREDATORS + i * 0.1] for i in range(0, number_of_initial_conditions)]

for initial_conditions in list_of_initial_conditions:
    given_solution = solve_ivp(
        fun = lotka_volterra_equations_2d, # *the* function we are numerically integrating:
        t_span = TIME_INTERVAL, 
        y0 = initial_conditions, 
        t_eval = TIME_SLICES,
        # due to some issues with integral curves not closing,
        # we needed to increase these tolerances...
        rtol = 1e-9,
        atol = 1e-12)
    
    # store the numerical integration:
    numerical_solutions_list.append({
        "initial_conditions": initial_conditions,
        "prey_dynamics": given_solution.y[0],
        "predator_dynamics": given_solution.y[1],
        "time": given_solution.t
    })

##################################################
# Plotting in **ORIGINAL COORIDNATES**
##################################################

print("[INFO]: Now plotting stuff...")

colors = plt.cm.winter(np.linspace(0, 1, number_of_initial_conditions))

figure_instance, axis_instance =  plt.subplots(
    nrows = 1, ncols = 1, figsize = (7.0, 7.0))

for index, numerical_solution in enumerate(numerical_solutions_list):

    prey_initial_condition = numerical_solution["prey_dynamics"][0]
    predator_initial_condition = numerical_solution["predator_dynamics"][0]
    
    # the integral curve
    axis_instance.plot(
        numerical_solution["prey_dynamics"], numerical_solution["predator_dynamics"],
        color = colors[index], alpha = 0.85, linewidth = 1.)
    
    # the initial condition points:
    axis_instance.scatter(
        prey_initial_condition, predator_initial_condition, 
        facecolor = 'white',  edgecolor = 'black',  linewidth = 0.9,  s = 8, zorder = 5)

# this is the nontrivial fixed point
axis_instance.scatter(
    PARAMETER_GAMMA / PARAMETER_DELTA, PARAMETER_ALPHA / PARAMETER_BETA,
    c = "red", s = 25,  edgecolor = "black")

axis_instance.set_xlabel(r"Prey Density ($x$)", rotation = 0.0, labelpad = 5.0, fontsize = 18.)
axis_instance.set_ylabel(r"Predators Density ($y$)", rotation = 90.0, labelpad = 17.0, fontsize = 18.)
axis_instance.set_title(
    fr"State Space Portrait with "
    fr"$\alpha = {PARAMETER_ALPHA:3g}, \beta = {PARAMETER_BETA:3g}, \gamma = {PARAMETER_GAMMA:3g}, \delta = {PARAMETER_DELTA:3g}$", 
    fontsize = 18.)

axis_instance.set_ylim(
    ymin = 0.0,
    # have to pad the maximum:
    ymax = max(np.max(integral_curve["predator_dynamics"]) for integral_curve in numerical_solutions_list) + 0.2)
axis_instance.set_xlim(
    xmin = 0.0,
    # have to pad the maximum:
    xmax = max(np.max(integral_curve["prey_dynamics"]) for integral_curve in numerical_solutions_list) + 0.2)

figure_instance.savefig(f"lv_foliated_state_portrait_v{_version_number}.eps", format = "eps")
figure_instance.savefig(f"lv_foliated_state_portrait_v{_version_number}.svg", format = "svg")
plt.close(figure_instance)

del figure_instance
del axis_instance

##################################################
# Plotting in **LOGARITHMIC COORIDNATES**
# i.e. all we are doing is *taking the log* of the
# original coordinates as we defined them!
##################################################

figure_instance, axis_instance =  plt.subplots(
    nrows = 1, ncols = 1, figsize = (7.0, 7.0))

for index, numerical_solution in enumerate(numerical_solutions_list):

    prey_initial_condition = numerical_solution["prey_dynamics"][0]
    predator_initial_condition = numerical_solution["predator_dynamics"][0]
    
    # the integral curve
    axis_instance.plot(
        np.log(numerical_solution["prey_dynamics"]), np.log(numerical_solution["predator_dynamics"]),
        color = colors[index], alpha = 0.7, linewidth = 1.)
    
    # the initial condition points:
    axis_instance.scatter(
        np.log(prey_initial_condition), np.log(predator_initial_condition), 
        facecolor = 'white',  edgecolor = 'black',  linewidth = 0.9,  s = 8, zorder = 5)

# this is the logarithm of the nontrivial fixed point
axis_instance.scatter(
    np.log(PARAMETER_GAMMA / PARAMETER_DELTA), np.log(PARAMETER_ALPHA / PARAMETER_BETA),
    c = "red", s = 25,  edgecolor = "black")

axis_instance.set_xlabel(r"Logarithm of Prey Density ($\ln(x)$)", rotation = 0.0, labelpad = 5.0, fontsize = 18.)
axis_instance.set_ylabel(r"Logarithm of Predators Density ($\ln(y)$)", rotation = 90.0, labelpad = 17.0, fontsize = 18.)
axis_instance.set_title(
    fr"Phase Space Portrait with "
    fr"$\alpha = {PARAMETER_ALPHA:3g}, \beta = {PARAMETER_BETA:3g}, \gamma = {PARAMETER_GAMMA:3g}, \delta = {PARAMETER_DELTA:3g}$", 
    fontsize = 18.)

figure_instance.savefig(f"lv_foliated_phase_portrait_v{_version_number}.eps", format = "eps")
figure_instance.savefig(f"lv_foliated_phase_portrait_v{_version_number}.svg", format = "svg")
plt.close(figure_instance)

del figure_instance
del axis_instance

print("[INFO]: End of script reached!")
