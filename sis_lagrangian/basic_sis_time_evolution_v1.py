"""
The script that was used to generate (i) plots showing the kinematics associated with the 
derived second-order ODE that represents the single "SIS particle" and (ii) plots showing the
"potential well" that we have interpreted the SIS particle to be moving within.
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# 3rd Party Library | SciPy:
from scipy.integrate import solve_ivp

# (X): Define a version number so we don't get confused:
_version_number = "1.5"

# (X): Dynamically set the plot title using the version number:
PLOT_TITLE = f"sis_kinematics_generic_params_v{_version_number}"

# (X): Fix the plot directory for the SIS analysis:
# PLOT_DIRECTORY = "plots"

# (X): We tell rcParams to use LaTeX. Note: this will *crash* your 
# | version of the code if you do not have TeX distribution installed!
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# (X): rcParams for the x-axis tick direction:
plt.rcParams['xtick.direction'] = 'in'

# (X): rcParams for the "major" (larger) x-axis vertical size:
plt.rcParams['xtick.major.size'] = 5

# (X): rcParams for the "major" (larger) x-axis horizonal width:
plt.rcParams['xtick.major.width'] = 0.5

# (X): rcParams for the "minor" (smaller) x-axis vertical size:
plt.rcParams['xtick.minor.size'] = 2.5

# (X): rcParams for the "minor" (smaller) x-axis horizonal width:
plt.rcParams['xtick.minor.width'] = 0.5

# (X): rcParams for the minor ticks to be *shown* versus invisible:
plt.rcParams['xtick.minor.visible'] = True

# (X): rcParams dictating that we want ticks along the x-axis on *top* (opposite side) of the bounding box:
plt.rcParams['xtick.top'] = True    

# (X): rcParams for the y-axis tick direction:
plt.rcParams['ytick.direction'] = 'in'

# (X): rcParams for the "major" (larger) y-axis vertical size:
plt.rcParams['ytick.major.size'] = 5

# (X): rcParams for the "major" (larger) y-axis horizonal width:
plt.rcParams['ytick.major.width'] = 0.5

# (X): rcParams for the "minor" (smaller) y-axis vertical size:
plt.rcParams['ytick.minor.size'] = 2.5

# (X): rcParams for the "minor" (smaller) y-axis horizonal width:
plt.rcParams['ytick.minor.width'] = 0.5

# (X): rcParams for the minor ticks to be *shown* versus invisible:
plt.rcParams['ytick.minor.visible'] = True

# (X): rcParams dictating that we want ticks along the y-axis on the *left* of the bounding box:
plt.rcParams['ytick.right'] = True

# (X): Simulation Parameter | probability of contracting illness:
PROBABILITY_OF_CONTRACTION = 0.1

# (X): Simulation Parameter | number of contacts individual makes per unit time:
NUMBER_OF_CONTACTS_PER_TIME = 2

# (X): Simulation Parameter | transmission probability::
PARAMETER_BETA = PROBABILITY_OF_CONTRACTION * NUMBER_OF_CONTACTS_PER_TIME

# (X): Simulation Parameter | average recovery rate:
PARAMETER_GAMMA = 0.1

# (X): Simulation Parameter | initial *percentage* of susceptible people:
INITIAL_SUSCEPTIBLE = 0.9

# (X): Simulation Parameter | initial *percentage* of infected people:
INITIAL_INFECTED = 0.1

# (X): Simulation Parameter | simulation's starting "t" value:
TIME_STARTING_VALUE = 0

# (X): Simulation Parameter | simulation's ending "t" value:
TIME_ENDING_VALUE = 100

# (X): Simulation Parameter | simulation's Δt:
NUMBER_OF_TIME_SLICES = 1000

# (X): Construct a tuple to use later in NumPy methods that 
# | represents the time interval over which the simulation will
# | take place.
TIME_INTERVAL = (TIME_STARTING_VALUE, TIME_ENDING_VALUE)

# (X): Unpack the tuple immediately and obtain an array of numerical
# | time-slices that we will plug-and-chug into the numerical ODE solver.
TIME_SLICES = np.linspace(*TIME_INTERVAL, NUMBER_OF_TIME_SLICES)

if (INITIAL_SUSCEPTIBLE + INITIAL_INFECTED != 1.0):
    raise ValueError(f"> Simulation parameters do not respect noramalization condition.")

def first_order_ode(t, vector_of_equations):
    """
    ## Description:
    Numerically evaluate a system of first-order, ordinary differential equations (ODE).
    """
    
    # (1): We first unpack the *vector* (list) of equations:
    x, y = vector_of_equations

    # (2): Here, we define the function that governs x', usually x' = f(x, y; t):
    x_dot = - PARAMETER_BETA * x * y + PARAMETER_GAMMA * y

    # (3): Same as above except for y' = g(x, y; t):
    y_dot = PARAMETER_BETA * x * y - PARAMETER_GAMMA * y

    # (4): Repackage the equations in a vector (list):
    return [x_dot, y_dot]

# (X): Use SciPy's IVP numerical integrator to do the calculation.
# | This is *the* actual simulation. This is where it happens:
numerical_solution = solve_ivp(

    # (X.1): It needs *the* function we are numerically integrating:
    fun = first_order_ode, 

    # (X.2): It needs the interval of the real line (time axis) to do the integration over:
    t_span = TIME_INTERVAL, 

    # (X.3): It needs the initial conditions as a *vector* (list):
    y0 = [INITIAL_SUSCEPTIBLE, INITIAL_INFECTED],

    # (X.4): Needs the total number of time-steps to numerically evaluate the system:
    t_eval = TIME_SLICES)

# (X): As part of SciPy's `solve_ivp`, extract the time "history":
time_axis = numerical_solution.t

# (X): Same as above, but extract the first component of the y *vector* (list):
susceptible_per_time = numerical_solution.y[0]

# (X): Same as above, but now the second component:
infected_per_time = numerical_solution.y[1]

# (1): Set up the Figure instance and the *two* Axes objects:
figure_instance, axis_instance =  plt.subplots(
    nrows = 1, 
    ncols = 1, 
    sharex = True,
    figsize = (10, 5.5))

# (X): The rest of the script just makes the plots. We will comment them better later:

axis_instance.plot(time_axis, susceptible_per_time, color = 'orange', label = "Percentage Susceptible")
axis_instance.plot(time_axis, infected_per_time, color = 'red', label = "Percentage Infected")
axis_instance.set_ylabel(r"$N$", rotation = 0, labelpad = 17.0, fontsize = 18)
axis_instance.set_title(fr"Compartmental Evolution with $\beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}$", fontsize = 18)
axis_instance.set_ylim(ymin = -0.1, ymax = 1.1)
axis_instance.set_xlim(xmin = TIME_STARTING_VALUE - 0.1, xmax = TIME_ENDING_VALUE + 0.1)
axis_instance.tick_params(labelsize = 17)

# (X): Add the legend for clarity:
plt.legend(fontsize = 17)

# (X): Tight layout... ya know:
plt.tight_layout(pad = 2.0)

# (X): Save a version of the figure according to .eps format for Overleaf stuff:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.eps", format = "eps")
plt.savefig(f"{PLOT_TITLE}.eps", format = "eps")

# (X): Save an immediately-visualizable figure with vector graphics:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.svg", format = "svg")
plt.savefig(f"{PLOT_TITLE}.svg", format = "svg")

# (X): Closing figures:
plt.close()
