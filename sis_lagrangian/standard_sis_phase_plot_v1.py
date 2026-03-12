"""
The script that was used to generate (i) plots showing the kinematics associated with the 
derived second-order ODE that represents the single "SIS particle" and (ii) plots showing the
"potential well" that we have interpreted the SIS particle to be moving within.

Date initialized: 20250701
Last edited: 20260312
"""

#################################
# Importing Libraries
#################################

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# 3rd Party Library | Matplotlib:
from matplotlib.collections import LineCollection

# 3rd Party Library | Matplotlib:
from matplotlib.colors import Normalize

# 3rd Party Library | SciPy:
from scipy.integrate import solve_ivp

#################################
# File Information
#################################

_version_number = "1.2"
PLOT_TITLE = f"sis_phase_dynamics_generic_params_v{_version_number}"

# (X): Fix the plot directory for the SIS analysis:
# PLOT_DIRECTORY = "plots"

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
TIME_ENDING_VALUE = 10

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
    nrows = 1, ncols = 1, sharex = True, figsize = (10, 5.5))

points = np.array([susceptible_per_time, infected_per_time]).T.reshape(-1, 1, 2)

segments = np.concatenate([points[:-1], points[1:]], axis = 1)

norm = Normalize(time_axis.min(), time_axis.max())
lc = LineCollection(segments, cmap = 'plasma', norm = norm)
lc.set_array(time_axis)
lc.set_linewidth(2)
line = axis_instance.add_collection(lc)

axis_instance.set_xlabel(r"Susceptible $S$")
axis_instance.set_ylabel(r"Infected $I$")
axis_instance.set_title(fr"SIS Phase Evolution with $\beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}$", fontsize = 18)

axis_instance.set_xlim(susceptible_per_time.min(), susceptible_per_time.max())
axis_instance.set_ylim(infected_per_time.min(), infected_per_time.max())
axis_instance.tick_params(labelsize = 17)
axis_instance.grid(True)


plt.savefig(f"{PLOT_TITLE}.eps", format = "eps")
plt.savefig(f"{PLOT_TITLE}.svg", format = "svg")

plt.close(figure_instance)
