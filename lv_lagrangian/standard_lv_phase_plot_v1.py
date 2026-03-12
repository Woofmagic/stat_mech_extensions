"""
The script that was used to generate (i) plots showing the kinematics associated with the 
derived second-order ODE that represents the single "LV particle" and (ii) plots showing the
"potential well" that we have interpreted the LV particle to be moving within.

Date initialized: 20260312
Last edited: 20260312
"""

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
PLOT_TITLE = f"lv_phase_dynamics_generic_params_v{_version_number}"

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
PARAMETER_ALPHA = 0.66666 # prey exponential increase
PARAMETER_BETA = 1.33333 # predator hunting "power
PARAMETER_GAMMA = 1.0 # predator exponential decrease
PARAMETER_DELTA = 1.0 # prey presence measurement:

INITIAL_PREY = 1.0
INITIAL_PREDATORS = 0.08

TIME_STARTING_VALUE = 0 #simulation's starting "t" value
TIME_ENDING_VALUE = 10 #simulation's ending "t" value
NUMBER_OF_TIME_SLICES = 1000 #simulation's Δt

TIME_INTERVAL = (TIME_STARTING_VALUE, TIME_ENDING_VALUE)
TIME_SLICES = np.linspace(*TIME_INTERVAL, NUMBER_OF_TIME_SLICES)

#################################
# Functions:
#################################

def second_order_ode(t, vector_of_equations):
    """
    ## Description:
    Numerically evaluate a second-order, ordinary differential equation (ODE)
    using the technique of reduction-of-order.

    ## Notes:
    The technique of reduction-of-order is nothing more than regarding the 
    first derivative of the dynamical variable, say x', as an entirely *separate* variable 
    according to the assignment below:
    (i) x' = v, (ii) v' = f(x), where (iii) x'' = f(x), the original ODE.
    (Primes are time-derivatives, of course.)
    """
    x, y = vector_of_equations
    x_dot = PARAMETER_ALPHA * x - PARAMETER_BETA * x * y
    y_dot = -PARAMETER_GAMMA * y + PARAMETER_DELTA * x * y
    return [x_dot, y_dot]

# (X): Use SciPy's IVP numerical integrator to do the calculation.
# | This is *the* actual simulation. This is where it happens:
numerical_solution = solve_ivp(
    fun = second_order_ode, 
    t_span = TIME_INTERVAL, 
    y0 = [INITIAL_PREY, INITIAL_PREDATORS],
    t_eval = TIME_SLICES)

time_axis = numerical_solution.t
prey_density = numerical_solution.y[0]
predator_density = numerical_solution.y[1]
a = PARAMETER_ALPHA * PARAMETER_GAMMA + (PARAMETER_ALPHA - PARAMETER_BETA * np.exp(prey_density)) * predator_density - PARAMETER_BETA * PARAMETER_GAMMA * np.exp(prey_density)

#################################
# Plotting flow:
#################################

figure_instance, axis_instance =  plt.subplots(
    nrows = 1, ncols = 1, figsize = (10, 5.5))

points = np.array([prey_density, predator_density]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis = 1)

norm = Normalize(time_axis.min(), time_axis.max())
lc = LineCollection(segments, cmap = 'plasma', norm = norm)
lc.set_array(time_axis)
lc.set_linewidth(2)
line = axis_instance.add_collection(lc)

axis_instance.set_xlabel(r"Prey Density $x$", rotation = 0, labelpad = 5.0, fontsize = 18)
axis_instance.set_ylabel(r"Predator Density $y$", rotation = 90, labelpad = 17.0, fontsize = 18)
axis_instance.set_title(
    fr"LV State Space Evolution with $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$", 
    fontsize = 18)

axis_instance.set_xlim(prey_density.min() - 0.5, prey_density.max() + 0.5)
axis_instance.set_ylim(predator_density.min() - 0.5, predator_density.max() + 0.5)
axis_instance.tick_params(labelsize = 17)

axis_instance.grid(True)

plt.savefig(f"{PLOT_TITLE}.eps", format = "eps")
plt.savefig(f"{PLOT_TITLE}.svg", format = "svg")
plt.close(figure_instance)

print("[INFO]: End of script reached!")
