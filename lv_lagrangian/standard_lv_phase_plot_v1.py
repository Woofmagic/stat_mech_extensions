"""
The script that was used to generate (i) plots showing the kinematics associated with the 
derived second-order ODE that represents the single "LV particle" and (ii) plots showing the
"potential well" that we have interpreted the LV particle to be moving within.
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

# (X): Define a version number so we don't get confused:
_version_number = "1.1"

# (X): Dynamically set the plot title using the version number:
PLOT_TITLE = f"lv_phase_dynamics_generic_params_v{_version_number}"

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

# (X): Simulation Parameter | prey exponential increase:
PARAMETER_ALPHA = 0.66666

# (X): Simulation Parameter | predator hunting "power":
PARAMETER_BETA = 1.33333

# (X): Simulation Parameter | predator exponential decrease:
PARAMETER_GAMMA = 1.0

# (X): Simulation Parameter | prey presence measurement:
PARAMETER_DELTA = 1.0

# (X): Simulation Parameter | predator's "initial position"
# | This corresponds to the predator's initial population density
# | in the original model.
INITIAL_POSITION = 1.0

# (X): Simulation Parameter | predator's "initial velocity"
# | This corresponds the 
INITIAL_VELOCITY = 0.08

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
    # (1): Unpack the standard "vector" of dynamical variables: x^[T] rep. as (x, v):
    x, y = vector_of_equations

    # (2): As per reduction of order, x' is nothing but v:
    x_dot = PARAMETER_ALPHA * x - PARAMETER_BETA * x * y

    # (3): As per reduction of order, v' is nothing but x'', and it
    # | *is* the standard ODE:
    y_dot = -PARAMETER_GAMMA * y + PARAMETER_DELTA * x * y
    
    # (4): Return an "effective vector" (list) that is just dot(x)^{T} = (x', v'):
    return [x_dot, y_dot]

# (X): Use SciPy's IVP numerical integrator to do the calculation.
# | This is *the* actual simulation. This is where it happens:
numerical_solution = solve_ivp(

    # (X.1): It needs *the* function we are numerically integrating:
    fun = second_order_ode, 

    # (X.2): It needs the interval of the real line (time axis) to do the integration over:
    t_span = TIME_INTERVAL, 

    # (X.3): It needs the initial conditions as a *vector* (list):
    y0 = [INITIAL_POSITION, INITIAL_VELOCITY],

    # (X.4): Needs the total number of time-steps to numerically evaluate the system:
    t_eval = TIME_SLICES)

# (X): As part of SciPy's `solve_ivp`, extract the time "history":
time_axis = numerical_solution.t

# (X): Same as above, but extract the first component of the y *vector* (list):
prey_density = numerical_solution.y[0]

# (X): Same as above, but now the second component:
predator_density = numerical_solution.y[1]

# (X): We solved for x and v, and *now* we can plug-and-chug for the corresponding x'':
a = PARAMETER_ALPHA * PARAMETER_GAMMA + (PARAMETER_ALPHA - PARAMETER_BETA * np.exp(prey_density)) * predator_density - PARAMETER_BETA * PARAMETER_GAMMA * np.exp(prey_density)

# (1): Set up the Figure instance and the *two* Axes objects:
figure_instance, axis_instance =  plt.subplots(
    nrows = 1, 
    ncols = 1, 
    sharex = True,
    figsize = (10, 5.5))

# (X): The rest of the script just makes the plots. We will comment them better later:
points = np.array([prey_density, predator_density]).T.reshape(-1, 1, 2)

segments = np.concatenate([points[:-1], points[1:]], axis = 1)

norm = Normalize(time_axis.min(), time_axis.max())
lc = LineCollection(segments, cmap = 'plasma', norm = norm)
lc.set_array(time_axis)
lc.set_linewidth(2)
line = axis_instance.add_collection(lc)

axis_instance.set_xlabel(r"Prey Density $x$")
axis_instance.set_ylabel(r"Predator Density $y$")
axis_instance.set_title(fr"LV Phase Evolution with with $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$", fontsize = 18)
axis_instance.set_xlim(prey_density.min(), prey_density.max())
axis_instance.set_ylim(predator_density.min(), predator_density.max())
axis_instance.tick_params(labelsize = 17)

# (X): Add a grid for clarity:
plt.grid(True)

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
