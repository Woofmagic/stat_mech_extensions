"""
The script that was used to generate (i) plots showing the kinematics associated with the 
derived second-order ODE that represents the single "LV particle" and (ii) plots showing the
"potential well" that we have interpreted the LV particle to be moving within.
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# 3rd Party Library | SciPy:
from scipy.integrate import solve_ivp

# (X): Define a version number so we don't get confused:
_version_number = "2.2"

# (X): Dynamically set the plot title using the version number:
PLOT_TITLE = f"lv_kinematics_generic_params_v{_version_number}"

# (X): Fix the plot directory for the LV analysis:
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
PARAMETER_ALPHA = 1.0

# (X): Simulation Parameter | predator hunting "power":
PARAMETER_BETA = 0.4

# (X): Simulation Parameter | predator exponential decrease:
PARAMETER_GAMMA = 0.1

# (X): Simulation Parameter | prey presence measurement:
PARAMETER_DELTA = 0.3

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
    position_value, velocity_value = vector_of_equations

    # (2): As per reduction of order, x' is nothing but v:
    x_dot = velocity_value

    # (3): As per reduction of order, v' is nothing but x'', and it
    # | *is* the standard ODE:
    v_dot = (
        PARAMETER_ALPHA * PARAMETER_GAMMA + 
        (PARAMETER_ALPHA - PARAMETER_BETA * np.exp(position_value)) * velocity_value - 
        PARAMETER_BETA * PARAMETER_GAMMA * np.exp(position_value)
        )
    
    # (4): Return an "effective vector" (list) that is just dot(x)^{T} = (x', v'):
    return [x_dot, v_dot]

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
t = numerical_solution.t

# (X): Same as above, but extract the first component of the y *vector* (list):
x = numerical_solution.y[0]

# (X): Same as above, but now the second component:
v = numerical_solution.y[1]

# (X): We solved for x and v, and *now* we can plug-and-chug for the corresponding x'':
a = PARAMETER_ALPHA * PARAMETER_GAMMA + (PARAMETER_ALPHA - PARAMETER_BETA * np.exp(x)) * v - PARAMETER_BETA * PARAMETER_GAMMA * np.exp(x)

# (1): Set up the Figure instance and the *two* Axes objects:
figure_instance, (axis_instance_position,
 axis_instance_velocity,
 axis_instance_acceleration) =  plt.subplots(
    nrows = 3, 
    ncols = 1, 
    sharex = True,
    figsize = (10, 5.5))

# (2): For an aesthetic feature, we define a horizonal line that shows the *stable equilibrium* point:
axis_instance_position.hlines(

    # (2.1): This value is solved for analytically and thus "hard-coded" in here:
    y = np.log(PARAMETER_ALPHA / PARAMETER_BETA),

    # (2.2): The line needs to be as long as the x-axis is, which means it starts at t = TIME_STARTING_VALUE...
    xmin = TIME_STARTING_VALUE,

    # (2.3); ... and ends at t = TIME_ENDING_VALUE.
    xmax = TIME_ENDING_VALUE,

    # (2.4): A dashed line:
    linestyle = '--',

    # (2.5): Purple will stand out:
    color = "purple",

    # (2.6): And we don't want it to be so big as to interrupt the main plot ingredients:
    linewidth = 0.6)

# (X): The rest of the script just makes the plots. We will comment them better later:

axis_instance_position.plot(t, x, color = 'green')
axis_instance_position.set_ylabel(r"$q(t)$", rotation = 0, labelpad = 17.0, fontsize = 18)
axis_instance_position.set_title(fr"Position vs. Time with $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$", fontsize = 18)
# axis_instance_position.set_ylim(ymin = 0.7, ymax = 1.3)
axis_instance_position.set_xlim(xmin = TIME_STARTING_VALUE - 0.1, xmax = TIME_ENDING_VALUE + 0.1)
axis_instance_position.tick_params(labelsize = 17)

axis_instance_velocity.plot(t, v, color = 'orange')
axis_instance_velocity.set_ylabel(r"$\dot{q}(t)$", rotation = 0, labelpad = 17.0, fontsize = 18)
axis_instance_velocity.set_title(fr"Velocity vs. Time with $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$", fontsize = 18)
# axis_instance_velocity.set_ylim(ymin = -0.08, ymax = 0.08)
axis_instance_velocity.set_xlim(xmin = TIME_STARTING_VALUE - 0.1, xmax = TIME_ENDING_VALUE + 0.1)
axis_instance_velocity.tick_params(labelsize = 17)

axis_instance_acceleration.plot(t, a, color = 'red')
axis_instance_acceleration.set_ylabel(r"$\ddot{q}(t)$", rotation = 0, labelpad = 17.0, fontsize = 18)
axis_instance_acceleration.set_xlabel(r"Time ($t$)", fontsize = 18)
axis_instance_acceleration.set_title(fr"Acceleration vs. Time with $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$", fontsize = 18)
# axis_instance_acceleration.set_ylim(ymin = -0.08, ymax = 0.08)
axis_instance_acceleration.set_xlim(xmin = TIME_STARTING_VALUE - 0.1, xmax = TIME_ENDING_VALUE + 0.1)
axis_instance_acceleration.tick_params(labelsize = 17)

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
