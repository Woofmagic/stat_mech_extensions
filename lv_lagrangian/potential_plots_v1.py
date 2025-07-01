"""
The script that was used to visualize the "potential well" that we interpret the 
"predator particle" to move within. In classical physics, the potential well picture
is quite informative, so it helps to see (if you can!) what the particle is moving "in."
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# (X): Define a version number so we don't get confused:
_version_number = "2.4"

# (X): Dynamically set the plot title using the version number:
PLOT_TITLE = f"lv_potential_generic_params_v{_version_number}"

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
PARAMETER_ALPHA = 0.66666

# (X): Simulation Parameter | predator hunting "power":
PARAMETER_BETA = 1.33333

# (X): Simulation Parameter | predator exponential decrease:
PARAMETER_GAMMA = 1.0

# (X): Simulation Parameter | prey presence measurement:
PARAMETER_DELTA = 1.0

# (X): Simulation Parameter | smallest value for "q" (predator "position"):
LOWEST_Q_VALUE = -3.1

# (X): Simulation Parameter | largest value for "q" (predator "position"):
HIGHEST_Q_VALUE = 3.1

# (X): Simulation Parameter | number of "position slices":
NUMBER_OF_POSITION_SLICES: 501

# (X): Obtain an array (iterable) that allows plug-and-chug evaluation of a function defined on itL
q_values = np.linspace(LOWEST_Q_VALUE, HIGHEST_Q_VALUE, NUMBER_OF_POSITION_SLICES)

# (X): Code in the form of the linear piece of the potential:
linear_potential = - PARAMETER_GAMMA * PARAMETER_ALPHA * q_values

# (X): Code in the form of the exponential piece of the potential:
exponential_potential = PARAMETER_GAMMA * PARAMETER_BETA * np.exp(q_values)

# (X): Code the entire potential function, U(q):
potential_function = exponential_potential + linear_potential

# (X): Set up the Figure instance
figure_instance = plt.figure(figsize = (10, 5.5))

# (X): Add an Axes Object:
axis_instance = figure_instance.add_subplot(1, 1, 1)

axis_instance.set_xlabel(xlabel = r"$q$", fontsize = 17, labelpad = 4.,)
axis_instance.set_ylabel(ylabel = r"$U \left( q \right)$", fontsize = 17, labelpad = 25.0, rotation = 0)

# axis_instance.vlines(
#     x = np.log(alpha / beta),
#     ymin = min(np.min(potential_function), np.min(linear_potential), np.min(exponential_potential)),
#     ymax = max(np.max(potential_function), np.max(linear_potential), np.max(exponential_potential)),
#     colors = "gray",
#     linestyles = '--')

axis_instance.vlines(
    x = 0.,
    ymin = min(np.min(potential_function), np.min(linear_potential), np.min(exponential_potential)),
    ymax = max(np.max(potential_function), np.max(linear_potential), np.max(exponential_potential)),
    colors = "black",
    linestyles = '--',
    alpha = 0.5)

axis_instance.hlines(
    y = 0.,
    xmin = np.min(q_values),
    xmax = np.max(q_values),
    colors = "black",
    linestyles = '--',
    alpha = 0.5)

axis_instance.plot(
    q_values, 
    potential_function, 
    color = "black",
    linestyle = '-',
    linewidth = 1.5,
    label = r"$U(q) = \gamma \left( \beta e^{q} - \alpha q \right)$")

axis_instance.plot(
    q_values,
    linear_potential,
    color = "blue",
    linestyle = '--',
    linewidth = 1.5,
    label = r"linear: $- \alpha \gamma q$")

axis_instance.plot(
    q_values, 
    exponential_potential, 
    color = "red", 
    linestyle = '--', 
    linewidth = 1.5,
    label = r"exponential: $\beta \gamma e^{q}$")

axis_instance.legend(
    fontsize = 18)

axis_instance.set_title(
    fr"Potential Well for $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$",
    fontsize = 18
)

axis_instance.set_xlim(xmin = LOWEST_Q_VALUE - 0.1, xmax = HIGHEST_Q_VALUE + 0.1)

axis_instance.scatter(
    np.log(PARAMETER_ALPHA / PARAMETER_BETA),
    np.min(potential_function),
    c = "purple",
    s = 25)

axis_instance.text(
    x = np.log(PARAMETER_ALPHA / PARAMETER_BETA),
    y = np.min(potential_function) + 0.037,
    s = r"$\ln \left( \alpha / \beta \right)$",
    c = "purple",
    fontsize = 16)

axis_instance.tick_params(labelsize = 17)

# (X): Save a version of the figure according to .eps format for Overleaf stuff:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.eps", format = "eps")
plt.savefig(f"{PLOT_TITLE}.eps", format = "eps")

# (X): Save an immediately-visualizable figure with vector graphics:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.svg", format = "svg")
plt.savefig(f"{PLOT_TITLE}.svg", format = "svg")

# (X): Closing figures:
plt.close()

# https://stackoverflow.com/a/24988486 -> reminder on vertical lines
# https://stackoverflow.com/a/27474400 -> for why Matplotlib rejected LaTeX
# https://stackoverflow.com/a/14716726 -> for adding custom x-ticks
# https://stackoverflow.com/a/43478214 -> for strange TypeErrors in tick labels