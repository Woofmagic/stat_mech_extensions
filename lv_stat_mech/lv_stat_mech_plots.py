"""
The script that was used to visualize the "potential well" that we interpret the 
"predator particle" to move within. In classical physics, the potential well picture
is quite informative, so it helps to see (if you can!) what the particle is moving "in."
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | ScIPy
from scipy.special import polygamma

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# (X): Define a version number so we don't get confused:
_version_number = "3.1"

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

inverse_temperature = np.linspace(start = 3.0, stop = 0.1, num = 10000)
temperature = np.linspace(start = 0.01, stop = 5.0, num = 10000)

# (X): We derived this analytically:
average_energy_vs_theta = (
    PARAMETER_ALPHA +
    PARAMETER_GAMMA +
    PARAMETER_ALPHA * np.log(PARAMETER_BETA * inverse_temperature) +
    PARAMETER_GAMMA * np.log(PARAMETER_DELTA * inverse_temperature) -
    PARAMETER_ALPHA * polygamma(0, PARAMETER_ALPHA * inverse_temperature) -
    PARAMETER_GAMMA * polygamma(0, PARAMETER_GAMMA * inverse_temperature)
    )

average_energy_vs_temperature = (
    PARAMETER_ALPHA +
    PARAMETER_GAMMA +
    PARAMETER_ALPHA * np.log(PARAMETER_BETA * (1./ temperature)) +
    PARAMETER_GAMMA * np.log(PARAMETER_DELTA * (1./ temperature)) -
    PARAMETER_ALPHA * polygamma(0, PARAMETER_ALPHA * (1./ temperature)) -
    PARAMETER_GAMMA * polygamma(0, PARAMETER_GAMMA * (1./ temperature))
    )

###################################
# PLOT: Average Energy vs. Theta
###################################
energy_vs_theta_figure = plt.figure(figsize = (10, 5.5))
energy_vs_theta_axis = energy_vs_theta_figure.add_subplot(1, 1, 1)

energy_vs_theta_axis.hlines(
    y = 0., xmin = 0.0 + 0.0 * np.min(inverse_temperature), xmax = np.max(inverse_temperature),
    colors = "black", linestyles = '--', alpha = 0.5,
    label = r"$\bar{E} = 0$")

energy_vs_theta_axis.plot(
    inverse_temperature, average_energy_vs_theta,
    color = "black", linestyle = '-', linewidth = 1.5,
    label = r"Average Energy Trend")

energy_vs_theta_axis.set_title(
    fr"$\bar{{E}}$ vs. Inverse Temperature for $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$",
    fontsize = 18
)

energy_vs_theta_axis.set_xlabel(xlabel = r"$\theta$ (Inverse Temperature)", fontsize = 17, labelpad = 4.,)
energy_vs_theta_axis.set_ylabel(ylabel = r"$\bar{E} \left( \theta \right)$", fontsize = 17, labelpad = 25.0, rotation = 0)
energy_vs_theta_axis.tick_params(labelsize = 17)
energy_vs_theta_axis.legend(fontsize = 18)

# (X): Save a version of the figure according to .eps format for Overleaf stuff:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.eps", format = "eps")
energy_vs_theta_figure.savefig(f"lv_avg_energy_inv_temp_{_version_number}.eps", format = "eps")

# (X): Save an immediately-visualizable figure with vector graphics:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.svg", format = "svg")
energy_vs_theta_figure.savefig(f"lv_avg_energy_inv_temp_{_version_number}.svg", format = "svg")

# (X): Closing figures:
plt.close(energy_vs_theta_figure)

###################################
# PLOT: Average Energy vs. Temperature
###################################
energy_vs_temp_figure = plt.figure(figsize = (10, 5.5))
energy_vs_emp_axis = energy_vs_temp_figure.add_subplot(1, 1, 1)

energy_vs_emp_axis.hlines(
    y = 0., xmin = 0.0 + 0.0 * np.min(temperature), xmax = np.max(temperature),
    colors = "black", linestyles = '--', alpha = 0.5,
    label = r"$\bar{E} = 0$")

energy_vs_emp_axis.plot(
    temperature, average_energy_vs_temperature,
    color = "black", linestyle = '-', linewidth = 1.5,
    label = r"Average Energy Trend")

energy_vs_emp_axis.set_title(
    fr"$\bar{{E}}$ vs. Temperature for $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$",
    fontsize = 18
)

energy_vs_emp_axis.set_xlabel(
    xlabel = r"$T$ (Temperature, ``$k_{\textrm{B}}$''$ = 1$)", fontsize = 17, labelpad = 4.,)
energy_vs_emp_axis.set_ylabel(
    ylabel = r"$\bar{E} \left( T \right)$", fontsize = 17, labelpad = 25.0, rotation = 0)

energy_vs_emp_axis.tick_params(labelsize = 17)
energy_vs_emp_axis.legend(fontsize = 18)

# (X): Save a version of the figure according to .eps format for Overleaf stuff:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.eps", format = "eps")
energy_vs_temp_figure.savefig(f"lv_avg_energy_temp_{_version_number}.eps", format = "eps")

# (X): Save an immediately-visualizable figure with vector graphics:
# plt.savefig(f"{PLOT_DIRECTORY}/{PLOT_TITLE}.svg", format = "svg")
energy_vs_temp_figure.savefig(f"lv_avg_energy_temp_{_version_number}.svg", format = "svg")

# (X): Closing figures:
plt.close(energy_vs_theta_figure)
