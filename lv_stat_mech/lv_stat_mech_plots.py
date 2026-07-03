"""
A large(r) script that generates a bunch of plots that depict the LV statistical quantities as 
functions of temperature or inverse temperature.
"""

print("[INFO]: Script started!")

####################################################################################################
# Loading libraries:
####################################################################################################

# Native library | Path
from pathlib import Path 

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | ScIPy
from scipy.special import polygamma

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt
import matplotlib as mpl

####################################################################################################
# Versioning:
####################################################################################################

_VERSION_NUMBER = "3_2"

####################################################################################################
# Plotting customization :
# [NOTE]: this will *crash* your version of the code if you do not have TeX distribution installed!
####################################################################################################

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['savefig.dpi'] = 300

####################################################################################################
# Pathing:
####################################################################################################

# (X): Define the base path:
PLOT_DIRECTORY = Path('./')

####################################################################################################
# PLOT: Required numerical ingredients, including simulations parameters and temperature ranges
####################################################################################################

PARAMETER_ALPHA = 0.66666
PARAMETER_BETA = 1.33333
PARAMETER_GAMMA = 1.0
PARAMETER_DELTA = 1.0

temperature = np.linspace(start = 0.01, stop = 5.0, num = 10000)
inverse_temperature = 1.0 / temperature

####################################################################################################
# PLOT: Average Energy vs. Inverse Temperature
####################################################################################################

# (X): We derived this analytically:
def lv_average_energy(temperature, alpha, beta, gamma, delta):
    return (
        alpha +
        gamma +
        alpha * np.log(beta * temperature) +
        gamma * np.log(delta * temperature) -
        alpha * polygamma(0, alpha * temperature) -
        gamma * polygamma(0, gamma * temperature)
        )

def lv_specific_heat(temperature, alpha, gamma):
    return ((
        -alpha
        - gamma
        + alpha**2 * polygamma(1, alpha / temperature) / temperature
        + gamma**2 * polygamma(1, gamma / temperature) / temperature
    ) / temperature)

####################################################################################################
# PLOT: Average Energy vs. Inverse Temperature
####################################################################################################
energy_vs_sigma_figure = plt.figure(figsize = (10, 5.5))
energy_vs_sigma_axis = energy_vs_sigma_figure.add_subplot(1, 1, 1)

energy_vs_sigma_axis.hlines(
    y = 0., xmin = 0.0 + 0.0 * np.min(inverse_temperature), xmax = np.max(inverse_temperature),
    colors = "black", linestyles = '--', alpha = 0.5,
    label = r"$\langle E \rangle = 0$")

energy_vs_sigma_axis.plot(
    inverse_temperature, lv_average_energy(inverse_temperature, PARAMETER_ALPHA, PARAMETER_BETA, PARAMETER_GAMMA, PARAMETER_DELTA),
    color = "black", linestyle = '-', linewidth = 1.5,
    label = r"Average Energy Trend")

energy_vs_sigma_axis.set_title(
    fr"$\langle E \rangle$ vs. Inverse Temperature for $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$",
    fontsize = 18
)

energy_vs_sigma_axis.set_xlabel(xlabel = r"$\sigma$ (Inverse Temperature)", fontsize = 17, labelpad = 4.,)
energy_vs_sigma_axis.set_ylabel(ylabel = r"$\langle E \rangle \left( \sigma \right)$", fontsize = 17, labelpad = 25.0, rotation = 0)
energy_vs_sigma_axis.tick_params(labelsize = 17)
energy_vs_sigma_axis.legend(fontsize = 18)

for extension in ["eps", "svg", "png"]:
    energy_vs_sigma_figure.savefig(
        PLOT_DIRECTORY /
        "plots" /
        f"lv_avg_energy_inv_temp_{_VERSION_NUMBER}.{extension}", 
        format = extension)

# (X): Closing figures:
plt.close(energy_vs_sigma_figure)

####################################################################################################
# PLOT: Average Energy vs. Temperature
####################################################################################################
energy_vs_temp_figure = plt.figure(figsize = (10, 5.5))
energy_vs_emp_axis = energy_vs_temp_figure.add_subplot(1, 1, 1)

energy_vs_emp_axis.hlines(
    y = 0., xmin = 0.0 + 0.0 * np.min(temperature), xmax = np.max(temperature),
    colors = "black", linestyles = '--', alpha = 0.5,
    label = r"$\langle E \rangle = 0$")

energy_vs_emp_axis.plot(
    temperature, lv_average_energy(temperature, PARAMETER_ALPHA, PARAMETER_BETA, PARAMETER_GAMMA, PARAMETER_DELTA),
    color = "black", linestyle = '-', linewidth = 1.5,
    label = r"Average Energy Trend")

energy_vs_emp_axis.set_title(
    fr"$\langle E \rangle$ vs. Temperature for $\alpha = {PARAMETER_ALPHA}, \beta = {PARAMETER_BETA}, \gamma = {PARAMETER_GAMMA}, \delta = {PARAMETER_DELTA}$",
    fontsize = 18
)

energy_vs_emp_axis.set_xlabel(
    xlabel = r"$T$ (Temperature, ``$k_{\textrm{B}}$''$ = 1$)", fontsize = 17, labelpad = 4.,)
energy_vs_emp_axis.set_ylabel(
    ylabel = r"$\langle E \rangle \left( T \right)$", fontsize = 17, labelpad = 25.0, rotation = 0)

energy_vs_emp_axis.tick_params(labelsize = 17)
energy_vs_emp_axis.legend(fontsize = 18)

for extension in ["eps", "svg", "png"]:
    energy_vs_temp_figure.savefig(
        PLOT_DIRECTORY /
        "plots" /
        f"lv_avg_energy_temp_{_VERSION_NUMBER}.{extension}", 
        format = extension)

# (X): Closing figures:
plt.close(energy_vs_sigma_figure)

####################################################################################################
# PLOT: Heat Capacity vs. Temperature:
# [NOTE]: this plot is a little bit more involved than the others:
####################################################################################################

# (X): make the range of temperature values:
temperature = np.linspace(1e-6, 11.0, 1000)
# (X): make the range of parameter values:
alpha_values = np.linspace(1e-4, 5.0, 50)
gamma_values = np.linspace(1e-4, 5.0, 50)

# (X): fix the parameters:
FIXED_GAMMA = 2.0
FIXED_ALPHA = 2.0

# (X): do some matplotlib colormapping business:
colormap = plt.get_cmap("magma")
normalized_colors = mpl.colors.Normalize(vmin = alpha_values.min(), vmax = alpha_values.max())

# (X): Create a ScalarMappable for the colorbar.
scalar_mappable = plt.cm.ScalarMappable(cmap = colormap, norm = normalized_colors)
scalar_mappable.set_array([])

# (X): Now, we actually start plotting things:

fixed_gamma_heat_capacity_figure = plt.figure(figsize = (10, 6))
fixed_gamma_heat_capacity_axis = fixed_gamma_heat_capacity_figure.add_subplot(1, 1, 1)

for alpha in alpha_values:

    fixed_gamma_heat_capacity_axis.plot(
        temperature, lv_specific_heat(temperature, alpha, FIXED_GAMMA),
        color = colormap(normalized_colors(alpha)), linewidth = 1.0, alpha = 0.8,
    )

# (X): Need a colorbar to indicate how the parameters vary:
fixed_gamma_colorbar = fixed_gamma_heat_capacity_figure.colorbar(
    scalar_mappable, ax = fixed_gamma_heat_capacity_axis, pad = 0.02)

fixed_gamma_colorbar.set_label(r"Varying parameter $\alpha$", fontsize = 16.0)
fixed_gamma_colorbar.ax.tick_params(labelsize = 14.0)

fixed_gamma_heat_capacity_axis.hlines(
    y = 1.0, xmin = 0.0, xmax = 10.0,
    linestyles = "--", color = "gray", label = r"$C_V(T) = 1$")
fixed_gamma_heat_capacity_axis.hlines(
    y = 2.0, xmin = 0.0, xmax = 10.0,
    linestyles = "--", color = "red", label = r"$C_V(T) = 2$")
fixed_gamma_heat_capacity_axis.set_xlim(-0.5, 10.5)
fixed_gamma_heat_capacity_axis.set_ylim(0.0, 2.25)

fixed_gamma_heat_capacity_axis.set_title(
    fr"$C_V(T)$ Trend for Varying $\alpha$ with Fixed $\gamma = {FIXED_GAMMA}$", fontsize = 18.0)
fixed_gamma_heat_capacity_axis.set_xlabel(
    r"$T$", fontsize = 17.0)
fixed_gamma_heat_capacity_axis.set_ylabel(
    r"$C_V(T)$", fontsize = 17.0)
fixed_gamma_heat_capacity_axis.legend(fontsize = 14.0, loc = "lower right")

fixed_gamma_heat_capacity_figure.tight_layout()

for extension in ["eps", "svg", "png"]:
    fixed_gamma_heat_capacity_figure.savefig(
        PLOT_DIRECTORY /
        "plots" /
        f"lv_hc_vary_alpha_{_VERSION_NUMBER}.{extension}", format = extension)

plt.close(fixed_gamma_heat_capacity_figure)

fixed_alpha_heat_capacity_figure = plt.figure(figsize = (10, 6))
fixed_alpha_heat_capacity_axis = fixed_alpha_heat_capacity_figure.add_subplot(1, 1, 1)

for gamma_value in gamma_values:

    fixed_alpha_heat_capacity_axis.plot(
        temperature, lv_specific_heat(temperature, FIXED_ALPHA, gamma_value),
        color = colormap(normalized_colors(gamma_value)), linewidth = 1.0, alpha = 0.8,
    )

# (X): Need a colorbar to indicate how the parameters vary:
fixed_alpha_colorbar = fixed_alpha_heat_capacity_figure.colorbar(
    scalar_mappable, ax = fixed_alpha_heat_capacity_axis, pad = 0.02)

fixed_alpha_colorbar.set_label(r"Varying parameter $\gamma$", fontsize = 16.0)
fixed_alpha_colorbar.ax.tick_params(labelsize = 14.0)

fixed_alpha_heat_capacity_axis.hlines(
    y = 1.0, xmin = 0.0, xmax = 10.0,
    linestyles = "--", color = "gray", label = r"$C_V(T) = 1$")
fixed_alpha_heat_capacity_axis.hlines(
    y = 2.0, xmin = 0.0, xmax = 10.0,
    linestyles = "--", color = "red", label = r"$C_V(T) = 2$")
fixed_alpha_heat_capacity_axis.set_xlim(-0.5, 10.5)
fixed_alpha_heat_capacity_axis.set_ylim(0.0, 2.25)

fixed_alpha_heat_capacity_axis.set_title(
    fr"$C_V(T)$ Trend for Varying $\gamma$ with Fixed $\alpha = {FIXED_ALPHA}$", fontsize = 18.0)
fixed_alpha_heat_capacity_axis.set_xlabel(
    r"$T$", fontsize = 17.0)
fixed_alpha_heat_capacity_axis.set_ylabel(
    r"$C_V(T)$", fontsize = 17.0)
fixed_alpha_heat_capacity_axis.legend(fontsize = 14.0, loc = "lower right")

fixed_gamma_heat_capacity_figure.tight_layout()

for extension in ["eps", "svg", "png"]:
    fixed_alpha_heat_capacity_figure.savefig(
        PLOT_DIRECTORY /
        "plots" /
        f"lv_hc_vary_gamma_{_VERSION_NUMBER}.{extension}", format = extension)

plt.close(fixed_alpha_heat_capacity_figure)

print("[INFO]: Script finished running!")
