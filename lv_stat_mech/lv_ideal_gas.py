"""
We use this script to make a plot of the phase space (in burner_q_values and p space)
of the "ideal gas" case.

Date initialized: 20260529
Last edited: 20260529

Notes:
    1. 20260529:
        Some values of these LV parameters will result in the plot showing nothing.
        That is likely because of some issues with the argument to the W-function that
        defines the bounding area. So, you may have to tinker a little bit with these
        numbers.
"""

##################################################
# Import 3rd-party libraries:
##################################################

import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt

##################################################
# File Information
##################################################

# Define a version number so we don't get confused:
_version_number = "1_1"

##################################################
# Plotting Configuration
##################################################

# (X): We tell rcParams to use LaTeX. [NOTE]: this will *crash* your 
# | version of the code if you do not have TeX distribution installed!
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True   
plt.rcParams['ytick.labelsize'] = 14
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
PARAMETER_BETA =  4.0/3.0 # predator hunting "power"
PARAMETER_GAMMA = 1.0 # predator exponential decrease
PARAMETER_DELTA = 1.0 # prey presence measurement:

LENGTH_OF_BOUNDING_BOX = 5.0
CONSTANT_ENERGY = 10.0

##################################################
# The bounds for p:
##################################################

bounding_p_function1 = (
    (
        CONSTANT_ENERGY - 
        PARAMETER_GAMMA * lambertw(
            -np.exp(-CONSTANT_ENERGY / PARAMETER_GAMMA) * PARAMETER_DELTA / PARAMETER_GAMMA, 0
            ).real) / PARAMETER_GAMMA
    )

bounding_p_function2 = (
    (
        CONSTANT_ENERGY - 
        PARAMETER_GAMMA * lambertw(
            -np.exp(-CONSTANT_ENERGY / PARAMETER_GAMMA) * PARAMETER_DELTA / PARAMETER_GAMMA, -1
            ).real) / PARAMETER_GAMMA
    )
##################################################
# Plotting in **ORIGINAL COORIDNATES**
##################################################

print("[INFO]: Now plotting stuff...")

figure_instance, axis_instance =  plt.subplots(nrows = 1, ncols = 1, figsize = (7.0, 7.0))

burner_q_values = np.linspace(0, LENGTH_OF_BOUNDING_BOX, 400)

pmax1 = np.full_like(burner_q_values, bounding_p_function1)
pmax2 = np.full_like(burner_q_values, bounding_p_function2)

axis_instance.fill_between(burner_q_values, pmax1, pmax2, color = 'skyblue', alpha = 0.5)

axis_instance.plot(burner_q_values, pmax1, color = 'black', linewidth = 2)
axis_instance.plot(burner_q_values, pmax2, color = 'black', linewidth = 2)

axis_instance.vlines([0, LENGTH_OF_BOUNDING_BOX], bounding_p_function1, bounding_p_function2, colors = 'black', linestyles = 'dashed')

axis_instance.axhline(0, color = 'black', linewidth = 1)
axis_instance.axvline(0, color = 'black', linewidth = 1)

axis_instance.set_xlabel(r'$q$', fontsize = 17.)
axis_instance.set_ylabel(r'$p$', fontsize = 17.)
axis_instance.set_title(
    fr"Phase Space Area with "
    fr"$\gamma = {PARAMETER_GAMMA:3g}, \delta = {PARAMETER_DELTA:3g}$", 
    fontsize = 18.)

axis_instance.grid(alpha = 0.6)

for image_format in ["eps", "svg"]:
    figure_instance.savefig(f"lv_ideal_gas_phase_space_v{_version_number}.{image_format}", format = image_format)

plt.close(figure_instance)

del figure_instance
del axis_instance

print("[INFO]: End of script reached!")
