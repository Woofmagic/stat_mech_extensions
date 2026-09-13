# `lv_lagrangian` Work

## Introduction:

Welcome to the GitHub repository that contains the related scripts that augmented the theoretical pursuit of finding a Lagrangian for the two-dimensional Lotka-Volterra system. In this repository, we feature all the Python scripts and Mathematica notebooks that we used. The Python scripts were primarily used for numerically solving ODEs and making plots, and the Mathematica notebooks were used for symbolic computations.

## List of Contents:

### Mathematica Notebooks:

1. `full_lagrangian_analysis.nb`: **This is a Mathematica notebook that contains all of the relevant computations** that we offer in the paper: (i) computing EOMs from the (a) "redundant" Lagrangian and (b) the reduced Lagrangian, (ii) computing EOMs from the rotated counterparts of the (a) "redundant" Lagrangian and (b) the reduced Lagrangian.

2. `examining_lv_lagrangian_differences.nb`: This is a notebook that just takes a look at some of the previous Lagrangians for the 2D LV system and seeing (i) if they encoded redundancies and, if true, then (ii) if they can be mapped onto the Lagrangian we generated.

3. `jlm-analysis-lv-lagrangian.nb`: This is a notebook that isn't actually relevant to the paper. We are just working on this on the side. Specifically, it is about the JLM (Jacobi Last Multiplier) approach to the same problem. Stay tuned!

4. `lv_nonhamiltonian.nb`: Another future work! Actually, in this notebook, we are just getting acquainted with Mathematica packages that do GR calculations because we are investigating geometrical ideas here.

### Python Scripts:

1. `basic_lv_time_evolution_v1.py`: A script that allows us to numerically solve the standard LV equations and make some plots depicting the solution.

2. `foliated_lv_kinematics_v1.py`: A script that makes the kinematic behavior of the "predator particle" according to its governing ODE. So, this makes a plot that shows its position vs. time, its velocity vs. time, and its acceleration vs. time.

3. `potential_plots_v1.py`: We make a (static) visualization of the potential well that the "predator particle" is in. This is just for visualization purposes.

4. `standard_lv_phase_plot_v1.py`: *This script isn't really finished*> All we are doing here is visualizing a single orbit of the LV system in its state space (*not phase space!*). We'll get back to it eventually...

### Plots:

There are a TON of plots. They all don't need to be there; they just represent the iterative process of generating the plots. So, take a look at them if you'd like, but know that you can generate all of these too according to using the Python scripts we offer.

