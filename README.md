# PyRSF
PyRSF is a rate-and-state friction (RSF) modelling package written in Python (2.7 and 3+ compatible). It features:

* Forward RSF modelling
* Inversion of the RSF parameters to data
* Bayesian inference / Markov Chain Monte Carlo (MCMC)
* A modified RSF formulation with cut-off velocity
* User-defined state evolution functions
* Stick-slip simulations with stable limit cycles for K < Kc, facilitated by radiation damping

There are various other RSF modelling tools out there, such as Chris Marone's [xlook](https://github.com/PennStateRockandSedimentMechanics/xlook) and John Leeman's [rsfmodel toolkit](https://github.com/jrleeman/rsfmodel). PyRSF is complementary to these.

PyRSF depends on various numerical and scientific libraries, including [SciPy](https://scipy.org/), [emcee](http://dfm.io/emcee/), and [Cython](http://cython.org/), as well as the plotting libraries [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/).

Note: in order to perform highly optimised (~50x speed-up) forward modelling, enter the `pyrsf` subdirectory and execute `python setup.py build_ext --inplace`. This will create an executable starting with `friction_rsf_opt` in the `/build` folder. Copy this executable to its parent directory, so that it can be accessed by the main API classes. The optimised solver is required for the Bayesian inference, which is computationally intensive.

### To-do

- [x] Make a to-do list
- [X] Bayesian inference and uncertainty estimation
- [ ] Create an iPython notebooks
- [ ] Inversion of the RSF framework to stick-slip data
- [ ] Re-introduce multiple state parameters
- [ ] See if the Cython workflow can be improved
