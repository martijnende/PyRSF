# PyRSF
PyRSF is a rate-and-state friction (RSF) modelling package written in Python (2.7 and 3+ compatible). It features:

* Forward RSF modelling
* Inversion of the RSF parameters to data
* Adaptive or dense time-stepping procedures for forward modelling and inversion/inference
* Bayesian inference / Markov Chain Monte Carlo (MCMC)
* A modified RSF formulation with cut-off velocity
* User-defined state evolution functions
* Stick-slip simulations with stable limit cycles for K < Kc, facilitated by radiation damping

There are various other RSF modelling tools out there, such as Chris Marone's [xlook](https://github.com/PennStateRockandSedimentMechanics/xlook) and John Leeman's [rsfmodel toolkit](https://github.com/jrleeman/rsfmodel). PyRSF is complementary to these.

PyRSF depends on various numerical and scientific libraries, including [SciPy](https://scipy.org/) and [emcee](http://dfm.io/emcee/), as well as the plotting libraries [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/).

### To-do

- [x] Make a to-do list
- [X] Bayesian inference and uncertainty estimation
- [X] Adaptive time-stepping
- [ ] iPython notebooks
- [ ] Inversion of the RSF framework to stick-slip data
- [ ] Re-introduction of multiple state parameters
