# PyRSF
PyRSF is a light-weight rate-and-state friction (RSF) modelling package written in Python. It features:

* Forward RSF modelling with multiple state parameters
* Inversion of the RSF parameters
* A modified RSF formulation with cut-off velocity
* User-defined state evolution functions
* Stick-slip simulations with stable limit cycles for $$K < K_c$$, facilitated by radiation damping

There are various other RSF modelling tools out there, such as Chris Marone's [xlook](https://github.com/PennStateRockandSedimentMechanics/xlook) and John Leeman's [rsfmodel toolkit](https://github.com/jrleeman/rsfmodel). PyRSF is complementary to these.

### To-do

- [x] Make a to-do list
- [ ] Create an iPython notebooks
- [ ] Inversion of the RSF framework to stick-slip data
- [ ] Bayesian inference and uncertainty estimation
