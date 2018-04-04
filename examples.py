from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from pyrsf.inversion import rsf_inversion

# Initialise inversion API
rsf = rsf_inversion()

"""

Examples include:

1) Simple forward model of a velocity-step
2) Forward model with two state variables
3) Inversion
4) Simulating regular stick-slips with radiation damping

"""

# A simple velocity-step forward model
def simple_forward_model():

    # Dictionary of input parameters
    params = {
        "a": 0.001,
        "b": np.array([0.0015]),
        "Dc": np.array([1e-4]),
        "k": 50.0,
        "mu0": 0.6,
        "V0": 1e-6,
        "V1": 1e-5,
        "eta": 0,
    }

    t = np.linspace(0, 100, int(1e3))

    # Set model parameters
    rsf.set_params(params)

    # Select ageing law
    rsf.set_state_evolution("ageing")

    # Set initial values (V0, theta0), taken at steady-state
    rsf.set_initial_values(np.hstack([params["V0"], params["Dc"] / params["V0"]]))

    # Perform forward model
    result = rsf.forward(t)

    # Time-series of friction and velocity
    mu = result["mu"]
    V = result["V"]

    # Plot results
    plt.figure()
    plt.subplot(211)
    plt.plot(t, mu)
    plt.ylabel("friction [-]")
    plt.subplot(212)
    plt.axhline(params["V1"], ls="--", c="k")
    plt.plot(t, V)
    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("velocity [m/s]")
    plt.tight_layout()
    plt.show()


# Forward model of slide-hold-slide sequence
def forward_SHS():

    # Dictionary of input parameters
    params = {
        "a": 0.001,
        "b": np.array([0.0015]),
        "Dc": np.array([1e-4]),
        "k": 50.0,
        "mu0": 0.6,
        "V0": 1e-6,
        "V1": 1e-6,
        "eta": 0,
    }

    t = np.linspace(0, 100, int(1e1))

    # Set model parameters
    rsf.set_params(params)

    # Select ageing law
    rsf.set_state_evolution("ageing")

    # Set initial values (V0, theta0), taken at steady-state
    rsf.set_initial_values(np.hstack([params["V0"], params["Dc"] / params["V0"]]))

    # Perform forward model
    result = rsf.forward(t)

    # Time-series of friction and velocity
    mu = result["mu"]
    V = result["V"]

    plt.figure()
    plt.subplot(211)
    plt.plot(t, mu)
    plt.subplot(212)
    plt.plot(t, V)

    t_prev = t[-1]
    t = np.linspace(t_prev, 1000+t_prev, int(1e3))
    params["V1"] = 0.0
    rsf.set_params(params)
    rsf.set_initial_values(np.hstack([result["V"][-1], result["theta"][:,-1]]))

    # Perform forward model
    result = rsf.forward(t)

    # Time-series of friction and velocity
    mu = result["mu"]
    V = result["V"]

    plt.subplot(211)
    plt.plot(t, mu)
    plt.subplot(212)
    plt.plot(t, V)

    t_prev = t[-1]
    t = np.linspace(t_prev, 1000 + t_prev, int(1e4))
    params["V1"] = 1e-6
    rsf.set_params(params)
    rsf.set_initial_values(np.hstack([result["V"][-1], result["theta"][:, -1]]))

    result = rsf.forward(t)

    mu = result["mu"]
    V = result["V"]

    plt.subplot(211)
    plt.plot(t, mu)
    plt.subplot(212)
    plt.plot(t, V)

    # Plot results
    plt.subplot(211)
    plt.ylabel("friction [-]")
    plt.subplot(212)
    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("velocity [m/s]")
    plt.tight_layout()
    plt.show()

def multiple_state_parameters():

    # Dictionary of input parameters
    params = {
        "a": 0.001,
        "b": np.array([0.001, 0.001]),
        "Dc": np.array([5e-5, 2e-4]),
        "k": 50.0,
        "mu0": 0.6,
        "V0": 1e-6,
        "V1": 1e-5,
        "eta": 0,
    }

    t = np.linspace(0, 100, int(1e4))

    # Set model parameters
    rsf.set_params(params)

    # Select ageing law
    rsf.set_state_evolution("ageing")

    # Set initial values (V0, theta0), taken at steady-state
    rsf.set_initial_values(np.hstack([params["V0"], params["Dc"] / params["V0"]]))

    # Perform forward model
    result = rsf.forward(t)

    # Time-series of friction and velocity
    mu = result["mu"]
    V = result["V"]

    # Plot results
    plt.figure()
    plt.subplot(211)
    plt.plot(t, mu)
    plt.ylabel("friction [-]")
    plt.subplot(212)
    plt.axhline(params["V1"], ls="--", c="k")
    plt.plot(t, V)
    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("velocity [m/s]")
    plt.tight_layout()
    plt.show()


def simple_inversion():

    # Dictionary of input parameters
    params = {
        "a": 0.001,
        "b": np.array([0.0015]),
        "Dc": np.array([1e-4]),
        "k": 50.0,
        "mu0": 0.6,
        "V0": 3e-6,
        "V1": 1e-5,
        "eta": 0,
    }

    t = np.linspace(0, 100, int(1e3))

    # Set model parameters
    rsf.set_params(params)

    # Select ageing law
    rsf.set_state_evolution("ageing")

    # Set initial values (V0, theta0), taken at steady-state
    rsf.set_initial_values(np.hstack([params["V0"], params["Dc"] / params["V0"]]))

    # Perform forward model
    result = rsf.forward(t)

    # Generate noisy signal
    np.random.seed(0)
    noise = 1e-4*(np.random.rand(len(result["mu"])) - 0.5)
    mu_noisy = result["mu"] + noise

    # Change initial parameters to make the inversion
    # scheme sweat a little bit

    params["a"] = 0.0008
    params["b"] = np.array([0.0011, 0.001])
    params["Dc"] = np.array([0.9e-4, 1e-2])
    params["k"] = 40.0
    rsf.set_params(params)
    rsf.set_initial_values(np.hstack([params["V0"], params["Dc"] / params["V0"]]))

    # Construct our data dictionary
    data_dict = {"mu": mu_noisy, "t": t}

    # The parameters to invert for
    inversion_params = ("a", "b", "Dc", "k")

    # Perform the inversion. The results are given as a dictionary
    # in pairs of (value, uncertainty)

    inv_result = rsf.inversion(data_dict, inversion_params, plot=False, bayes=False)
    print(inv_result)

def regular_stickslips():

    # Define parameters for radiation damping. See e.g. Thomas et al. (2014)
    G = 30e9
    Vs = 3e3
    sigma = 1e7
    eta = 0.5*G/(Vs*sigma)

    # Dictionary of input parameters
    params = {
        "a": 0.001,
        "b": np.array([0.0015]),
        "Dc": np.array([3e-5]),
        "k": 10.0,
        "mu0": 0.6,
        "V0": 1e-6,
        "V1": 1e-5,
        "eta": eta,
    }

    kc = (params["b"][0]-params["a"])/params["Dc"][0]

    print("k/kc = %.3f" % (params["k"]/kc))

    t = np.linspace(0, 500, int(1e4))

    # Set model parameters
    rsf.set_params(params)

    # Select ageing law
    rsf.set_state_evolution("ageing")

    # Set initial values (V0, theta0), taken at steady-state
    rsf.set_initial_values(np.hstack([params["V0"], params["Dc"] / params["V0"]]))

    # Perform forward model
    result = rsf.forward(t)

    # Time-series of friction and velocity
    mu = result["mu"]
    V = result["V"]

    # Plot results
    plt.figure()
    plt.subplot(211)
    plt.plot(t, mu)
    plt.ylabel("friction [-]")
    plt.subplot(212)
    plt.axhline(params["V1"], ls="--", c="k")
    plt.plot(t, V)
    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("velocity [m/s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # simple_forward_model()
    # forward_SHS()
    # multiple_state_parameters()
    # simple_inversion()
    # regular_stickslips()

    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.run("simple_inversion()")
    ps = pstats.Stats(pr).sort_stats("tottime")
    ps.print_stats()