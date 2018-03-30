from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from inversion import rsf_inversion

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

	t = np.linspace(0, 100, 1e4)

	# Set model parameters
	rsf.set_params(params)

	# Select ageing law
	rsf.set_state_evolution("ageing")

	# Perform forward model
	result = rsf.forward(t)

	# Calculate slip velocity based on model results
	V = rsf.calc_V(result, params)

	# Time-series of friction
	mu = result["mu"]
	
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


def multiple_state_parameters():

	# Dictionary of input parameters
	params = {
		"a": 0.001,
		"b": np.array([0.001, 0.0005]),
		"Dc": np.array([5e-5, 2e-4]),
		"k": 50.0,
		"mu0": 0.6,
		"V0": 1e-6,
		"V1": 1e-5,
		"eta": 0,
	}

	t = np.linspace(0, 100, 1e4)

	# Set model parameters
	rsf.set_params(params)

	# Select ageing law
	rsf.set_state_evolution("ageing")

	# Perform forward model
	result = rsf.forward(t)

	# Calculate slip velocity based on model results
	V = rsf.calc_V(result, params)

	# Time-series of friction
	mu = result["mu"]
	
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
		"V0": 1e-6,
		"V1": 1e-5,
		"eta": 0,
	}

	t = np.linspace(0, 100, 1e3)

	# Set model parameters
	rsf.set_params(params)

	# Select ageing law
	rsf.set_state_evolution("ageing")

	# Perform forward model
	result = rsf.forward(t)

	# Generate noisy signal
	noise = 1e-4*(np.random.rand(len(result["mu"])) - 0.5)
	mu_noisy = result["mu"] + noise

	# Change initial parameters to make the inversion
	# scheme sweat a little bit

	params["a"] = 0.0008
	params["b"] = np.array([0.0011])
	params["Dc"] = np.array([0.9e-4])
	rsf.set_params(params)

	# Construct our data dictionary
	data_dict = {"mu": mu_noisy, "t": t}

	# The parameters to invert for. Use b1, b2, Dc1, etc. when
	# inverting for more than one state variable
	inversion_params = ("a", "b", "Dc")

	# Perform the inversion. The results are given as a dictionary
	# in pairs of (value, uncertainty)
	inv_result = rsf.inversion(data_dict, inversion_params, plot=True)

def regular_stickslips():

	# Define parameters for radiation damping. See e.g. Thomas et al. (2014)
	G = 30e9
	Vs = 3e3
	sigma = 1e7
	eta = 0.5*G/(Vs*sigma)

	# Dictionary of input parameters
	params = {
		"a": 0.001,
		"b": np.array([0.0015,]),
		"Dc": np.array([3e-5]),
		"k": 10.0,
		"mu0": 0.6,
		"V0": 1e-6,
		"V1": 1e-5,
		"eta": eta,
	}

	kc = (params["b"][0]-params["a"])/params["Dc"][0]

	print("k/kc = %.3f" % (params["k"]/kc))

	t = np.linspace(0, 500, 1e4)

	# Set model parameters
	rsf.set_params(params)

	# Select ageing law
	rsf.set_state_evolution("ageing")

	# Perform forward model
	result = rsf.forward(t)

	# Calculate slip velocity based on model results
	V = rsf.calc_V(result, params)

	# Time-series of friction
	mu = result["mu"]
	
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



#simple_forward_model()
#multiple_state_parameters()
#simple_inversion()
regular_stickslips()
