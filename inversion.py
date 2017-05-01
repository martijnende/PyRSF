"""
Author: Martijn van den Ende
Modification date: 01 May 2017

References:

Dieterich, J. H. (2007), Applications of rate- and state-dependent 
friction to models of fault slip and earthquake occurrence, 
in: Treatise on Geophysics, edited by Kanamori, H., and G. Schubert, 
Earthquake Seismology, vol. 4, Elsevier, Amsterdam, 6054.

Thomas, M. Y., N. Lapusta, H. Noda, and J.-P. Avouac (2014), 
Quasi-dynamic versus fully dynamic simulations of earthquakes and 
aseismic slip with and without enhanced coseismic weaken-ing, 
J. Geophys. Res. Solid Earth, 119, 1986–2004, doi:10.1002/2013JB010615.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.optimize import leastsq
import re

"""
The rate-and-state friction framework
"""
class rsf_framework:

	# Initialise class, set default state evolution function and
	# evolution term description
	def __init__(self):
		self.state_evolution = self._ageing_law
		self.evolution_term = self._evolution_regular
		pass

	# Compute dtheta/dt following the Dieterich ageing law
	def _ageing_law(self, theta, V, Dc):
		return 1.0 - V*theta/Dc

	# Compute dtheta/dt following the Ruina slip law
	def _slip_law(self, theta, V, Dc):
		x = V*theta/Dc
		return -x*np.log(x)

	# Classical evolution effect
	def _evolution_regular(self, theta, params):

		buf = 0.0
		for i in xrange(len(theta)):
			buf += params["b"][i]*np.log(params["V0"]*theta[i]/params["Dc"][i])

		return buf

	# Modified evolution effect with cut-off velocity Vc (see Dieterich, 2007)
	def _evolution_cutoff(self, theta, params):

		buf = 0.0
		for i in xrange(len(theta)):
			buf += params["b"][i]*np.log(params["Vc"]*theta[i]/params["Dc"][i] + 1.0)

		return buf

	# Calculate velocity corresponding with mu and theta
	def calc_V(self, vars, params):

		# Unpack variables
		if type(vars) is dict:
			mu = vars["mu"]
			theta = vars["theta"]
		else:
			mu = vars[0]
			theta = vars[1:]

		# Calculate the contribution from the evolution term(s)
		evolution_buf = self.evolution_term(theta, params)

		# Compute velocity
		V = params["V0"]*np.exp( (mu - params["mu0"] - evolution_buf)/params["a"] )

		return V

	# Main ODE to solve
	def constitutive_relation(self, t, vars, params):

		# Compute velocity V
		V = self.calc_V(vars, params)

		# Compute state evolution dtheta/dt
		theta = vars[1:]
		Dc = params["Dc"]
		dtheta = self.state_evolution(theta, V, Dc)

		# Rate of change of friction, including the quasi-dynamic radiation
		# damping approximation to limit slip velocities (see e.g. Thomas et al., 2014)
		# dmu/dt = (k*(V_lp - V) - eta*dV/dtheta * dtheta/dt) / (1 + eta*dV/dmu)
		# eta = 0.5*G/(c*sigma), 
		# with G: shear modulus, c: shear wave speed, sigma: normal stress
		# Higher values of eta result in lower slip velocities (more damping)
		# Default value = 0 (no radiation damping)
		radiation_term1 = -params["eta"]*(params["b"]/params["a"])*(V/theta)*dtheta
		radiation_term2 = params["eta"]*V/params["a"]
		dmu = (params["k"]*(params["V1"] - V) - radiation_term1.sum())/(1 + radiation_term2)

		# Pack output
		out = np.hstack([dmu, dtheta])
		return out

"""
Integrator class
"""
class integrator_class(rsf_framework):

	def __init__(self):
		self.setup()
		pass

	# Initialise integrator
	def setup(self):
		self.integrator = ode(self.constitutive_relation)
		self.integrator.set_integrator("lsoda")

	# Main integrator function. Output vector:
	# 0: friction, 1+: state parameters
	def integrate(self, t, params):

		# Initial values of theta
		theta0 = params["Dc"]/params["V0"]

		# Initial values used for integration
		y0 = np.hstack([params["mu0"], theta0])

		# Allocate results
		result = np.zeros( (len(t), 1+len(params["b"])) )*np.nan
		result[0] = y0

		integrator = self.integrator
		# Set initial value
		integrator.set_initial_value(y0, t[0])
		# Set function parameters
		integrator.set_f_params(params)		

		i = 1
		# While integrator is alive, keep integrating up to t_max
		while integrator.successful() and integrator.t < np.max(t):
			# Perform one integration step
			integrator.integrate(t[i])
			# Store result
			result[i] = integrator.y[:]
			i += 1

		# Transpose result first
		return result.T


"""
Main API for this RSF inversion tool
"""
class rsf_inversion(integrator_class, rsf_framework):

	# Pattern to match: (b or Dc) + underscore (optional) + numbers
	regex_pattern = "^((b|Dc)+(_+)?)(?=.*[0-9]).+$"

	def __init__(self):
		rsf_framework.__init__(self)
		integrator_class.__init__(self)
		pass

	def set_params(self, params):

		self.params = params
		# Do sanity check

	def set_state_evolution(self, law):

		law_book = ("ageing", "slip")

		# Convert US spelling to proper spelling...
		if law is "aging": law = "ageing"

		# Check is state evolution law exists
		if law not in law_book:
			print "The requested state evolution law (%s) is not available" % (law)
			print "Available laws: %s" % ( ", ".join(law_book) )
			exit()

		# Set evolution law
		law_func = "_%s_law" % (law)
		self.state_evolution = getattr(self, law_func)

		pass

	# Call this function with a=True to use the modified RSF relations
	# with cut-off velocity (see Dieterich, 2007)
	def set_cutoff(self, a):

		if a is True:
			if not "Vc" in self.params:
				print "The params dictionary does not contain the cutoff velocity Vc"
				exit()

			self.evolution_term = self._evolution_cutoff
		else:
			self.evolution_term = self._evolution_regular

		pass

	# Construct forward RSF model. If params == False, the default
	# parameters are used that were set through set_params()
	def forward(self, t, params=False):
		
		# If params dict is not given, get one from self
		if params is False: params = self.params

		# Integrate forward model
		result = self.integrate(t, params)

		# Pack output dict
		out = {
			"mu": result[0],
			"theta": result[1:]
		}
		
		return out

	# Error function to minimise during the inversion
	def error(self, p):

		# Prepare parameter dict based in input parameters
		params = self.unpack_params(p)

		# Run forward model
		result = self.forward(self.data["t"], params)

		# Difference between data and forward model
		diff = self.data["mu"] - result["mu"]
		return diff

	# Estimate the uncertainty in the inverted parameters, based
	# on the Jacobian matrix provided by the leastsq function
	def estimate_uncertainty(self, popt, pcov):

		N_data = len(self.data["mu"])
		N_params = len(popt)

		# Check if the problem is well-posed
		if N_data > N_params and pcov is not None:
			# Calculate chi-squared
			s_sq = (self.error(popt)**2).sum() / (N_data - N_params)
			pcov = pcov*s_sq
		else:
			pcov = np.inf

		# Compute uncertainty in each inverted parameter
		buf = []
		for i in xrange(N_params):
			try:
				buf.append(np.abs(pcov[i,i])**0.5)
			except:
				buf.append(0.0)

		return np.array(buf)

	# Print the results of the inversion on-screen
	def print_result(self, popt, err):

		print "\nResult of inversion:\n"
		for i, key in enumerate(self.inversion_params):
			print "%s = %.3e +/- %.3e" % (key, popt[i], err[i])
		print "\n(errors reported as single standard deviation) \n"

		pass

	# Auxiliary function to prepare the vector containing the
	# to-be inverted parameters from the params dict
	def pack_params(self):

		x0 = []

		# For optimal flexibility, a regex pattern is used to filter
		# out any occurences of "b" and "Dc", with or without
		# trailing underscore and/or number. This way, this function
		# can accept "b", "Dc1", "b2", "b_3", etc.		

		# Loop over all parameters to invert for
		for key in self.inversion_params:

			# Check if any parameter matches pattern b1, Dc_2, etc.
			if re.match(self.regex_pattern, key) is not None:
				# Get the quantity (b or Dc)
				obj = re.match("b|Dc", key, flags=re.I).group()
				# Get the number (1, 2, 3, etc.)
				i = int(re.search("[0-9]+", key).group())
				# Get value from parameter dict
				val = self.params[obj][i-1]
			elif re.match("b|Dc", key, flags=re.I):
				# Case we have only one b and Dc
				val = self.params[key][0]
			else:
				# Any other parameter
				val = self.params[key]

			# Append parameter value to list
			x0.append(val)

		return x0

	# Auxiliary function to prepare the params dict from the
	# vector of to-be inverted parameters
	def unpack_params(self, p):

		# Construct parameter dict
		params = {
			"b": [], 
			"Dc": [], 
			"k": self.params["k"],
			"mu0": self.params["mu0"],
			"V0": self.params["V0"],
			"V1": self.params["V1"],
			"eta": self.params["eta"],
		}

		# Check if Vc is given
		if "Vc" in self.params:
			params["Vc"] = self.params["Vc"]

		# Loop over all inversion parameters
		for i, key in enumerate(self.inversion_params):
			# Check if any parameter matches pattern b1, Dc_2, etc.
			if re.match(self.regex_pattern, key) is not None:
				# Get the quantity (b or Dc)
				obj = re.match("b|Dc", key, flags=re.I).group()
				# Append value to list
				params[obj].append(p[i])
			elif re.match("b|Dc", key, flags=re.I):
				# Case we have only one b and Dc
				params[key] = np.array([p[i]])
			else:
				# Any other parameter
				params[key] = p[i]

		return params

	# Perform least-squares regression
	def inv_leastsq(self):

		# Prepare a vector with our initial guess
		x0 = self.pack_params()

		# OLS function
		popt, pcov, infodict, errmsg, success = leastsq(self.error, x0, full_output=True)

		# Return best-fit parameters and Jacobian matrix
		return popt, pcov

	# Main inversion function
	def inversion(self, data_dict, inversion_params, plot=True):

		# Store some input parameters
		self.inversion_params = inversion_params
		self.data = data_dict

		# Get best-fit parameters and Jacobian
		popt, pcov = self.inv_leastsq()

		# Calculate error of estimate
		uncertainty = self.estimate_uncertainty(popt, pcov)

		# Print results to screen
		self.print_result(popt, uncertainty)

		# Prepare output dictionary
		out = {}
		for i, key in enumerate(inversion_params):
			# Each parameter result is stored as a pair of
			# (value, uncertainty) in output dict
			out[key] = (popt[i], uncertainty[i])

		# Check if a plot is requested
		if plot is True:
			
			# Construct forward model with best-fit parameters
			params = self.unpack_params(popt)
			result = self.forward(self.data["t"], params)

			mu_model = result["mu"]

			plt.figure()
			plt.plot(self.data["t"], self.data["mu"], label="Data")
			plt.plot(self.data["t"], mu_model, label="Inversion")
			plt.legend(loc=1)
			plt.xlabel("time")
			plt.ylabel("friction [-]")
			plt.tight_layout()
			plt.show()

		return out
