

# TODO:
"""
 [ ] Finalise docstrings
 [ ] Add unittests
 [ ] Re-organise classes into different files
 [ ] Test framework before implementation Bayesian inference
 [ ] Implement Bayesian inference
 [ ] Clean-up uncertainty/inversion methods
"""

# Importing python compatibility functions
from __future__ import print_function

from collections import defaultdict
from itertools import chain

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq, curve_fit

from pyrsf.friction_rsf import rsf_framework
from pyrsf.integrator import integrator_class


class rsf_inversion(integrator_class, rsf_framework):
    """
    Main API for the RSF inversion tool
    """

    # Pattern to match: (b or Dc) + underscore (optional) + numbers
    regex_pattern = "^((b|Dc)+(_+)?)(?=.*[0-9]).+$"

    def __init__(self):
        rsf_framework.__init__(self)
        integrator_class.__init__(self)
        pass

    def set_params(self, params):
        """
        Set the parameter dictionary and perform sanity checks. The following
        parameters are required: (a, b, Dc, mu0, V0, V1)
        Input: parameter dictionary
        Returns: None
        """
        # TODO: Add check for cut-off velocity, turn on cut-off if found
        # Also print warning that cut-off is enabled

        self.params = params

        required_params = ("a", "b", "Dc", "mu0", "V0", "V1")

        # Perform sanity check
        error = False
        for key in required_params:
            if key not in self.params:
                print("Parameter '%s' missing" % key)
                error = True

        if type(self.params["b"]) is not type(np.array([])):
            print("Parameter 'b' should be a NumPy array (e.g. np.array([0.01, 0.02]) )")
            error = True

        if type(self.params["Dc"]) is not type(np.array([])):
            print("Parameter 'Dc' should be a NumPy array (e.g. np.array([0.01, 0.02]) )")
            error = True

        # If we caught an error, exit program
        if error:
            exit()

        pass

    def set_state_evolution(self, law):
        """
        Define the state evolution law as used in the simulations
        Input: (aging, ageing, slip)
        Returns: None
        """

        law_book = self.law_book

        # Convert US spelling to proper spelling...
        if law is "aging":
            law = "ageing"

        # Check is state evolution law exists
        if law not in law_book:
            print("The requested state evolution law (%s) is not available" % law)
            print("Available laws: %s" % (", ".join(law_book)))
            exit()

        # Set evolution law
        law_func = "_%s_law" % law
        self._state_evolution = getattr(self, law_func)

        pass

    def forward(self, t):
        """
        Construct forward RSF model at specified time intervals
        Input: time vector
        Returns: dictionary of friction, velocity, and state parameter(s)
        """

        result = self.integrate(t)

        return result


    # Error function to minimise during the inversion
    def error(self, p):

        # Prepare parameter dict based in input parameters
        params = self.unpack_params(p)

        # Run forward model
        result = self.forward(self.data["t"])

        # Difference between data and forward model
        diff = self.data["mu"] - result["mu"]
        return diff

    def error_curvefit(self, t, *p):
        self.unpack_params(p)
        result = self.forward(t)
        return result["mu"]

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
        for i in range(N_params):
            try:
                buf.append(np.abs(pcov[i,i])**0.5)
            except:
                buf.append(0.0)

        return np.array(buf)

    # Print the results of the inversion on-screen
    def print_result(self, popt, err):

        return None

        # TODO: fix this for popt containing multiple b, Dc

        params = self.unpack_params(popt)

        print("\nResult of inversion:\n")
        for key, val in params.items():
            print("%s = %.3e +/- %.3e" % (key, popt[i], err[i]))
        print("\n(errors reported as single standard deviation) \n")

        pass

    def flatten(self, L):
        for item in L:
            try:
                yield from self.flatten(item)
            except TypeError:
                yield item

    # Auxiliary function to prepare the vector containing the
    # to-be inverted parameters from the params dict
    def pack_params(self):
        x = list(self.flatten([self.params[key] for key in self.inversion_params]))
        return x

    # Auxiliary function to prepare the params dict from the
    # vector of to-be inverted parameters
    def unpack_params(self, p):

        params = defaultdict(list)

        for key, val in zip(self.inversion_names, p):
            params[key].append(val)

        self.params.update(params)
        self.params["b"] = np.array(self.params["b"])
        self.params["Dc"] = np.array(self.params["Dc"])

        return params

    # Perform least-squares regression
    def inv_leastsq(self):

        # Prepare a vector with our initial guess
        x0 = self.pack_params()

        # OLS function
        popt, pcov, infodict, errmsg, success = leastsq(self.error, x0, full_output=True)

        # Return best-fit parameters and Jacobian matrix
        return popt, pcov

    # Perform non-linear least-squares
    def inv_curvefit(self):

        # TODO: set parameters to self.params during packing

        # Prepare a vector with our initial guess
        x0 = self.pack_params()

        # NL-LS inversion
        popt, pcov = curve_fit(self.error_curvefit, xdata=self.data["t"], ydata=self.data["mu"], p0=x0)

        # Return best-fit parameters and covariance matrix
        return popt, pcov

    def inv_bayes(self):
        np.random.seed(0)

        # popt, pcov = self.inv_curvefit()

        # When performing Bayesian inference, append
        # 'uncertainty' nuisance parameter
        self.inversion_params += "sigma",
        ndim = len(self.inversion_params)
        nwalkers = 20
        nburn = 0
        nsteps = 500

        starting_guess = np.random.random((nwalkers, ndim))*1e-4

        # Confine starting guess to reasonable ranges

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, args=[self.data["t"], self.data["mu"]])
        sampler.run_mcmc(starting_guess, nsteps)

        emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

        plt.subplot(121)
        for i, y in enumerate(sampler.chain):
            print(i)
            plt.plot(y)
        plt.subplot(122)
        plt.hist(emcee_trace[0], bins=20)
        plt.show()
        exit()


        pass

    # Main inversion function
    def inversion(self, data_dict, inversion_params, plot=True, bayes=False):

        # Make sure the set of inversion parameters is a tuple
        inversion_params = tuple(inversion_params)

        # Store input parameters
        self.inversion_params = inversion_params
        self.data = data_dict

        names = ()
        for key in self.inversion_params:
            if key == "b" or key == "Dc":
                for i in range(len(self.params[key])):
                    names += key,
            else:
                names += key,
        self.inversion_names = names

        if bayes is True:
            # Perform Bayesian inference
            popt, uncertainty = self.inv_bayes()
        else:
            # Get best-fit parameters and Jacobian
            # popt, pcov = self.inv_leastsq()
            # Note, pcov not correct!!!! See documentation Scipy
            popt, pcov = self.inv_curvefit()

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
            result = self.forward(self.data["t"])

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
