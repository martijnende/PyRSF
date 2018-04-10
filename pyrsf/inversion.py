

# TODO:
"""
 [ ] Finalise docstrings
 [ ] Add unittests
 [ ] Clean-up uncertainty/inversion methods
"""

# Importing python compatibility functions
from __future__ import print_function

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq, curve_fit

from pyrsf.friction_rsf import rsf_framework
from pyrsf.integrator import integrator_class
import pyrsf.friction_rsf_opt as rsf_opt
from pyrsf.bayes import bayes_framework

import warnings
warnings.simplefilter("once", UserWarning)


class rsf_inversion(integrator_class, rsf_framework, bayes_framework):
    """
    Main API for the RSF inversion tool
    """

    def __init__(self):
        integrator_class.__init__(self)
        rsf_framework.__init__(self)
        bayes_framework.__init__(self)
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

        # If we caught an error, exit program
        if error:
            exit()

        self.params["inv_Dc"] = 1.0/self.params["Dc"]
        self.params["inv_V0"] = 1.0 / self.params["V0"]

        pass

    def set_state_evolution(self, law):
        """
        Define the state evolution law as used in the simulations
        Input: (aging, ageing, slip)
        Returns: None
        """

        law = law.lower()
        law_book = self.law_book

        # Check is state evolution law exists
        for key, val in law_book.items():
            if law in val:
                law = key
                break
        else:
            print("The requested state evolution law (%s) is not available" % law)
            msg = ", ".join("%s => %s" % (key, val) for (key, val) in law_book.items())
            print("Available laws: %s" % msg)
            exit()

        # Set evolution law
        law_func = "_%s_law" % law
        self.params["state_evolution"] = law
        self._state_evolution = getattr(self, law_func)

        pass

    def forward(self, t, mode="dense"):
        """
        Construct forward RSF model at specified time intervals
        Input: time vector
        Returns: dictionary of friction, velocity, and state parameter
        """

        result = self.integrate(t, mode)

        return result

    def forward_opt(self, t):
        """
        Construct forward RSF model at specified time intervals, using an
        optimised (Cython) ODE solver
        Input: time vector
        Returns: dictionary of friction, velocity, and state parameter
        """

        params = self.params
        params_opt = np.array([
            params["a"], params["b"], params["inv_Dc"], params["mu0"],
            params["V0"], params["inv_V0"], params["V1"], params["k"],
            params["eta"]
        ])

        opt_result = rsf_opt.integrate(t, params_opt, self.initial_values)

        result = {"mu": opt_result[0], "V": opt_result[1], "theta": opt_result[2]}

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

    def error_curvefit_opt(self, t, *p):
        self.unpack_params(p)
        params = self.params
        params_opt = np.array([
            params["a"], params["b"], params["inv_Dc"], params["mu0"],
            params["V0"], params["inv_V0"], params["V1"], params["k"],
            params["eta"]
        ])
        result = rsf_opt.integrate(t, params_opt, self.initial_values)
        return result[0]

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

    # Print the results of the inversion on-screen (TODO)
    def print_result(self, popt, err):

        return None

        params = self.unpack_params(popt)

        print("\nResult of inversion:\n")
        for key, val in params.items():
            print("%s = %.3e +/- %.3e" % (key, popt[i], err[i]))
        print("\n(errors reported as single standard deviation) \n")

        pass

    # Auxiliary function to prepare the vector containing the
    # to-be inverted parameters from the params dict
    def pack_params(self):
        x = [self.params[key] for key in self.inversion_params]
        return x

    # Auxiliary function to prepare the params dict from the
    # vector of to-be inverted parameters
    def unpack_params(self, p):

        params = dict((key, val) for key, val in zip(self.inversion_params, p))
        params["inv_Dc"] = 1.0/params["Dc"]
        self.params.update(params)

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
    def inv_curvefit(self, opt=False):

        # Prepare a vector with our initial guess
        x0 = self.pack_params()

        if opt is True:
            print("Performing inversion in optimised mode")
            if self.params["state_evolution"] is not "ageing":
                print(
                    "Warning: the '%s' law is requested, but the optimised"
                    % (self.params["state_evolution"]),
                    "solver only supports the ageing (Dieterich) law\n",
                    "Will now continue using the ageing law..."
                )
            func = self.error_curvefit_opt
        else:
            func = self.error_curvefit

        # NL-LS inversion
        popt, pcov = curve_fit(func, xdata=self.data["t"], ydata=self.data["mu"], p0=x0)

        # Return best-fit parameters and covariance matrix
        return popt, pcov

    # Main inversion function
    def inversion(self, data_dict, inversion_params, opt=False, plot=True, bayes=False, load_pickle=False):

        # Make sure the set of inversion parameters is a tuple
        inversion_params = tuple(inversion_params)

        # Store input parameters
        self.inversion_params = inversion_params
        self.data = data_dict

        if bayes is True:
            # Perform Bayesian inference
            if load_pickle is not False:
                self.inversion_params += "sigma",
                self.unpickle_chain(load_pickle)
                popt, uncertainty = self.get_mcmc_stats().T
            else:
                popt0, _ = self.inv_curvefit(opt=True)
                self.inversion_params += "sigma",
                self.params["sigma"] = 1.0
                popt0 = np.hstack([popt0, self.params["sigma"]])
                popt, uncertainty = self.inv_bayes(popt0)
        else:
            # Get best-fit parameters and Jacobian
            # popt, pcov = self.inv_leastsq()
            # Note, pcov not correct!!!! See documentation Scipy
            popt, pcov = self.inv_curvefit(opt)

            # Calculate error of estimate
            uncertainty = self.estimate_uncertainty(popt, pcov)

        # Print results to screen
        self.print_result(popt, uncertainty)

        # Prepare output dictionary
        out = {}
        for i, key in enumerate(self.inversion_params):
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

    def rtol_check(self, t):
        """
        Auxiliary function to estimate the error in the optimised solver,
        compared to the non-optimised (VODE/LSODA) solver
        Input: time vector at which output is desired
        Returns: mean, max, and standard deviation of the relative difference
        """

        result = self.forward(t)
        result_opt = self.forward_opt(t)

        rdiff = np.abs(result["mu"]-result_opt["mu"])/result["mu"]

        return [rdiff.mean(), rdiff.max(), rdiff.std()]
