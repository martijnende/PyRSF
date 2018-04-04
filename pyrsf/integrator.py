# Importing python compatibility functions
from __future__ import print_function

from scipy.integrate import ode
import numpy as np

from pyrsf.friction_rsf import rsf_framework

class integrator_class(rsf_framework):
    """
    Integrator class that handles the ODE solver. Inherits from rsf_framework
    """

    def __init__(self):
        rsf_framework.__init__(self)
        self.setup()
        pass

    def setup(self):
        """Initialises integrator to vode"""
        self.integrator = ode(self.constitutive_relation)
        self.integrator.set_integrator("vode")

    def set_initial_values(self, p0):
        self.initial_values = p0
        pass

    def integrate(self, t):
        """
        Main ODE solver
        Input: time vector at which output is desired
        """

        self.params["inv_Dc"] = 1.0 / self.params["Dc"]
        self.params["inv_V0"] = 1.0 / self.params["V0"]

        # Initial values used for integration [mu0, theta0]
        y0 = self.initial_values

        # Allocate results
        result = np.zeros((len(t), 2))*np.nan
        result[0] = y0

        integrator = self.integrator
        # Set initial value
        integrator.set_initial_value(y0, t[0])

        i = 1
        # While integrator is alive, keep integrating up to t_max
        while integrator.successful() and integrator.t < np.max(t):
            # Perform one integration step
            integrator.integrate(t[i])
            # Store result
            result[i] = integrator.y[:]
            i += 1

        # Transpose result first
        result = result.T
        V = result[0]
        theta = result[1]
        mu = self.calc_mu(V, theta)
        output = {
            "mu": mu,
            "theta": theta,
            "V": V,
        }
        return output