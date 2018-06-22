# Importing python compatibility functions
from __future__ import print_function

import numpy as np
from scipy.integrate import ode

from pyrsf.friction_rsf import rsf_framework


class integrator_class(rsf_framework):
    """
    Integrator class that handles the ODE solver. Inherits from rsf_framework
    """

    resize_step = int(1e3)
    mode = "dense"

    def __init__(self):
        rsf_framework.__init__(self)
        self.setup()
        pass

    def setup(self):
        """Initialises integrator to VODE (default)"""
        self.integrator = ode(self.constitutive_relation)
        self.integrator.set_integrator("vode")

    def set_initial_values(self, p0):
        self.initial_values = np.array(p0)
        pass

    def resize(self):
        """
        When using an adaptive time-step, we need to expand the size of the
        output vectors when more steps are taken than initially anticipated
        """
        extension = np.zeros((self.resize_step, 3))*np.nan
        self.result = np.concatenate((self.result, extension), axis=0)
        pass

    def solout(self, t, y):
        """
        Solout is called after each successful time step, stores the result,
        and resizes the result vector if necessary
        """
        i = self.integrator_step

        # Unpack results from ODE solver
        V, theta = y

        # Check if resize is needed for storing results
        if i >= self.result.shape[0]:
            self.resize()

        # Store results
        self.result[i, 0] = t
        self.result[i, 1] = V
        self.result[i, 2] = theta
        self.integrator_step += 1		# Increment number of time steps
        pass

    def integrate(self, t):
        """
        Main ODE solver
        Input: time vector at which output is desired
        """

        self.params["inv_Dc"] = 1.0 / self.params["Dc"]
        self.params["inv_V0"] = 1.0 / self.params["V0"]

        # Initial values used for integration [V0, theta0]
        y0 = self.initial_values

        # Allocate results
        self.result = np.zeros((self.resize_step, 3))*np.nan
        self.result[0] = np.hstack([0.0, y0])

        t_max = t[-1]

        integrator = self.integrator
        integrator.set_initial_value(y0, t[0])

        self.integrator_step = 1
        mode = self.solver_mode

        if mode == "dense":
            integrator.set_integrator("vode")
            # While integrator is alive, keep integrating up to t_max
            while integrator.successful() and integrator.t < t_max:
                # Perform one integration step
                result = integrator.integrate(t[self.integrator_step])
                # Call solout to store results
                self.solout(t[self.integrator_step], result)
        elif mode == "step":
            # Switch to Dormand-Price (Runge-Kutta) method
            integrator.set_integrator("dopri5", nsteps=1e6, rtol=1e-6, verbosity=-1)
            # Set function to call after each successful step
            integrator.set_solout(self.solout)
            # Perform integration up to t_max
            integrator.integrate(t_max)
        else:
            print("Mode '%s' not available" % mode)
            exit()

        # Transpose result first
        result = self.result.T
        t, V, theta = result
        inds = np.isfinite(t)
        t = t[inds]
        V = V[inds]
        theta = theta[inds]
        mu = self.calc_mu(V, theta)
        output = {
            "t": t,
            "mu": mu,
            "theta": theta,
            "V": V,
        }
        return output
