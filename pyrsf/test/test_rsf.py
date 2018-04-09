from __future__ import print_function

import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import odeint

from pyrsf.inversion import rsf_inversion

rsf = rsf_inversion()


class test_rsf(unittest.TestCase):
    """Test suite for testing the implementation of rate-and-state friction"""

    rtol = 1e-6     # Desired relative tolerance for assertion

    def integrate_wrapper(self, y, t, *args):
        """Wrapper function to integrate state evolution laws"""
        return self.f_int(y, *args)

    def print_name(self, name):
        """Print out the class and method names during testing"""
        print("__%s__.%s... " % (self.__class__.__name__, name), end="")

    @staticmethod
    def prepare_state_evolution():
        """Prepare standard data for state evolution tests"""
        t = np.linspace(0, 100, int(1e4))
        V = 1e-6
        Dc = 1e-4
        O = V / Dc
        rsf.params = {"inv_Dc": 1.0 / Dc}
        return t, V, O

    def test_ageing_law(self):
        """Test for the ageing law"""

        self.print_name(sys._getframe().f_code.co_name)
        t, V, O = self.prepare_state_evolution()
        theta0 = 0.0
        self.f_int = rsf._ageing_law
        theta = odeint(self.integrate_wrapper, theta0, t, args=(V,))[:,0]

        # Analytical solution for constant V, Dc
        theta_ana = (1.0/O)*(1 - np.exp(-O*t))

        assert_allclose(theta, theta_ana, self.rtol)
        print("OK")

    def test_slip_law(self):
        """Test for the slip law"""

        self.print_name(sys._getframe().f_code.co_name)
        t, V, O = self.prepare_state_evolution()
        theta0 = 1e3
        self.f_int = rsf._slip_law
        theta = odeint(self.integrate_wrapper, theta0, t, args=(V,))[:, 0]

        # Analytical solution for constant V, Dc
        theta_ana = (1.0/O) * np.exp(np.exp(-O*t + np.log(np.log(O*theta0))))

        assert_allclose(theta, theta_ana, self.rtol)
        print("OK")

    def test_forward(self):
        """Test for forward RSF modelling with ageing law"""

        self.print_name(sys._getframe().f_code.co_name)

        rtol = 1e-12

        params = {
            "a": 0.001,
            "b": 0.0015,
            "Dc": 1e-4,
            "k": 50.0,
            "mu0": 0.6,
            "V0": 1e-6,
            "V1": 1e-5,
            "eta": 0,
        }

        t = np.linspace(0, 100, int(1e3))

        rsf.set_params(params)
        rsf.integrator.set_integrator("vode")
        rsf.set_state_evolution("ageing")
        rsf.set_initial_values([params["V0"], params["Dc"] / params["V0"]])

        result = rsf.forward(t)
        truth = np.load("forward_data.npy")

        assert_allclose(result["mu"], truth[1], rtol)

        print("OK")

    def test_forward_opt(self):
        """Test for optimised forward RSF modelling"""

        self.print_name(sys._getframe().f_code.co_name)

        rtol = 1e-5

        params = {
            "a": 0.001,
            "b": 0.0015,
            "Dc": 1e-4,
            "k": 50.0,
            "mu0": 0.6,
            "V0": 1e-6,
            "V1": 1e-5,
            "eta": 0,
        }

        t = np.linspace(0, 100, int(1e3))

        rsf.set_params(params)
        rsf.set_initial_values([params["V0"], params["Dc"] / params["V0"]])

        result = rsf.forward_opt(t)
        truth = np.load("forward_data.npy")

        assert_allclose(result["mu"], truth[1], rtol)

        print("OK")

    def test_stickslip(self):
        """Test for radiation damped stick-slips"""

        self.print_name(sys._getframe().f_code.co_name)

        rtol = 1e-12

        # Define parameters for radiation damping. See e.g. Thomas et al. (2014)
        G = 30e9
        Vs = 3e3
        sigma = 1e7
        eta = 0.5 * G / (Vs * sigma)

        # Dictionary of input parameters
        params = {
            "a": 0.001,
            "b": 0.0015,
            "Dc": 3e-5,
            "k": 10.0,
            "mu0": 0.6,
            "V0": 1e-6,
            "V1": 1e-5,
            "eta": eta,
        }

        t = np.linspace(0, 500, int(1e4))

        rsf.set_params(params)
        rsf.integrator.set_integrator("vode")
        rsf.set_state_evolution("ageing")
        rsf.set_initial_values([params["V0"], params["Dc"] / params["V0"]])

        result = rsf.forward(t)
        truth = np.load("stickslip_data.npy")

        assert_allclose(result["mu"], truth[1], rtol)

        print("OK")
