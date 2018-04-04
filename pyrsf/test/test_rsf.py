from __future__ import print_function

import sys

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import odeint

from pyrsf.friction_rsf import rsf_framework

rsf = rsf_framework()

class test_rsf:
    """Test suite for testing the implementation of rate-and-state friction"""

    rtol = 1e-6     # Desired relative tolerance for assertion

    def integrate_wrapper(self, y, t, *args):
        """Wrapper function to integrate state evolution laws"""
        return self.f_int(y, *args)

    def print_name(self, name):
        """Print out the class and method names during testing"""
        print("%s.%s... " % (self.__class__.__name__, name), end="")

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
