
import numpy as np

class rsf_framework:
    """
    This class contains the (non-optimised) formulations that fall under the
    rate-and-state friction (RSF) framework, including state evolution laws,
    and RSF formulations extended with cut-off velocities

    References:

    Dieterich, J. H. (2007), Applications of rate- and state-dependent friction
    to models of fault slip and earthquake occurrence, in: Treatise on Geophysics,
    edited by Kanamori, H., and G. Schubert, Earthquake Seismology, vol. 4,
    Elsevier, Amsterdam, 6054.

    Rice, J.R. (1993), Spatio-temporal Complexity of Slip on a Fault
    J. Geophys. Res., 98(B6), 9885-9907, doi:10.1029/93JB00191
    """

    law_book = ("ageing", "slip")   # Available state evolution laws

    # Initialise class, set default state evolution function and
    # evolution term description
    def __init__(self):
        """
        Initialise class, set default state evolution function and
        evolution term cut-off request state
        """
        self._state_evolution = self._ageing_law
        self.use_cutoff = False
        pass

    def _ageing_law(self, theta, V):
        """
        Computes state parameter evolution following the Dieterich ageing law
        Input: theta, velocity
        Returns: dtheta/dt
        """
        return 1.0 - V*theta*self.params["inv_Dc"]

    def _slip_law(self, theta, V):
        """
        Computes state parameter evolution following the Ruina slip law
        Input: theta, velocity
        Returns: dtheta/dt
        """
        x = V*theta*self.params["inv_Dc"]
        return -x*np.log(x)

    def calc_mu(self, V, theta):
        """
        Computes the coefficient of friction as a function of velocity, state,
        and RSF parameters. If the extended RSF formulation is used, the
        coefficient of friction is calculated from a modified evolution effect
        with cut-off velocity Vc (see Dieterich, 2007)
        Input: velocity, state parameter
        Returns: (vector of) the coefficient of friction
        """
        params = self.params
        b = params["b"].reshape(-1, 1)
        inv_Dc = params["inv_Dc"].reshape(-1, 1)

        direct_effect = params["a"]*np.log(V*params["inv_V0"])
        if self.use_cutoff:
            evolution_effect = b*np.log(theta*params["Vc"]*inv_Dc + 1.0)
        else:
            evolution_effect = b*np.log(theta*params["V0"]*inv_Dc)

        return params["mu0"] + direct_effect + evolution_effect.sum(axis=0)

    def _dmu_dV(self, V):
        """
        Computes the partial derivative of the coefficient of friction (see
        rsf_framework.calc_mu) to velocity
        Input: velocity
        Returns: partial derivative dmu/dV
        """
        return self.params["a"]/V

    def _dmu_dtheta(self, theta):
        """
        Computes the partial derivative of the coefficient of friction (see
        rsf_framework.calc_mu) to the state parameter
        Input: state parameter
        Returns: partial derivative dmu/dtheta
        """
        params = self.params
        if self.use_cutoff:
            return params["b"]*params["Vc"] / (theta*params["Vc"] + params["Dc"])
        else:
            return params["b"] / theta

    def _dmu_dt(self, V):
        """
        Computes the time-derivative of the coefficient of friction
        Input: velocity
        Returns: time-derivative dmu/dt
        """
        return self.params["k"]*(self.params["V1"] - V)

    def constitutive_relation(self, t, vars):
        """
        Main ODE describing the constitutive behaviour, called by the ODE
        solver. Note that the ODE is solved for velocity and state as:
        Y_dot = f(Y), with Y = [velocity, state1, state2, state_n...]
        Input: time, Y
        Returns: Y_dot
        """
        V = vars[0]
        theta = vars[1:]

        inv_Dc = self.params["inv_Dc"]

        dtheta = self._state_evolution(theta, V)
        dmu = self._dmu_dt(V)
        dmu_dV = self._dmu_dV(V)
        dmu_dtheta = self._dmu_dtheta(theta)

        # Rate of change of velocity, including the quasi-dynamic radiation
        # damping approximation (which is exact for a spring-block) to limit
        # the stress drop (see e.g. Rice, 1993)
        # dV/dt = (dmu/dt - dmu/dtheta * dtheta/dt) / (dmu/dv + eta)
        # eta = 0.5*G/(c*sigma)
        # with G: shear modulus, c: shear wave speed, sigma: normal stress
        # Higher values of eta result in lower slip velocities (more damping)
        # Default value = 0 (no radiation damping)
        dV = (dmu - (dmu_dtheta*dtheta).sum()) / (dmu_dV + self.params["eta"])

        # Pack output
        out = np.hstack([dV, dtheta])
        return out