#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=False

cimport numpy as np
import numpy as np
from numpy cimport ndarray
from libc.math cimport log
ctypedef np.double_t DTYPE_t
ctypedef np.int_t DTYPE_i
import cython

cdef DTYPE_t ageing_law(DTYPE_t theta, DTYPE_t V, DTYPE_t inv_Dc):
    return 1.0 - V*theta*inv_Dc

cdef DTYPE_t slip_law(DTYPE_t theta, DTYPE_t V, DTYPE_t inv_Dc):
    cdef DTYPE_t O
    O = V*theta*inv_Dc
    return -O*log(O)

@cython.cdivision(True)
cdef DTYPE_t calc_dmu_dV(DTYPE_t V, DTYPE_t a):
    return a/V

@cython.cdivision(True)
cdef DTYPE_t calc_dmu_dtheta(DTYPE_t theta, DTYPE_t b):
    return b/theta

cdef DTYPE_t calc_dmu_dt(DTYPE_t V, DTYPE_t k, DTYPE_t V1):
    return k*(V1 - V)

cdef calc_mu(ndarray[DTYPE_t, ndim=1] params, ndarray[DTYPE_t, ndim=2] result):
    cdef DTYPE_t a, b, inv_Dc, mu0, V0, inv_V0
    cdef DTYPE_t direct_effect, evolution_effect
    cdef DTYPE_i i

    a = params[0]
    b = params[1]
    inv_Dc = params[2]
    mu0 = params[3]
    V0 = params[4]
    inv_V0 = params[5]

    for i in range(result.shape[0]):
        direct_effect = a*log(result[i,1]*inv_V0)
        evolution_effect = b*log(result[i,2]*V0*inv_Dc)
        result[i,0] = mu0 + direct_effect + evolution_effect

@cython.cdivision(True)
cdef constitutive_relation(ndarray[DTYPE_t, ndim=1, negative_indices=False, mode="c"] out, DTYPE_t V, DTYPE_t theta, DTYPE_t a, DTYPE_t b, DTYPE_t inv_Dc, DTYPE_t mu0, DTYPE_t V0, DTYPE_t inv_V0, DTYPE_t V1, DTYPE_t k, DTYPE_t eta):
    cdef DTYPE_t dmu, dmu_dV, dmu_dtheta, dtheta, dV

    dtheta = ageing_law(theta, V, inv_Dc)
    dmu = calc_dmu_dt(V, k, V1)
    dmu_dV = calc_dmu_dV(V, a)
    dmu_dtheta = calc_dmu_dtheta(theta, b)
    dV = (dmu - dmu_dtheta*dtheta) / (dmu_dV + eta)
    out[0] = dV
    out[1] = dtheta

@cython.cdivision(True)
cdef RK_solver(ndarray[DTYPE_t, ndim=1] t, ndarray[DTYPE_t, ndim=1] params, ndarray[DTYPE_t, ndim=1] y0, ndarray[DTYPE_t, ndim=2] result):
    cdef DTYPE_t a, b, inv_Dc, mu0, V0, inv_V0, V1, k, eta, dt
    cdef DTYPE_t V_prev, theta_prev
    cdef DTYPE_i i
    cdef ndarray[DTYPE_t, ndim=1, negative_indices=False, mode="c"] k1, k2, k3, k4

    a = params[0]
    b = params[1]
    inv_Dc = params[2]
    mu0 = params[3]
    V0 = params[4]
    inv_V0 = params[5]
    V1 = params[6]
    k = params[7]
    eta = params[8]

    k1 = ndarray(2)
    k2 = ndarray(2)
    k3 = ndarray(2)
    k4 = ndarray(2)

    result[0,1:] = y0
    dt = t[1] - t[0]
    half_dt = 0.5*dt
    sixth_dt = dt/6.0

    V_prev = y0[0]
    theta_prev = y0[1]

    for i in range(1, len(t)):

        constitutive_relation(k1, V_prev, theta_prev, a, b, inv_Dc, mu0, V0, inv_V0, V1, k, eta)
        constitutive_relation(k2, V_prev+half_dt*k1[0], theta_prev+half_dt*k1[1], a, b, inv_Dc, mu0, V0, inv_V0, V1, k, eta)
        constitutive_relation(k3, V_prev+half_dt*k2[0], theta_prev+half_dt*k2[1], a, b, inv_Dc, mu0, V0, inv_V0, V1, k, eta)
        constitutive_relation(k4, V_prev+dt*k3[0], theta_prev+dt*k3[0], a, b, inv_Dc, mu0, V0, inv_V0, V1, k, eta)

        V_prev = V_prev + sixth_dt*(k1[0] + 2*(k2[0] + k3[0]) + k4[0])
        theta_prev = theta_prev + sixth_dt*(k1[1] + 2*(k2[1] + k3[1]) + k4[1])

        result[i,1] = V_prev
        result[i,2] = theta_prev


def integrate(ndarray[DTYPE_t, ndim=1] t, ndarray[DTYPE_t, ndim=1] params, ndarray[DTYPE_t, ndim=1] y0):
    cdef ndarray[DTYPE_t, ndim=2] result = np.zeros((len(t), 3))
    RK_solver(t, params, y0, result)
    calc_mu(params, result)
    return result.T