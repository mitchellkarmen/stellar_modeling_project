import scipy
import numpy as np
from stellar_utils import *

def newton_root(fn, x0, max_iter=50):
    """
    Find the root of a function using Newton's method.

    Parameters:
    - fn (callable): The function for which to find the root.
    - x0 (float): Initial guess for the root.
    - max_iter (int, optional): Maximum number of iterations (default is 50).

    Returns:
    - root: The root of the function.
    """
    # root = scipy.optimize.minimize(fn, x0, method = 'SLSQP',
    root = scipy.optimize.newton(fn, x0, maxiter = max_iter)
    return root

def get_score(x1, x2):
    """
    Calculate the total fractional difference between two arrays.

    Parameters:
    - x1 (numpy.ndarray): First array.
    - x2 (numpy.ndarray): Second array.

    Returns:
    - score (float): Total fractional difference between x1 and x2.
    """

    norm_diff = (x1 - x2) / (x1 + x2)
    diff = x1 - x2

    print('diff is', norm_diff)
    if np.sum([np.abs(d/x1) for d in diff]) < 0.01:
        print('forcing convergence! diff is', diff)
        return 0
    return 10*norm_diff**3

def shootf(bounds, M_fit = 0.25):
    """
    Perform a shooting method to solve a differential equation.

    Parameters:
    - bounds (tuple): Tuple containing Pc, Tc, L_star, R_star.
    - M_fit (float, optional): Fractional mass where the shooting ends (default is 0.5).

    Returns:
    - score (float): The score representing the difference between inner and outer boundaries.
    """
    Pc, Tc, L_star, R_star = bounds
    M_r = 1e-8*M
    M_fit = M_fit*M
    inner_boundary = np.array(load1(Pc, Tc, M_r))
    outer_boundary = np.array(load2(L_star, R_star))
    bad_score = [10, 10, 10, 10]
    if np.isnan(np.min(inner_boundary)):
        print('found nan!')
        #inner_boundary = np.array([0, 0, 0, 0])
        return bad_score
    if np.isnan(np.min(outer_boundary)):
        print('found nan!')
        return bad_score


    n_steps = 1e7 # number of mass steps
    try:
        M_inner = np.linspace(M_r, M_fit, int(n_steps/2))
        M_outer = np.linspace(M, M_fit, int(n_steps/2))
        print('outer bounds', outer_boundary)
        print('inner_bounds', inner_boundary)
        inner_result = scipy.integrate.solve_ivp(derivs, [M_r, M_fit], inner_boundary, t_eval=M_inner)
        outer_result = scipy.integrate.solve_ivp(derivs, [M, M_fit], outer_boundary, t_eval=M_outer)
        score = get_score(inner_result.y.T[-1], outer_result.y.T[-1])
    except Exception as e:
        print(e)
        score = bad_score
    print('score', score)
    return score

