import numpy as np
import scipy
print(scipy.__version__)
from scipy.integrate import quad
from scipy.linalg import solve

def integrate_linear(m, b, a, c):
    """
    Integrates the linear function f(x) = m*x + b over the range [a, c].

    Parameters:
        m (float): Slope of the linear function.
        b (float): Y-intercept of the linear function.
        a (float): Lower limit of integration.
        c (float): Upper limit of integration.

    Returns:
        float: The definite integral value.
    """ 
    def linear_function(x):
        return m * x + b

    result, _ = quad(linear_function, a, c)
    return result

def solve_system_of_equations(coefficients, constants):
    """
    Solves a system of two linear equations.

    Parameters:
        coefficients (list[list[float]]): Coefficients of the equations as a 2x2 list.
        constants (list[float]): Constants on the right-hand side as a list of two elements.

    Returns:
        dict: A dictionary with keys 'X' and 'Y' representing the solution.
    """
    solution = solve(coefficients, constants)
    return {'X': solution[0], 'Y': solution[1]}

def generate_normal_samples(mean, std, num_samples):

    return np.random.normal(mean, std, num_samples).tolist()



















def test_integrate_linear():
    result = integrate_linear(2, 3, 0, 1)  # f(x) = 2x + 3 over [0, 1]
    assert abs(result - 4) < 1e-6, f"Expected 4 but got {result}"

def test_solve_system_of_equations():
    coefficients = [[2, 1], [1, -1]]
    constants = [5, -1]
    result = solve_system_of_equations(coefficients, constants)
    assert abs(result['X'] - 4/3) < 1e-6, f"Expected X=4/3 but got {result['X']}"
    assert abs(result['Y'] - 7/3) < 1e-6, f"Expected Y=7/3 but got {result['Y']}"

def test_generate_normal_samples():
    mean, std, num_samples = 0, 1, 1000
    samples = generate_normal_samples(mean, std, num_samples)
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)

    assert abs(sample_mean - mean) < 0.1, f"Expected mean ~0 but got {sample_mean}"
    assert abs(sample_std - std) < 0.1, f"Expected std ~1 but got {sample_std}"
