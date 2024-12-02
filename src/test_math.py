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























def test_integrate_linear():
    result = integrate_linear(2, 3, 0, 1)  # f(x) = 2x + 3 over [0, 1]
    assert abs(result - 4) < 1e-6, f"Expected 4 but got {result}"

