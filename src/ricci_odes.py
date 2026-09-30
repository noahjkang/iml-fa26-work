import numpy as np

def ricci_soliton(r, y):
    """
    Core vector field for the 4D cohomogeneity-one gradient expanding Ricci soliton.
    y = [a, a_prime, b, b_prime, f, f_prime]
    """
    a, a_prime, b, b_prime, f, f_prime = y
    
    # Floor values to prevent division by zero near the origin
    # (used primarily for floating point numerical integration)
    a = max(a, 1e-10)
    b = max(b, 1e-10)

    a_double_prime = -2 * (a_prime * b_prime) / b + a_prime * f_prime + a
    b_double_prime = (1 - b_prime**2) / b - (a_prime * b_prime) / a + b_prime * f_prime + b
    f_double_prime = a_double_prime / a + 2 * (b_double_prime / b) - 1
    
    return [a_prime, a_double_prime, b_prime, b_double_prime, f_prime, f_double_prime]
