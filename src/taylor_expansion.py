import sympy as sp

def compute_taylor_series(order=6):
    r = sp.Symbol('r')
    a0 = sp.Symbol('a0')
    f2_param = sp.Symbol('f2')
    
    a_coeffs = {0: a0}
    b_coeffs = {1: sp.S(1)}
    f_coeffs = {0: sp.S(0), 2: f2_param}  # Parameterize by f2
    
    for k in range(2, order+1, 2):
        if k != 0:
            a_coeffs[k] = sp.Symbol(f'a{k}')
        if k != 0 and k != 2:
            f_coeffs[k] = sp.Symbol(f'f{k}')
    for k in range(3, order+2, 2):
        b_coeffs[k] = sp.Symbol(f'b{k}')

    a_series = sum(c * r**k for k, c in a_coeffs.items())
    b_series = sum(c * r**k for k, c in b_coeffs.items())
    f_series = sum(c * r**k for k, c in f_coeffs.items())
    
    a_prime = sp.diff(a_series, r)
    a_double_prime = sp.diff(a_prime, r)
    b_prime = sp.diff(b_series, r)
    b_double_prime = sp.diff(b_prime, r)
    f_prime = sp.diff(f_series, r)
    f_double_prime = sp.diff(f_prime, r)
    
    eq1 = a_series * b_series * f_double_prime - b_series * a_double_prime - 2 * a_series * b_double_prime + a_series * b_series
    eq2 = b_series * a_double_prime + 2 * a_prime * b_prime - b_series * a_prime * f_prime - a_series * b_series
    eq3 = a_series * b_series * b_double_prime - a_series * (1 - b_prime**2) + b_series * a_prime * b_prime - a_series * b_series * b_prime * f_prime - a_series * b_series**2
    
    eq1_exp = sp.series(eq1, r, 0, order+2).removeO()
    eq2_exp = sp.series(eq2, r, 0, order+2).removeO()
    eq3_exp = sp.series(eq3, r, 0, order+2).removeO()
    
    # Collect all equations
    equations = []
    variables = []
    
    for current_order in range(2, order+1, 2):
        variables.append(a_coeffs[current_order])
        if current_order > 2:
            variables.append(f_coeffs[current_order])
        variables.append(b_coeffs[current_order+1])
        
        # eq2 coeff of r^(current_order-1)
        equations.append(eq2_exp.coeff(r, current_order-1))
        # eq1 coeff of r^(current_order-1)
        equations.append(eq1_exp.coeff(r, current_order-1))
        # eq3 coeff of r^(current_order)
        equations.append(eq3_exp.coeff(r, current_order))
        
    print(f"Solving {len(equations)} equations for {len(variables)} variables.")
    # Some equations might be redundant (like eq1 and eq3 at lowest order)
    # So we use sp.solve on the whole system
    solutions = sp.solve(equations, variables)
    sol_dict = {}
    if isinstance(solutions, dict):
        sol_dict = solutions
    elif isinstance(solutions, list) and len(solutions) > 0:
        sol_dict = dict(zip(variables, solutions[0]))
    return sol_dict, a_coeffs, b_coeffs, f_coeffs

if __name__ == "__main__":
    solutions, _, _, _ = compute_taylor_series(order=6)
    print("Solutions:")
    for k, v in solutions.items():
        print(f"{k} = {v}")
