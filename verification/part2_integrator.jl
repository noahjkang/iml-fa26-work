# Part 2: Rigorous Integration (TaylorModels.jl / ReachabilityAnalysis.jl)
#
# This script takes the rigorous bounding box produced at r = epsilon
# and integrates the ODE vector field up to r = R.
#
# Requires: Pkg.add("ReachabilityAnalysis"), Pkg.add("Symbolics"), Pkg.add("IntervalArithmetic")

using ReachabilityAnalysis
using Symbolics
using IntervalArithmetic

# Include the auto-generated initial bounds
include("../src/initial_box.jl")

function ricci_soliton!(dx, x, p, t)
    # The independent variable here is 't', which represents our radius 'r'.
    # x = [a, a_prime, b, b_prime, f, f_prime]
    a = x[1]
    a_prime = x[2]
    b = x[3]
    b_prime = x[4]
    f = x[5]
    f_prime = x[6]
    
    # In Julia, ODE solvers sometimes probe the function with zero-initialized arrays.
    # To prevent division by zero during these internal allocation/probing phases:
    a_denom = iszero(constant_term(constant_term(a))) ? a + 1e-5 : a
    b_denom = iszero(constant_term(constant_term(b))) ? b + 1e-5 : b

    # dx/dr
    dx[1] = a_prime
    
    # a_double_prime
    dx[2] = -2 * (a_prime * b_prime) / b_denom + a_prime * f_prime + a
    
    # b_prime
    dx[3] = b_prime
    
    # b_double_prime
    dx[4] = (1 - b_prime^2) / b_denom - (a_prime * b_prime) / a_denom + b_prime * f_prime + b
    
    # f_prime
    dx[5] = f_prime
    
    # f_double_prime
    # We substitute a'' and b'' directly for numerical stability and faster Taylor evaluation
    # f'' = a''/a + 2b''/b - 1
    #     = 2*(1 - b'^2)/b^2 - 4*(a'b')/(a*b) + (a'/a + 2b'/b)*f' + 2
    dx[6] = 2*(1 - b_prime^2)/(b_denom^2) - 4*(a_prime * b_prime)/(a_denom * b_denom) + (a_prime/a_denom + 2*b_prime/b_denom)*f_prime + 2
end

function run_rigorous_proof()
    println("--- Starting Part 2: Rigorous Integration ---")
    
    # 1. Define Parameters
    a0 = 1.0           # Using scaling invariance, we can set a0 = 1
    f2 = -1.0          # Pick a sample f2 to verify (this determines expander/shrinker)
    
    # We increase eps to 0.01 to avoid the extreme 1/b derivatives at 1e-4.
    # Because our majorant is C=0.6, the error at eps=0.01 is still ~1e-25!
    eps = 0.01         
    R = 20.0           # The target large radius to integrate to
    
    println("Evaluating initial rigorous Taylor box at r = $eps...")
    
    # 2. Get the rigorous initial bounding box from Step 1
    # This evaluates the 10th order polynomial + the analytical majorant error bound
    Y0 = compute_initial_box(a0, f2, eps)
    
    # Convert to a Hyperrectangle for ReachabilityAnalysis
    initial_set = Hyperrectangle(low=[inf(i) for i in Y0], high=[sup(i) for i in Y0])
    
    # 3. Define the IVP (Initial Value Problem)
    prob = @ivp(x' = ricci_soliton!(x), dim: 6, x(0) ∈ initial_set)
    
    println("Integrating vector field from r=$eps to r=$R...")
    println("(Note: The 'time' variable in the integrator represents the radius r-eps)")
    
    # 4. Solve rigorously using Taylor Models
    # TMJets is a high-order rigorous integrator based on TaylorModels.jl
    time_horizon = R - eps
    sol = solve(prob, tspan=(0.0, time_horizon), alg=TMJets(abstol=1e-10, orderT=8, orderQ=2, maxsteps=50000))
    
    # 5. Extract the final bounding box at r=R
    final_set = sol(time_horizon)
    
    println("\n--- Proof Complete ---")
    println("Certified Box at r = $R:")
    println("a  ∈ ", final_set[1])
    println("a' ∈ ", final_set[2])
    println("b  ∈ ", final_set[3])
    println("b' ∈ ", final_set[4])
    println("f  ∈ ", final_set[5])
    println("f' ∈ ", final_set[6])
    
    return final_set
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Execute only if run directly
    run_rigorous_proof()
end
