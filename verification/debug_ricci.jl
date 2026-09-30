using ReachabilityAnalysis
using Symbolics
using IntervalArithmetic
using TaylorSeries

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

    if typeof(b) <: Taylor1 && typeof(constant_term(b)) <: TaylorN
        if _isthinzero(constant_term(b))
            println("FATAL: constant_term(b) is exactly 0!")
            println("x vector is:")
            for i in 1:6
                println("x[$i] = ", constant_term(x[i]))
            end
        end
    end

    dx[1] = a_prime
    dx[2] = -2 * (a_prime * b_prime) / b + a_prime * f_prime + a
    dx[3] = b_prime
    dx[4] = (1 - b_prime^2) / b - (a_prime * b_prime) / a + b_prime * f_prime + b
    dx[5] = f_prime
    dx[6] = 2*(1 - b_prime^2)/(b^2) - 4*(a_prime * b_prime)/(a * b) + (a_prime/a + 2*b_prime/b)*f_prime + 2
end

a0 = 1.0           
f2 = -1.0          
eps = 1e-4         
Y0 = compute_initial_box(a0, f2, eps)
initial_set = Hyperrectangle(low=[inf(i) for i in Y0], high=[sup(i) for i in Y0])
prob = @ivp(x' = ricci_soliton!(x), dim: 6, x(0) ∈ initial_set)
sol = solve(prob, tspan=(0.0, 10.0), alg=TMJets(abstol=1e-10, orderT=4, orderQ=2))
println("Success!")
