using ReachabilityAnalysis
using IntervalArithmetic
include("../../src/initial_box.jl")

# Load part2 model
function get_final_tm()
    a0 = 1.0; f2 = -1.0; eps = 0.01; R = 20.0
    Y0 = compute_initial_box(a0, f2, eps)
    initial_set = Hyperrectangle(low=[inf(i) for i in Y0], high=[sup(i) for i in Y0])
    
    function ricci!(dx, x, p, t)
        a = x[1]; ap = x[2]; b = x[3]; bp = x[4]; f = x[5]; fp = x[6]
        a_denom = iszero(constant_term(constant_term(a))) ? a + 1e-5 : a
        b_denom = iszero(constant_term(constant_term(b))) ? b + 1e-5 : b
        dx[1] = ap
        dx[2] = -2 * (ap * bp) / b_denom + ap * fp + a
        dx[3] = bp
        dx[4] = (1 - bp^2) / b_denom - (ap * bp) / a_denom + bp * fp + b
        dx[5] = fp
        dx[6] = (dx[2] / a_denom) + 2 * (dx[4] / b_denom) - 1
    end
    
    prob = @ivp(x' = ricci!(x), dim: 6, x(0) ∈ initial_set)
    time_horizon = R - eps
    sol = solve(prob, tspan=(0.0, time_horizon), alg=TMJets(abstol=1e-10, orderT=8, orderQ=2, maxsteps=50000))
    return sol(time_horizon)
end

tm = get_final_tm()
println("Final TaylorModel ReachSet: ", typeof(tm))
