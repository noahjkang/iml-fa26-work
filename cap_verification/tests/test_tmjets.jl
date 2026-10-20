using ReachabilityAnalysis
using Symbolics
using IntervalArithmetic

function test_ode!(dx, x, p, t)
    dx[1] = 1.0 / x[1]
end

prob = @ivp(x' = test_ode!(x), dim: 1, x(0) ∈ Hyperrectangle(low=[1.0], high=[1.0]))
sol = solve(prob, tspan=(0.0, 1.0), alg=TMJets(abstol=1e-10, orderT=4, orderQ=2))
println("Success!")
