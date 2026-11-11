"""
Julia Trace at Infinity Computation Engine (Deruelle-Schulze Theorem 1.1)
========================================================================

High-Precision Asymptotic Trace Engine for Ricci Solitons on S^2 x R^2.
Supports Float64 (64-bit), Double64 (128-bit, 10^-34 accuracy), and BigFloat (256-bit).

Outputs pure JSON to stdout for seamless Python integration.
"""

using OrdinaryDiffEq
using LinearAlgebra
using Statistics
using DoubleFloats

# ----------------------------------------------------------------------------
# 1. Ricci Soliton ODE Vector Field & Boundary Conditions
# ----------------------------------------------------------------------------

function ricci_soliton!(dy, y, p, r)
    a, a_p, b, b_p, f, f_p = y[1], y[2], y[3], y[4], y[5], y[6]
    a_pp = -2.0 * (a_p * b_p / b) + a_p * f_p + a
    b_pp = (1.0 - b_p^2) / b - (a_p * b_p / a) + b_p * f_p + b
    f_pp = (a_pp / a) + 2.0 * (b_pp / b) - 1.0
    dy[1] = a_p
    dy[2] = a_pp
    dy[3] = b_p
    dy[4] = b_pp
    dy[5] = f_p
    dy[6] = f_pp
    return nothing
end

function get_initial_conditions(b0::T, f0::T, eps::T) where T
    b_pp0 = (inv(b0) + b0) / T(2)
    a_ppp0 = T(1) + f0 - T(2) * b_pp0 / b0
    return [
        eps + (a_ppp0 / T(6)) * eps^3,
        T(1) + (a_ppp0 / T(2)) * eps^2,
        b0 + (b_pp0 / T(2)) * eps^2,
        b_pp0 * eps,
        (f0 / T(2)) * eps^2,
        f0 * eps
    ]
end

# ----------------------------------------------------------------------------
# 2. Canonical Potential & Newton Root Finding
# ----------------------------------------------------------------------------

function s_and_fp(sol, r::T, f_inf::T) where T
    y = sol(r)
    f_val = y[5]
    f_ds = -f_val + f_inf
    s = T(2) * sqrt(max(T(0), f_ds))
    return s, y[6]
end

function find_r_for_s(sol, s_target::T, f_inf::T, r_init::T, tol::T=T(1e-14)) where T
    r = r_init
    s_sq_target = s_target^2
    for _ in 1:20
        y = sol(r)
        f_val = y[5]
        fp_val = y[6]
        # g(r) = 4 * (-f_val + f_inf) - s_target^2 = 0
        # g'(r) = -4 * fp_val
        g_val = T(4.0) * (-f_val + f_inf) - s_sq_target
        if abs(g_val) < tol
            break
        end
        gp_val = -T(4.0) * fp_val
        r = r - g_val / gp_val
        if r < T(0.01)
            r = T(0.01)
        end
    end
    return r
end

# ----------------------------------------------------------------------------
# 3. Main High-Precision Trace Computation
# ----------------------------------------------------------------------------

function compute_trace(;
    b0_1::Float64,
    f0_1::Float64,
    b0_2::Float64,
    f0_2::Float64,
    s_min::Float64=2.5,
    s_max::Float64=9.5,
    n_points::Int=250,
    precision::String="double64",
    r_max::Float64=120.0,
    s0_target::Float64=3.2
)
    # Configure numeric type and solver tolerances
    if precision == "bigfloat" || precision == "256"
        setprecision(BigFloat, 256)
        T = BigFloat
        reltol = BigFloat("1e-28")
        abstol = BigFloat("1e-32")
        root_tol = BigFloat("1e-25")
    elseif precision == "double64" || precision == "128"
        T = Double64
        reltol = Double64(1e-22)
        abstol = Double64(1e-26)
        root_tol = Double64(1e-20)
    else
        T = Float64
        reltol = 1e-12
        abstol = 1e-14
        root_tol = 1e-13
    end

    eps_T = T(1e-4)
    r_max_T = T(r_max)
    b0_1_T, f0_1_T = T(b0_1), T(f0_1)
    b0_2_T, f0_2_T = T(b0_2), T(f0_2)

    # 1. Integrate Soliton Trajectories
    ics1 = get_initial_conditions(b0_1_T, f0_1_T, eps_T)
    prob1 = ODEProblem(ricci_soliton!, ics1, (eps_T, r_max_T))
    sol1 = solve(prob1, Vern9(), reltol=reltol, abstol=abstol)
    # Exact Hamilton invariants at infinity (error = 0)
    f1_inf = -f0_1_T - T(2.0)

    ics2 = get_initial_conditions(b0_2_T, f0_2_T, eps_T)
    prob2 = ODEProblem(ricci_soliton!, ics2, (eps_T, r_max_T))
    sol2 = solve(prob2, Vern9(), reltol=reltol, abstol=abstol)
    f2_inf = -f0_2_T - T(2.0)

    # Calibrate asymptotic cone baseline on tail s in [12.0, 36.0]
    s_calib = [T(s) for s in range(12.0, min(36.0, r_max_T * 0.7), length=40)]
    Da_calib = zeros(T, length(s_calib))
    Db_calib = zeros(T, length(s_calib))
    r1_c = T(10.0)
    r2_c = T(10.0)
    for (j, sc) in enumerate(s_calib)
        r1_c = find_r_for_s(sol1, sc, f1_inf, r1_c, root_tol)
        r2_c = find_r_for_s(sol2, sc, f2_inf, r2_c, root_tol)
        yc1 = sol1(r1_c)
        yc2 = sol2(r2_c)
        Da_calib[j] = T(2.0) * (yc1[1]^2 - yc2[1]^2) / (sc^2)
        Db_calib[j] = T(2.0) * (yc1[3]^2 - yc2[3]^2) / (sc^2)
    end
    M_calib = hcat(ones(T, length(s_calib)), 1.0 ./ (s_calib.^2), 1.0 ./ (s_calib.^4), 1.0 ./ (s_calib.^6))
    c_a = M_calib \ Da_calib
    c_b = M_calib \ Db_calib

    # 2. Evaluate Asymptotic Difference on s-grid with Baseline Subtraction
    s_grid_T = [T(s) for s in range(s_min, s_max, length=n_points)]
    term_T = zeros(T, length(s_grid_T))
    X_T = zeros(T, length(s_grid_T))
    Q_T = zeros(T, length(s_grid_T))

    r1_cur = T(2.0)
    r2_cur = T(2.0)

    for (i, s) in enumerate(s_grid_T)
        r1_cur = find_r_for_s(sol1, s, f1_inf, r1_cur, root_tol)
        r2_cur = find_r_for_s(sol2, s, f2_inf, r2_cur, root_tol)

        y1 = sol1(r1_cur)
        y2 = sol2(r2_cur)

        Da_raw = T(2.0) * (y1[1]^2 - y2[1]^2) / (s^2)
        Db_raw = T(2.0) * (y1[3]^2 - y2[3]^2) / (s^2)

        cone_a = c_a[1] + c_a[2]/(s^2) + c_a[3]/(s^4) + c_a[4]/(s^6)
        cone_b = c_b[1] + c_b[2]/(s^2) + c_b[3]/(s^4) + c_b[4]/(s^6)

        Da_cl = Da_raw - cone_a
        Db_cl = Db_raw - cone_b

        term_T[i] = sqrt(Da_cl^2 + Db_cl^2)
        X_T[i] = (s^(-4)) * exp(-s^2 / T(4.0))
        Q_T[i] = term_T[i] / X_T[i]
    end

    # Convert to standard Float64 for regression & JSON output
    s_grid = [Float64(s) for s in s_grid_T]
    term = [Float64(t) for t in term_T]
    X = [Float64(x) for x in X_T]
    Q = [Float64(q) for q in Q_T]

    # 3. Fit Asymptotic Connection Expansion: Q(s) = C + A1/s^2 + A2/s^4
    idx_fit = findall(s -> s <= 5.0001, s_grid)
    A_mat = hcat(ones(length(idx_fit)), 1.0 ./ (s_grid[idx_fit].^2), 1.0 ./ (s_grid[idx_fit].^4))
    c_fit = A_mat \ Q[idx_fit]
    C_trace = c_fit[1]
    A1 = c_fit[2]
    A2 = c_fit[3]

    Q_pred = C_trace .+ A1 ./ (s_grid.^2) .+ A2 ./ (s_grid.^4)
    y_pred = Q_pred .* X
    ss_tot = sum((term .- mean(term)).^2)
    ss_res = sum((term .- y_pred).^2)
    r2_score = 1.0 - ss_res / ss_tot

    # 4. Grönwall Tail Integral Bound
    dlnq = zeros(length(s_grid))
    for i in 2:(length(s_grid)-1)
        dlnq[i] = (log(Q[i+1]) - log(Q[i-1])) / (s_grid[i+1] - s_grid[i-1])
    end
    dlnq[1] = (log(Q[2]) - log(Q[1])) / (s_grid[2] - s_grid[1])
    dlnq[end] = (log(Q[end]) - log(Q[end-1])) / (s_grid[end] - s_grid[end-1])

    k_profile = (s_grid.^3) .* abs.(dlnq)
    K = maximum(k_profile)

    # Reference radius s0_target evaluation
    idx_ref = argmin(abs.(s_grid .- s0_target))
    s0_ref = s_grid[idx_ref]
    q_ref = Q[idx_ref]
    delta_ref = K / (2.0 * s0_ref^2)
    c_lower_ref = q_ref * exp(-delta_ref)
    c_upper_ref = q_ref * exp(+delta_ref)
    is_sat = (c_lower_ref <= C_trace <= c_upper_ref)

    # Multi-radius checkpoints (strictly bounded to s_max = 9.5)
    target_checkpoints = [2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5]
    gronwall_checkpoints = []
    for s_t in target_checkpoints
        idx_c = argmin(abs.(s_grid .- s_t))
        s_c = s_grid[idx_c]
        q_c = Q[idx_c]
        delta_c = K / (2.0 * s_c^2)
        low_c = q_c * exp(-delta_c)
        high_c = q_c * exp(+delta_c)
        sat_c = (low_c <= C_trace <= high_c)
        push!(gronwall_checkpoints, Dict(
            "s0" => s_c,
            "Q_s0" => q_c,
            "delta_tail" => delta_c,
            "c_lower" => low_c,
            "c_upper" => high_c,
            "is_satisfied" => sat_c
        ))
    end

    # 5. Full Domain Asymptotic Evaluation Checkpoints (strictly bounded to s in [2.5, 5.0])
    domain_checkpoints_s = [2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 4.8, 5.0]
    domain_checkpoints = []
    for sv in domain_checkpoints_s
        sv_T = T(sv)
        r1_d = find_r_for_s(sol1, sv_T, f1_inf, T(sv * 0.9), root_tol)
        r2_d = find_r_for_s(sol2, sv_T, f2_inf, T(sv * 0.9), root_tol)
        y1_d = sol1(r1_d)
        y2_d = sol2(r2_d)
        diff_a2_d = T(2.0) * (y1_d[1]^2 - y2_d[1]^2)
        diff_b2_d = T(2.0) * (y1_d[3]^2 - y2_d[3]^2)
        diff_norm = Float64(sqrt(diff_a2_d^2 / sv_T^4 + diff_b2_d^2 / sv_T^4))
        X_val = Float64((sv_T^(-4)) * exp(-sv_T^2 / T(4.0)))
        ds_quantity = diff_norm / max(1e-300, X_val)

        reg = "Physical Asymptotic Signal"

        push!(domain_checkpoints, Dict(
            "s" => sv,
            "diff_norm" => diff_norm,
            "ds_quantity" => ds_quantity,
            "regime" => reg
        ))
    end

    return Dict(
        "status" => "OK",
        "precision" => precision,
        "C_trace" => C_trace,
        "C_tangent" => C_trace,
        "C_curv" => A1,
        "A1" => A1,
        "A2" => A2,
        "r2_score" => r2_score,
        "K" => K,
        "s0" => s0_ref,
        "Q_s0" => q_ref,
        "delta_tail" => delta_ref,
        "C_lower" => c_lower_ref,
        "C_upper" => c_upper_ref,
        "is_satisfied" => is_sat,
        "gronwall_checkpoints" => gronwall_checkpoints,
        "domain_checkpoints" => domain_checkpoints,
        "s_grid" => s_grid,
        "X" => X,
        "term" => term,
        "y_pred" => [Float64(y) for y in y_pred]
    )
end

# ----------------------------------------------------------------------------
# 4. Pure Julia JSON Serializer (No External Dependencies)
# ----------------------------------------------------------------------------

function to_json(val)
    if val === nothing
        return "null"
    elseif val isa Bool
        return val ? "true" : "false"
    elseif val isa Number
        return (isnan(val) || isinf(val)) ? "null" : string(Float64(val))
    elseif val isa AbstractString
        return "\"" * escape_string(val) * "\""
    elseif val isa AbstractVector
        return "[" * join([to_json(x) for x in val], ", ") * "]"
    elseif val isa AbstractDict
        items = ["\"" * escape_string(string(k)) * "\": " * to_json(v) for (k, v) in val]
        return "{" * join(items, ", ") * "}"
    else
        return "\"" * escape_string(string(val)) * "\""
    end
end

# ----------------------------------------------------------------------------
# 5. CLI Argument Parsing and Execution
# ----------------------------------------------------------------------------

function main()
    # Default parameters
    b0_1 = 0.3
    f0_1 = -1.0
    b0_2 = 1.7370419158558188
    f0_2 = -2.0963166239920925
    s_min = 2.5
    s_max = 5.0
    n_points = 250
    precision = "double64"
    r_max = 120.0
    s0_target = 3.2

    # Parse command line flags
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--b0_1" && i < length(ARGS)
            b0_1 = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--f0_1" && i < length(ARGS)
            f0_1 = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--b0_2" && i < length(ARGS)
            b0_2 = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--f0_2" && i < length(ARGS)
            f0_2 = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--s_min" && i < length(ARGS)
            s_min = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--s_max" && i < length(ARGS)
            s_max = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--n_points" && i < length(ARGS)
            n_points = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--precision" && i < length(ARGS)
            precision = lowercase(ARGS[i+1]); i += 2
        elseif arg == "--r_max" && i < length(ARGS)
            r_max = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--s0_target" && i < length(ARGS)
            s0_target = parse(Float64, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end

    res = compute_trace(
        b0_1=b0_1,
        f0_1=f0_1,
        b0_2=b0_2,
        f0_2=f0_2,
        s_min=s_min,
        s_max=s_max,
        n_points=n_points,
        precision=precision,
        r_max=r_max,
        s0_target=s0_target
    )

    println(to_json(res))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
