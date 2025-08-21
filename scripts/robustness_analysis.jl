#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics
using OrdinaryDiffEq
using Flux
using BSON
using Random
using Distributions

println("🔬 Robustness Analysis for UDE/BNode System")
println("="^50)

# Load data and models
println("📊 Loading data and trained models...")
test_csv = joinpath(@__DIR__, "..", "data", "test_roadmap_correct.csv")
test_df = CSV.read(test_csv, DataFrame)

ude_file = joinpath(@__DIR__, "..", "results", "best_ude_model.bson")
bnode_file = joinpath(@__DIR__, "..", "results", "best_bnode_model.bson")

# Load models if available
best_ude_nn = nothing
best_bnode_nn1 = nothing
best_bnode_nn2 = nothing

if isfile(ude_file)
    ude_data = BSON.load(ude_file)
    best_ude_nn = ude_data[:best_nn]
    println("✅ UDE model loaded")
end

if isfile(bnode_file)
    bnode_data = BSON.load(bnode_file)
    best_bnode_nn1 = bnode_data[:best_nn1]
    best_bnode_nn2 = bnode_data[:best_nn2]
    println("✅ BNode model loaded")
end

# Physics parameters
ηin = 0.9f0; ηout = 0.9f0; α = 0.1f0; β = 1.0f0; γ = 0.02f0

# Define model RHS functions
function physics_only_rhs!(du, x, p, t)
    x1, x2 = x
    u = p.u_interp(t)
    d = p.d_interp(t)
    Pgen = p.pg_interp(t)
    Pload = p.pl_interp(t)
    
    du[1] = ηin * u * (u > 0 ? 1.0 : 0.0) - (1.0/ηout) * u * (u < 0 ? 1.0 : 0.0) - d
    du[2] = -α * x2 + β * Pgen - β * Pload + γ * x1
    return nothing
end

function ude_rhs!(du, x, p, t)
    x1, x2 = x
    u = p.u_interp(t)
    d = p.d_interp(t)
    Pgen = p.pg_interp(t)
    Pload = p.pl_interp(t)
    
    du[1] = ηin * u * (u > 0 ? 1.0 : 0.0) - (1.0/ηout) * u * (u < 0 ? 1.0 : 0.0) - d
    nn_out = best_ude_nn([Pgen])[1]
    du[2] = -α * x2 + nn_out - β * Pload + γ * x1
    return nothing
end

function bnode_rhs!(du, x, p, t)
    x1, x2 = x
    u = p.u_interp(t)
    d = p.d_interp(t)
    Pgen = p.pg_interp(t)
    Pload = p.pl_interp(t)
    
    du[1] = best_bnode_nn1([x1, x2, u, d])[1]
    du[2] = best_bnode_nn2([x1, x2, Pgen, Pload])[1]
    return nothing
end

# Evaluation function with noise
function evaluate_with_noise(rhs_func, scenario_data, noise_level, model_name)
    sorted_data = sort(scenario_data, :time)
    
    # Add noise to inputs
    noisy_data = copy(sorted_data)
    noise_dist = Normal(0, noise_level)
    
    noisy_data.u .+= rand(noise_dist, size(sorted_data, 1))
    noisy_data.d .+= rand(noise_dist, size(sorted_data, 1))
    noisy_data.Pgen .+= rand(noise_dist, size(sorted_data, 1))
    noisy_data.Pload .+= rand(noise_dist, size(sorted_data, 1))
    
    # Create interpolators with noisy data
    u_interp = LinearInterpolation(noisy_data.time, noisy_data.u)
    d_interp = LinearInterpolation(noisy_data.time, noisy_data.d)
    pg_interp = LinearInterpolation(noisy_data.time, noisy_data.Pgen)
    pl_interp = LinearInterpolation(noisy_data.time, noisy_data.Pload)
    
    # Initial conditions (no noise)
    x0 = [sorted_data.x1[1], sorted_data.x2[1]]
    tspan = (sorted_data.time[1], sorted_data.time[end])
    p = (u_interp=u_interp, d_interp=d_interp, pg_interp=pg_interp, pl_interp=pl_interp)
    
    # Solve ODE
    prob = ODEProblem(rhs_func, x0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=sorted_data.time)
    
    if sol.retcode != :Success
        return Dict("rmse_x1" => Inf, "rmse_x2" => Inf, "rmse_total" => Inf)
    end
    
    # Calculate metrics against clean data
    pred_x1 = [u[1] for u in sol.u]
    pred_x2 = [u[2] for u in sol.u]
    
    rmse_x1 = sqrt(mean((pred_x1 .- sorted_data.x1).^2))
    rmse_x2 = sqrt(mean((pred_x2 .- sorted_data.x2).^2))
    rmse_total = sqrt(mean([rmse_x1^2, rmse_x2^2]))
    
    return Dict(
        "rmse_x1" => rmse_x1,
        "rmse_x2" => rmse_x2,
        "rmse_total" => rmse_total
    )
end

# Parameter sensitivity analysis
function parameter_sensitivity_analysis(rhs_func, scenario_data, param_name, param_range)
    sorted_data = sort(scenario_data, :time)
    results = []
    
    for param_val in param_range
        # Modify global parameters
        global ηin, ηout, α, β, γ
        
        if param_name == "ηin"
            ηin = param_val
        elseif param_name == "ηout"
            ηout = param_val
        elseif param_name == "α"
            α = param_val
        elseif param_name == "β"
            β = param_val
        elseif param_name == "γ"
            γ = param_val
        end
        
        # Evaluate with modified parameter
        metrics = evaluate_with_noise(rhs_func, sorted_data, 0.0, "sensitivity")
        push!(results, (param_val, metrics["rmse_total"]))
    end
    
    return results
end

# Run robustness analysis
println("\n🎯 Running robustness analysis...")
scenarios = unique(test_df.scenario)
noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]

robustness_results = Dict()

# 1. Input noise robustness
println("\n📊 Input noise robustness analysis...")
for noise_level in noise_levels
    println("  Noise level: $noise_level")
    
    # Physics-only baseline
    physics_results = []
    for scenario in scenarios[1:min(5, length(scenarios))]
        scenario_data = test_df[test_df.scenario .== scenario, :]
        metrics = evaluate_with_noise(physics_only_rhs!, scenario_data, noise_level, "physics")
        push!(physics_results, metrics["rmse_total"])
    end
    
    # UDE robustness
    ude_results = []
    if best_ude_nn !== nothing
        for scenario in scenarios[1:min(5, length(scenarios))]
            scenario_data = test_df[test_df.scenario .== scenario, :]
            metrics = evaluate_with_noise(ude_rhs!, scenario_data, noise_level, "ude")
            push!(ude_results, metrics["rmse_total"])
        end
    end
    
    # BNode robustness
    bnode_results = []
    if best_bnode_nn1 !== nothing
        for scenario in scenarios[1:min(5, length(scenarios))]
            scenario_data = test_df[test_df.scenario .== scenario, :]
            metrics = evaluate_with_noise(bnode_rhs!, scenario_data, noise_level, "bnode")
            push!(bnode_results, metrics["rmse_total"])
        end
    end
    
    robustness_results["noise_$(noise_level)"] = Dict(
        "physics" => mean(physics_results),
        "ude" => best_ude_nn !== nothing ? mean(ude_results) : Inf,
        "bnode" => best_bnode_nn1 !== nothing ? mean(bnode_results) : Inf
    )
end

# 2. Parameter sensitivity
println("\n🔧 Parameter sensitivity analysis...")
param_ranges = Dict(
    "ηin" => [0.7, 0.8, 0.9, 0.95, 1.0],
    "ηout" => [0.7, 0.8, 0.9, 0.95, 1.0],
    "α" => [0.05, 0.1, 0.15, 0.2],
    "β" => [0.8, 1.0, 1.2, 1.5],
    "γ" => [0.01, 0.02, 0.03, 0.05]
)

sensitivity_results = Dict()

for (param_name, param_range) in param_ranges
    println("  Parameter: $param_name")
    
    # Test on first scenario
    scenario_data = test_df[test_df.scenario .== scenarios[1], :]
    
    # Physics-only sensitivity
    physics_sens = parameter_sensitivity_analysis(physics_only_rhs!, scenario_data, param_name, param_range)
    
    # UDE sensitivity
    ude_sens = best_ude_nn !== nothing ? 
        parameter_sensitivity_analysis(ude_rhs!, scenario_data, param_name, param_range) : []
    
    # BNode sensitivity
    bnode_sens = best_bnode_nn1 !== nothing ? 
        parameter_sensitivity_analysis(bnode_rhs!, scenario_data, param_name, param_range) : []
    
    sensitivity_results[param_name] = Dict(
        "physics" => physics_sens,
        "ude" => ude_sens,
        "bnode" => bnode_sens
    )
end

# Save results
println("\n💾 Saving robustness analysis results...")
BSON.@save joinpath(@__DIR__, "..", "results", "robustness_analysis.bson") robustness_results sensitivity_results

# Print summary
println("\n📋 Robustness Analysis Summary:")
println("="^40)

println("\n🎯 Input Noise Robustness:")
for noise_level in noise_levels
    results = robustness_results["noise_$(noise_level)"]
    println("  Noise $(noise_level*100)%:")
    println("    Physics: $(round(results["physics"], digits=4))")
    if results["ude"] != Inf
        println("    UDE: $(round(results["ude"], digits=4))")
    end
    if results["bnode"] != Inf
        println("    BNode: $(round(results["bnode"], digits=4))")
    end
end

println("\n🔧 Parameter Sensitivity:")
for (param_name, _) in param_ranges
    sens = sensitivity_results[param_name]
    println("  $param_name:")
    if !isempty(sens["physics"])
        physics_range = [r[2] for r in sens["physics"]]
        println("    Physics: $(round(maximum(physics_range) - minimum(physics_range), digits=4))")
    end
    if !isempty(sens["ude"])
        ude_range = [r[2] for r in sens["ude"]]
        println("    UDE: $(round(maximum(ude_range) - minimum(ude_range), digits=4))")
    end
    if !isempty(sens["bnode"])
        bnode_range = [r[2] for r in sens["bnode"]]
        println("    BNode: $(round(maximum(bnode_range) - minimum(bnode_range), digits=4))")
    end
end

println("\n✅ Robustness analysis completed!")
println("📁 Saved: results/robustness_analysis.bson")
