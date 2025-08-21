#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics
using OrdinaryDiffEq
using Flux
using BSON
using Random

println("📊 Comprehensive Model Evaluation")
println("="^50)

# Load data
println("📊 Loading test data...")
test_csv = joinpath(@__DIR__, "..", "data", "test_roadmap_correct.csv")
test_df = CSV.read(test_csv, DataFrame)
println("✅ Test: $(size(test_df,1)) rows, $(size(test_df,2)) cols")

# Load trained models
println("\n📥 Loading trained models...")
ude_file = joinpath(@__DIR__, "..", "results", "best_ude_model.bson")
bnode_file = joinpath(@__DIR__, "..", "results", "best_bnode_model.bson")

if isfile(ude_file)
    ude_data = BSON.load(ude_file)
    best_ude_nn = ude_data[:best_nn]
    best_ude_config = ude_data[:best_config]
    println("✅ UDE model loaded: width=$(best_ude_config.width)")
else
    println("⚠️  UDE model not found, skipping UDE evaluation")
    best_ude_nn = nothing
end

if isfile(bnode_file)
    bnode_data = BSON.load(bnode_file)
    best_bnode_nn1 = bnode_data[:best_nn1]
    best_bnode_nn2 = bnode_data[:best_nn2]
    best_bnode_config = bnode_data[:best_config]
    println("✅ BNode model loaded: width=$(best_bnode_config.width)")
else
    println("⚠️  BNode model not found, skipping BNode evaluation")
    best_bnode_nn1 = nothing
    best_bnode_nn2 = nothing
end

# Physics parameters
ηin = 0.9f0; ηout = 0.9f0; α = 0.1f0; β = 1.0f0; γ = 0.02f0

# Physics-only baseline (no neural terms)
function physics_only_rhs!(du, x, p, t)
    x1, x2 = x
    u = p.u_interp(t)
    d = p.d_interp(t)
    Pgen = p.pg_interp(t)
    Pload = p.pl_interp(t)
    
    # Eq1: Physics-only
    du[1] = ηin * u * (u > 0 ? 1.0 : 0.0) - (1.0/ηout) * u * (u < 0 ? 1.0 : 0.0) - d
    
    # Eq2: Physics-only (with β*Pgen)
    du[2] = -α * x2 + β * Pgen - β * Pload + γ * x1
    
    return nothing
end

# UDE RHS (Eq1 physics, Eq2 with neural term)
function ude_rhs!(du, x, p, t)
    x1, x2 = x
    u = p.u_interp(t)
    d = p.d_interp(t)
    Pgen = p.pg_interp(t)
    Pload = p.pl_interp(t)
    
    # Eq1: Physics-only
    du[1] = ηin * u * (u > 0 ? 1.0 : 0.0) - (1.0/ηout) * u * (u < 0 ? 1.0 : 0.0) - d
    
    # Eq2: Replace β*Pgen with fθ(Pgen)
    nn_out = best_ude_nn([Pgen])[1]
    du[2] = -α * x2 + nn_out - β * Pload + γ * x1
    
    return nothing
end

# BNode RHS (both equations black box)
function bnode_rhs!(du, x, p, t)
    x1, x2 = x
    u = p.u_interp(t)
    d = p.d_interp(t)
    Pgen = p.pg_interp(t)
    Pload = p.pl_interp(t)
    
    # Eq1: Completely black box fθ1(x1, x2, u, d)
    du[1] = best_bnode_nn1([x1, x2, u, d])[1]
    
    # Eq2: Completely black box fθ2(x1, x2, Pgen, Pload)
    du[2] = best_bnode_nn2([x1, x2, Pgen, Pload])[1]
    
    return nothing
end

# Evaluation function
function evaluate_model(rhs_func, scenario_data, model_name)
    sorted_data = sort(scenario_data, :time)
    
    # Create interpolators
    u_interp = LinearInterpolation(sorted_data.time, sorted_data.u)
    d_interp = LinearInterpolation(sorted_data.time, sorted_data.d)
    pg_interp = LinearInterpolation(sorted_data.time, sorted_data.Pgen)
    pl_interp = LinearInterpolation(sorted_data.time, sorted_data.Pload)
    
    # Initial conditions
    x0 = [sorted_data.x1[1], sorted_data.x2[1]]
    
    # Time span
    tspan = (sorted_data.time[1], sorted_data.time[end])
    
    # Parameters
    p = (u_interp=u_interp, d_interp=d_interp, pg_interp=pg_interp, pl_interp=pl_interp)
    
    # Create and solve problem
    prob = ODEProblem(rhs_func, x0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=sorted_data.time)
    
    if sol.retcode != :Success
        return Dict("rmse_x1" => Inf, "rmse_x2" => Inf, "rmse_total" => Inf)
    end
    
    # Calculate metrics
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

# Run evaluation
println("\n🎯 Running comprehensive evaluation...")
scenarios = unique(test_df.scenario)
println("📈 Evaluating on $(length(scenarios)) test scenarios")

results = Dict()

# Physics-only baseline
println("\n🔬 Physics-only baseline...")
physics_results = []
for scenario in scenarios
    scenario_data = test_df[test_df.scenario .== scenario, :]
    metrics = evaluate_model(physics_only_rhs!, scenario_data, "physics")
    push!(physics_results, metrics)
end

# Average physics results
avg_physics = Dict(
    "rmse_x1" => mean([r["rmse_x1"] for r in physics_results]),
    "rmse_x2" => mean([r["rmse_x2"] for r in physics_results]),
    "rmse_total" => mean([r["rmse_total"] for r in physics_results])
)
results["physics"] = avg_physics

println("   RMSE x1: $(round(avg_physics["rmse_x1"], digits=4))")
println("   RMSE x2: $(round(avg_physics["rmse_x2"], digits=4))")
println("   RMSE total: $(round(avg_physics["rmse_total"], digits=4))")

# UDE evaluation
if best_ude_nn !== nothing
    println("\n🎯 UDE evaluation...")
    ude_results = []
    for scenario in scenarios
        scenario_data = test_df[test_df.scenario .== scenario, :]
        metrics = evaluate_model(ude_rhs!, scenario_data, "ude")
        push!(ude_results, metrics)
    end
    
    avg_ude = Dict(
        "rmse_x1" => mean([r["rmse_x1"] for r in ude_results]),
        "rmse_x2" => mean([r["rmse_x2"] for r in ude_results]),
        "rmse_total" => mean([r["rmse_total"] for r in ude_results])
    )
    results["ude"] = avg_ude
    
    println("   RMSE x1: $(round(avg_ude["rmse_x1"], digits=4))")
    println("   RMSE x2: $(round(avg_ude["rmse_x2"], digits=4))")
    println("   RMSE total: $(round(avg_ude["rmse_total"], digits=4))")
end

# BNode evaluation
if best_bnode_nn1 !== nothing
    println("\n🧠 BNode evaluation...")
    bnode_results = []
    for scenario in scenarios
        scenario_data = test_df[test_df.scenario .== scenario, :]
        metrics = evaluate_model(bnode_rhs!, scenario_data, "bnode")
        push!(bnode_results, metrics)
    end
    
    avg_bnode = Dict(
        "rmse_x1" => mean([r["rmse_x1"] for r in bnode_results]),
        "rmse_x2" => mean([r["rmse_x2"] for r in bnode_results]),
        "rmse_total" => mean([r["rmse_total"] for r in bnode_results])
    )
    results["bnode"] = avg_bnode
    
    println("   RMSE x1: $(round(avg_bnode["rmse_x1"], digits=4))")
    println("   RMSE x2: $(round(avg_bnode["rmse_x2"], digits=4))")
    println("   RMSE total: $(round(avg_bnode["rmse_x2"], digits=4))")
end

# Save results
println("\n💾 Saving evaluation results...")
BSON.@save joinpath(@__DIR__, "..", "results", "comprehensive_evaluation.bson") results

# Print summary
println("\n📋 Evaluation Summary:")
println("="^30)
for (model, metrics) in results
    println("$model:")
    println("  RMSE x1: $(round(metrics["rmse_x1"], digits=4))")
    println("  RMSE x2: $(round(metrics["rmse_x2"], digits=4))")
    println("  RMSE total: $(round(metrics["rmse_total"], digits=4))")
end

println("\n✅ Comprehensive evaluation completed!")
println("📁 Saved: results/comprehensive_evaluation.bson")
