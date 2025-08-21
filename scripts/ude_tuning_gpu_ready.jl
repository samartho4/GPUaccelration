#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics
using OrdinaryDiffEq, Optim
using Flux
using BSON
using Random

println("🎯 UDE Tuning (GPU-Ready Implementation)")
println("="^50)

# Load data
println("📊 Loading data...")
train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap_correct.csv")
val_csv = joinpath(@__DIR__, "..", "data", "validation_roadmap_correct.csv")

train_df = CSV.read(train_csv, DataFrame)
val_df = CSV.read(val_csv, DataFrame)

println("✅ Training: $(size(train_df,1)) rows, $(size(train_df,2)) cols")
println("✅ Validation: $(size(val_df,1)) rows, $(size(val_df,2)) cols")

# Physics parameters (from SS)
ηin = 0.9f0; ηout = 0.9f0; α = 0.1f0; β = 1.0f0; γ = 0.02f0

# UDE RHS: Eq1 physics-only, Eq2 with neural term fθ(Pgen)
function make_ude_rhs(nn_width::Int)
    # Neural network for fθ(Pgen) in Eq2
    nn = Chain(
        Dense(1 => nn_width, tanh),
        Dense(nn_width => nn_width, tanh),
        Dense(nn_width => 1)
    )
    
    function ude_rhs!(du, x, p, t)
        x1, x2 = x
        u = p.u_interp(t)
        d = p.d_interp(t)
        Pgen = p.pg_interp(t)
        Pload = p.pl_interp(t)
        
        # Eq1: Physics-only (from SS)
        du[1] = ηin * u * (u > 0 ? 1.0 : 0.0) - (1.0/ηout) * u * (u < 0 ? 1.0 : 0.0) - d
        
        # Eq2: Replace β*Pgen with fθ(Pgen)
        nn_out = nn([Pgen])[1]
        du[2] = -α * x2 + nn_out - β * Pload + γ * x1
        
        return nothing
    end
    
    return ude_rhs!, nn
end

# Training function
function train_ude_on_scenario(scenario_data, nn_width, lr, weight_decay)
    # Sort by time
    sorted_data = sort(scenario_data, :time)
    
    # Create interpolators
    u_interp = LinearInterpolation(sorted_data.time, sorted_data.u)
    d_interp = LinearInterpolation(sorted_data.time, sorted_data.d)
    pg_interp = LinearInterpolation(sorted_data.time, sorted_data.Pgen)
    pl_interp = LinearInterpolation(sorted_data.time, sorted_data.Pload)
    
    # Create UDE
    ude_rhs!, nn = make_ude_rhs(nn_width)
    
    # Initial conditions
    x0 = [sorted_data.x1[1], sorted_data.x2[1]]
    
    # Time span
    tspan = (sorted_data.time[1], sorted_data.time[end])
    
    # Parameters for ODE
    p = (u_interp=u_interp, d_interp=d_interp, pg_interp=pg_interp, pl_interp=pl_interp)
    
    # Create problem
    prob = ODEProblem(ude_rhs!, x0, tspan, p)
    
    # Loss function
    function loss(p)
        # Update neural network parameters
        Flux.loadparams!(nn, p)
        
        # Solve ODE
        sol = solve(prob, Tsit5(), saveat=sorted_data.time)
        
        if sol.retcode != :Success
            return Inf
        end
        
        # MSE loss
        pred_x1 = [u[1] for u in sol.u]
        pred_x2 = [u[2] for u in sol.u]
        
        mse_x1 = mean((pred_x1 .- sorted_data.x1).^2)
        mse_x2 = mean((pred_x2 .- sorted_data.x2).^2)
        
        # Regularization
        reg = weight_decay * sum(p.^2)
        
        return mse_x1 + mse_x2 + reg
    end
    
    # Get initial parameters
    p_init = Flux.params(nn)
    
    # Optimize
    result = optimize(loss, p_init, LBFGS(), inplace=false)
    
    return result.minimum, nn
end

# Hyperparameter search
println("\n🔍 Hyperparameter tuning...")
configs = [
    (width=32, lr=0.01, weight_decay=1e-4),
    (width=64, lr=0.01, weight_decay=1e-4),
    (width=32, lr=0.001, weight_decay=1e-5),
    (width=64, lr=0.001, weight_decay=1e-5),
]

# Get scenarios
scenarios = unique(train_df.scenario)
println("📈 Training on $(length(scenarios)) scenarios")

best_loss = Inf
best_config = nothing
best_nn = nothing

for (i, config) in enumerate(configs)
    println("  Config $i: width=$(config.width), lr=$(config.lr), wd=$(config.weight_decay)")
    
    total_loss = 0.0
    nn = nothing
    
    # Train on first few scenarios for speed
    for scenario in scenarios[1:min(5, length(scenarios))]
        scenario_data = train_df[train_df.scenario .== scenario, :]
        loss, trained_nn = train_ude_on_scenario(scenario_data, config.width, config.lr, config.weight_decay)
        total_loss += loss
        nn = trained_nn
    end
    
    avg_loss = total_loss / min(5, length(scenarios))
    println("    Avg loss: $avg_loss")
    
    if avg_loss < best_loss
        best_loss = avg_loss
        best_config = config
        best_nn = nn
    end
end

println("\n🏆 Best config: width=$(best_config.width), lr=$(best_config.lr), wd=$(best_config.weight_decay)")
println("   Loss: $best_loss")

# Save best model
println("\n💾 Saving best UDE model...")
BSON.@save joinpath(@__DIR__, "..", "results", "best_ude_model.bson") best_nn best_config best_loss

println("✅ UDE tuning completed!")
println("📁 Saved: results/best_ude_model.bson")
