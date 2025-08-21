#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics
using OrdinaryDiffEq, Optim
using Flux
using BSON
using Random

println("🧠 BNode Training (Both Equations Black Box)")
println("="^50)

# Load data
println("📊 Loading data...")
train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap_correct.csv")
val_csv = joinpath(@__DIR__, "..", "data", "validation_roadmap_correct.csv")

train_df = CSV.read(train_csv, DataFrame)
val_df = CSV.read(val_csv, DataFrame)

println("✅ Training: $(size(train_df,1)) rows, $(size(train_df,2)) cols")
println("✅ Validation: $(size(val_df,1)) rows, $(size(val_df,2)) cols")

# BNode RHS: Both equations as black boxes (Objective 1)
function make_bnode_rhs(nn_width::Int)
    # Neural network for fθ1(x1, x2, u, d) - replaces Eq1
    nn1 = Chain(
        Dense(4 => nn_width, tanh),
        Dense(nn_width => nn_width, tanh),
        Dense(nn_width => 1)
    )
    
    # Neural network for fθ2(x1, x2, Pgen, Pload) - replaces Eq2
    nn2 = Chain(
        Dense(4 => nn_width, tanh),
        Dense(nn_width => nn_width, tanh),
        Dense(nn_width => 1)
    )
    
    function bnode_rhs!(du, x, p, t)
        x1, x2 = x
        u = p.u_interp(t)
        d = p.d_interp(t)
        Pgen = p.pg_interp(t)
        Pload = p.pl_interp(t)
        
        # Eq1: Completely black box fθ1(x1, x2, u, d)
        du[1] = nn1([x1, x2, u, d])[1]
        
        # Eq2: Completely black box fθ2(x1, x2, Pgen, Pload)
        du[2] = nn2([x1, x2, Pgen, Pload])[1]
        
        return nothing
    end
    
    return bnode_rhs!, nn1, nn2
end

# Training function
function train_bnode_on_scenario(scenario_data, nn_width, lr, weight_decay)
    # Sort by time
    sorted_data = sort(scenario_data, :time)
    
    # Create interpolators
    u_interp = LinearInterpolation(sorted_data.time, sorted_data.u)
    d_interp = LinearInterpolation(sorted_data.time, sorted_data.d)
    pg_interp = LinearInterpolation(sorted_data.time, sorted_data.Pgen)
    pl_interp = LinearInterpolation(sorted_data.time, sorted_data.Pload)
    
    # Create BNode
    bnode_rhs!, nn1, nn2 = make_bnode_rhs(nn_width)
    
    # Initial conditions
    x0 = [sorted_data.x1[1], sorted_data.x2[1]]
    
    # Time span
    tspan = (sorted_data.time[1], sorted_data.time[end])
    
    # Parameters for ODE
    p = (u_interp=u_interp, d_interp=d_interp, pg_interp=pg_interp, pl_interp=pl_interp)
    
    # Create problem
    prob = ODEProblem(bnode_rhs!, x0, tspan, p)
    
    # Loss function
    function loss(p)
        # Split parameters for both networks
        n_params1 = length(Flux.params(nn1))
        p1 = p[1:n_params1]
        p2 = p[n_params1+1:end]
        
        # Update neural network parameters
        Flux.loadparams!(nn1, p1)
        Flux.loadparams!(nn2, p2)
        
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
    
    # Get initial parameters for both networks
    p1_init = Flux.params(nn1)
    p2_init = Flux.params(nn2)
    p_init = vcat(p1_init, p2_init)
    
    # Optimize
    result = optimize(loss, p_init, LBFGS(), inplace=false)
    
    return result.minimum, nn1, nn2
end

# Hyperparameter search
println("\n🔍 BNode hyperparameter tuning...")
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
best_nn1 = nothing
best_nn2 = nothing

for (i, config) in enumerate(configs)
    println("  Config $i: width=$(config.width), lr=$(config.lr), wd=$(config.weight_decay)")
    
    total_loss = 0.0
    nn1 = nothing
    nn2 = nothing
    
    # Train on first few scenarios for speed
    for scenario in scenarios[1:min(3, length(scenarios))]
        scenario_data = train_df[train_df.scenario .== scenario, :]
        loss, trained_nn1, trained_nn2 = train_bnode_on_scenario(scenario_data, config.width, config.lr, config.weight_decay)
        total_loss += loss
        nn1 = trained_nn1
        nn2 = trained_nn2
    end
    
    avg_loss = total_loss / min(3, length(scenarios))
    println("    Avg loss: $avg_loss")
    
    if avg_loss < best_loss
        best_loss = avg_loss
        best_config = config
        best_nn1 = nn1
        best_nn2 = nn2
    end
end

println("\n🏆 Best BNode config: width=$(best_config.width), lr=$(best_config.lr), wd=$(best_config.weight_decay)")
println("   Loss: $best_loss")

# Save best model
println("\n💾 Saving best BNode model...")
BSON.@save joinpath(@__DIR__, "..", "results", "best_bnode_model.bson") best_nn1 best_nn2 best_config best_loss

println("✅ BNode training completed!")
println("📁 Saved: results/best_bnode_model.bson")
