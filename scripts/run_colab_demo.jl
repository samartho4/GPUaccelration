#!/usr/bin/env julia

println("🚀 GPUacceleration Colab Demo - Complete Pipeline")
println("="^50)

# 1. Environment setup
println("📦 Setting up environment...")
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# 2. Load dependencies
println("📚 Loading dependencies...")
using CSV, DataFrames, Statistics
using OrdinaryDiffEq
using DiffEqGPU
using CUDA
using StaticArrays
using BSON

# 3. Check data
println("📊 Checking data files...")
data_dir = joinpath(@__DIR__, "..", "data")
required_files = ["training_roadmap_correct.csv", "validation_roadmap_correct.csv", "test_roadmap_correct.csv"]

for file in required_files
    path = joinpath(data_dir, file)
    if isfile(path)
        df = CSV.read(path, DataFrame)
        println("✅ $file: $(size(df,1)) rows, $(size(df,2)) cols")
    else
        error("❌ Missing required file: $file")
    end
end

# 4. Run GPU ensemble demo
println("\n🎯 Running GPU Ensemble Demo...")
include(joinpath(@__DIR__, "gpu_ensemble_ude.jl"))

# 5. Verify results
println("\n📋 Verifying results...")
results_file = joinpath(@__DIR__, "..", "results", "gpu_ensemble_summary.bson")
if isfile(results_file)
    data = BSON.load(results_file)
    println("✅ Results saved:")
    println("   - dt: $(data[:dt])")
    println("   - t0: $(data[:t0])") 
    println("   - len: $(data[:len])")
    println("   - Ns: $(data[:Ns]) scenarios")
else
    error("❌ Results file not found")
end

println("\n🎉 GPUacceleration demo completed successfully!")
println("📁 Check results/gpu_ensemble_summary.bson for details")
