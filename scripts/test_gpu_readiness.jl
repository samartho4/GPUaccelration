#!/usr/bin/env julia

println("🔍 GPU Readiness Test")
println("="^30)

# Test 1: Basic imports
println("📚 Testing imports...")
try
    using CUDA
    println("✅ CUDA.jl loaded")
    
    if CUDA.functional()
        println("✅ CUDA functional")
        println("   Device: $(CUDA.name())")
        println("   Memory: $(CUDA.totalmem() ÷ 1024^3) GB")
    else
        println("⚠️  CUDA not functional (will use CPU fallback)")
    end
catch e
    println("❌ CUDA.jl failed: $e")
end

# Test 2: DiffEqGPU
println("\n🎯 Testing DiffEqGPU...")
try
    using DiffEqGPU
    println("✅ DiffEqGPU loaded")
catch e
    println("❌ DiffEqGPU failed: $e")
end

# Test 3: Data files
println("\n📊 Testing data files...")
data_dir = joinpath(@__DIR__, "..", "data")
required_files = ["training_roadmap_correct.csv"]

for file in required_files
    path = joinpath(data_dir, file)
    if isfile(path)
        println("✅ $file exists")
    else
        println("❌ $file missing")
    end
end

# Test 4: Simple GPU array
println("\n🧮 Testing GPU arrays...")
try
    using CUDA
    x = cu(rand(Float32, 100))
    y = 2.0f0 .* x
    println("✅ GPU array operations work")
catch e
    println("❌ GPU array test failed: $e")
end

println("\n🎉 GPU readiness test completed!")
