#!/usr/bin/env python3
"""
Google Colab Setup Script for GPU Acceleration Project

This script sets up and runs the GPU acceleration demos on Google Colab.
It handles Julia installation, project setup, and demo execution.
"""

import os
import subprocess
import sys
import time

def run_command(command, description=""):
    """Run a shell command and print output"""
    print(f"🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print("Output:")
            print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🚀 GPU Acceleration Project - Google Colab Setup")
    print("=" * 50)
    
    # Step 1: Install Julia
    print("\n📦 Step 1: Installing Julia...")
    if not run_command("wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz", 
                      "Downloading Julia"):
        print("❌ Failed to download Julia")
        return
    
    if not run_command("tar -xzf julia-1.9.4-linux-x86_64.tar.gz", "Extracting Julia"):
        print("❌ Failed to extract Julia")
        return
    
    if not run_command("ln -s /content/julia-1.9.4/bin/julia /usr/local/bin/julia", "Creating symlink"):
        print("❌ Failed to create symlink")
        return
    
    if not run_command("julia --version", "Verifying Julia installation"):
        print("❌ Julia installation verification failed")
        return
    
    # Step 2: Clone and setup project
    print("\n📁 Step 2: Setting up project...")
    if not run_command("git clone https://github.com/samartho4/GPUaccelration.git", "Cloning repository"):
        print("❌ Failed to clone repository")
        return
    
    if not run_command("cd GPUaccelration", "Changing to project directory"):
        print("❌ Failed to change directory")
        return
    
    if not run_command("julia --project=. -e 'using Pkg; Pkg.instantiate()'", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        return
    
    # Step 3: Run basic demo
    print("\n🎯 Step 3: Running basic GPU acceleration demo...")
    if not run_command("julia --project=. scripts/gpu_acceleration_demo.jl", "Running basic demo"):
        print("❌ Basic demo failed")
        return
    
    # Step 4: Run advanced demo
    print("\n📊 Step 4: Running advanced performance analysis demo...")
    if not run_command("julia --project=. scripts/advanced_gpu_demo.jl", "Running advanced demo"):
        print("❌ Advanced demo failed")
        return
    
    # Step 5: Display results
    print("\n📈 Step 5: Displaying results...")
    try:
        import pandas as pd
        
        print("\n📊 Basic Demo Results:")
        try:
            basic_results = pd.read_csv('results/gpu_acceleration_summary.csv')
            print(basic_results.to_string(index=False))
        except:
            print("Basic results not found")
        
        print("\n📈 Advanced Demo Results:")
        try:
            advanced_results = pd.read_csv('results/advanced_gpu_benchmark.csv')
            print(advanced_results.to_string(index=False))
        except:
            print("Advanced results not found")
            
    except ImportError:
        print("Pandas not available, skipping results display")
    
    # Step 6: GPU information
    print("\n🖥️ Step 6: GPU Information:")
    run_command("nvidia-smi", "Checking GPU availability")
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n💡 Key Insights:")
    print("- The demos show GPU acceleration concepts on CPU")
    print("- Real GPU acceleration requires CUDA.jl installation")
    print("- Performance varies with data size and complexity")
    print("- Threaded CPU shows some parallelization benefits")
    
    print("\n🚀 Next Steps for Real GPU Acceleration:")
    print("1. Install CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
    print("2. Use @cuda macro for actual GPU kernels")
    print("3. Profile memory transfers and kernel execution")
    print("4. Optimize for GPU memory hierarchy")
    print("5. Use specialized GPU libraries (cuBLAS, cuDNN, etc.)")

if __name__ == "__main__":
    main()
