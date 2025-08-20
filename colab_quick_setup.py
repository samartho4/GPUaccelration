#!/usr/bin/env python3
"""
Quick Google Colab Setup for GPU Acceleration Project

This is a simplified version that can be run directly in Colab cells.
"""

import subprocess
import os

def run_cmd(cmd, desc=""):
    """Run command and print output"""
    print(f"🔄 {desc}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0

# Step 1: Install Julia
print("🚀 GPU Acceleration Project - Colab Setup")
print("=" * 40)

print("\n📦 Installing Julia...")
run_cmd("wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz", "Download Julia")
run_cmd("tar -xzf julia-1.9.4-linux-x86_64.tar.gz", "Extract Julia")
run_cmd("ln -s /content/julia-1.9.4/bin/julia /usr/local/bin/julia", "Create symlink")
run_cmd("julia --version", "Verify installation")

# Step 2: Setup project
print("\n📁 Setting up project...")
run_cmd("git clone https://github.com/samartho4/GPUaccelration.git", "Clone repo")
run_cmd("cd GPUaccelration", "Change directory")
run_cmd("julia --project=. -e 'using Pkg; Pkg.instantiate()'", "Install deps")

print("\n✅ Setup complete! Ready to run demos.")
