#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using OrdinaryDiffEq
using DiffEqGPU
using CUDA
using StaticArrays

println("🚀 GPU Ensemble UDE Demo (with CPU fallback)")

train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap_correct.csv")
@assert isfile(train_csv) "Missing training_roadmap_correct.csv"

df = CSV.read(train_csv, DataFrame)

# Assume uniform dt from data (first scenario)
first_s = df[df.scenario .== df.scenario[1], :]
T = collect(Float32.(first_s.time))
dt = T[2] - T[1]
t0 = T[1]
len = Int32(length(T))

# Group scenarios
scs = collect(keys(groupby(df, :scenario)))
Ns = length(scs)

# Pack signals contiguously
function pack(df)
    u = Float32[]; d = Float32[]; pg = Float32[]; pl = Float32[]; x10 = Float32[]; x20 = Float32[]
    for s in scs
        sdf = sort(df[df.scenario .== s, :], :time)
        append!(u, Float32.(sdf.u))
        append!(d, Float32.(sdf.d))
        append!(pg, Float32.(sdf.Pgen))
        append!(pl, Float32.(sdf.Pload))
        push!(x10, Float32(sdf.x1[1]))
        push!(x20, Float32(sdf.x2[1]))
    end
    return (u, d, pg, pl, x10, x20)
end

u_h, d_h, pg_h, pl_h, x10_h, x20_h = pack(df)
offsets = Int32.(cumsum([0; fill(Int(len), Ns-1)]))

# Device arrays
u_d = cu(u_h); d_d = cu(d_h); pg_d = cu(pg_h); pl_d = cu(pl_h)
x0s = [SVector{2,Float32}(x10_h[i], x20_h[i]) for i in 1:Ns]

# Physics params (placeholders; training not included in this demo)
ηin = 0.9f0; ηout = 0.9f0; α = 0.1f0; β = 1.0f0; γ = 0.02f0

struct GInputs{T}
  u::CuDeviceVector{T}
  d::CuDeviceVector{T}
  pg::CuDeviceVector{T}
  pl::CuDeviceVector{T}
  dt::T; t0::T; len::Int32; ηin::T; ηout::T; α::T; β::T; γ::T
end

@inline function idx_from_time(t, p::GInputs)
  i = Int32(floor((t - p.t0)/p.dt)) + 1
  return clamp(i, 1, p.len)
end

# GPU-safe RHS (single-scenario; Ensemble provides offset by index)
@inline function rhs_gpu!(du, x, p::Tuple{GInputs{Float32}, Int32}, t)
  base, off = p
  i = idx_from_time(t, base) + off
  u = base.u[i]; d = base.d[i]; Pgen = base.pg[i]; Pload = base.pl[i]
  x1, x2 = x
  du[1] = base.ηin*u*(u>0) - (1f0/base.ηout)*u*(u<0) - d
  nn = tanh(0.5f0*Pgen)  # tiny surrogate for fθ(Pgen)
  du[2] = -base.α*x2 + nn - base.β*Pload + base.γ*x1
  return nothing
end

base = GInputs(u_d, d_d, pg_d, pl_d, dt, t0, len, ηin, ηout, α, β, γ)

x0 = SVector{2,Float32}(0,0)
prob = ODEProblem(rhs_gpu!, x0, (t0, t0 + dt*(len-1)), (base, Int32(0)))

prob_func = function (prob,i,repeat)
  remake(prob; u0=x0s[i], p=(base, offsets[i]))
end

ensemble = EnsembleProblem(prob; prob_func)

traj = Ns
saveat = T

try
    CUDA.allowscalar(false)
    sol = solve(ensemble, Tsit5(); trajectories=traj, saveat=saveat, ensemblealg=EnsembleGPUArray())
    println("✅ GPU solve finished: $(length(sol)) scenarios")
catch e
    @warn "GPU failed, falling back to CPU" exception=(e, catch_backtrace())
    sol = solve(ensemble, Tsit5(); trajectories=traj, saveat=saveat)
    println("✅ CPU solve finished: $(length(sol)) scenarios")
end

# Save a tiny summary
using BSON
BSON.@save joinpath(@__DIR__, "..", "results", "gpu_ensemble_summary.bson") dt t0 len Ns
println("📁 Saved results/gpu_ensemble_summary.bson")
