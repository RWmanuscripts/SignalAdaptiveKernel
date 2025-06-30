# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# run this once: include("a.jl")

import PythonPlot as PLT
using Random, LinearAlgebra, Statistics

import SpatialTrees as ST
import PatchWorkKriging as PK
import Colors
include("helpers/visualization.jl")

PLT.close("all")
fig_num = 1

const T = Float64
D = 2

rng = Random.Xoshiro(0)
N = 500
min_t = T(-5.0)
max_t = T(5.0)
max_N_t = 5000

X = collect(randn(rng, T, D) for _ in 1:N)

# # Levels
levels = 2 # 2^(levels-1) leaf nodes. Must be larger than 1. This creates 2 regions.
s_tree = ST.PCTree(X, levels)
X_parts, X_parts_inds = ST.label_leaf_nodes(s_tree, X)

centroid = Statistics.mean(Statistics.mean(x) for x in X_parts)
max_dist = maximum(maximum(norm(xj - centroid) for xj in x) for x in X_parts) * 1.1
y_set = Vector{Vector{T}}(undef, 0)
t_set = Vector{Vector{T}}(undef, 0)
ST.get_partition_lines!(y_set, t_set, s_tree, min_t, max_t, max_N_t, centroid, max_dist)

boundary_pts = PK.generate_pseudo_obs(X, s_tree)
p_obs = boundary_pts[1:25:end] # throw away every 25 points along the projected points. Otherwise we'll have too many boundary points for this illustration.

# Oracle function
f_oracle = xx -> sinc(norm(xx))
y1 = f_oracle.(X_parts[1])
y2 = f_oracle.(X_parts[2])
y = vcat(y1, y2)

# # Patchwork kriging
# use the same square exponential bandwidth of 20 for both regions.
θs = [PK.SqExp(T(20)); PK.SqExp(T(20))]
ps = PK.GPParameters(X_parts, s_tree, θs)
model = PK.Model(p_obs, ps)

# ## Demonstrate continuity break.

function query_line(lb::T, ub::T, y_val::T, Nq::Integer, y, model, ps) where {T}
    xs = LinRange(lb, ub, Nq)

    mqs = Memory{T}(undef, Nq)
    vqs = Memory{T}(undef, Nq)
    xqs = Memory{Vector{T}}(undef, Nq)
    for i in eachindex(mqs, vqs, xs)
        xqs[i] = [xs[i]; y_val]
        mqs[i], vqs[i] = PK.query(xqs[i], y, model, ps)
    end
    return mqs, vqs, xqs
end

function query_line_rkhs(lb::T, ub::T, y_val::T, Nq::Integer, model) where {T}
    xs = LinRange(lb, ub, Nq)

    mqs = Memory{T}(undef, Nq)
    vqs = Memory{T}(undef, Nq)
    xqs = Memory{Vector{T}}(undef, Nq)
    for i in eachindex(mqs, vqs, xs)
        xqs[i] = [xs[i]; y_val]
        mqs[i], vqs[i] = PK.query_rkhs(xqs[i], model)
    end
    return mqs, vqs, xqs
end

Nq = 100
save_fig = true
#save_fig = false
fig_size = (7, 6)
dpi = 96 * 2

# ## Set the x2 coordinate to y_val == 2.
y_val = T(2)
lb, ub = -one(T), one(T)
mqs, vqs, xqs = query_line(lb, ub, y_val, Nq, y, model, ps)

fig_num, _ = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "Training inputs"; fig_size = fig_size, dpi = dpi)
PLT.plot(map(xx -> first(xx), p_obs), map(xx -> last(xx), p_obs), "x", label = "pseudo obs")
xqs1 = map(xx -> first(xx), xqs)
xqs2 = map(xx -> last(xx), xqs)
PLT.plot(xqs1, xqs2, "--", label = "query positions", color = "green")
PLT.legend()
PLT.xlabel("x₁")
PLT.ylabel("x₂")
if save_fig
    PLT.savefig(joinpath("figs", "manuscript", "pw_inputs_y2.png"), bbox_inches = "tight")
end

PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs)
PLT.plot(LinRange(lb, ub, Nq), mqs, "*", label = "query positions")
PLT.legend()
PLT.title("Patchwork kriging results")

y_rkhs = f_oracle.(X)
σ² = T(1.0e-8)
θ = θs[1]
model_rkhs = PK.RKHSModel(X, y_rkhs, σ², θ)
mqs_rkhs, vqs_rkhs, _ = query_line_rkhs(lb, ub, y_val, Nq, model_rkhs)

PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs)
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs, "*", label = "query positions")
PLT.legend()
PLT.title("Predictive posterior mean")

fqs = f_oracle.(xqs)
PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), fqs)
PLT.title("Oracle function")

PLT.figure(fig_num, figsize = fig_size, dpi = dpi)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs, "x", label = "Conventional GPR")
PLT.plot(LinRange(lb, ub, Nq), mqs, "o", label = "Patchwork kriging")
PLT.plot(LinRange(lb, ub, Nq), fqs, label = "Oracle")
PLT.legend()
PLT.xlabel("x₁")
PLT.ylabel("Function output")
PLT.title("Predictive posterior mean, x₂ = $(y_val)")
if save_fig
    PLT.savefig(joinpath("figs", "manuscript", "pw_results_y2.png"), bbox_inches = "tight")
end

# ## Set the x2 coordinate to y_val == 0
y_val = zero(T)
lb, ub = T(-0.5), T(0.5)
mqs, vqs, xqs = query_line(lb, ub, y_val, Nq, y, model, ps)

fig_num, _ = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "Training inputs"; fig_size = fig_size, dpi = dpi)
PLT.plot(map(xx -> first(xx), p_obs), map(xx -> last(xx), p_obs), "x", label = "pseudo obs")
xqs1 = map(xx -> first(xx), xqs)
xqs2 = map(xx -> last(xx), xqs)
PLT.plot(xqs1, xqs2, "--", label = "query positions", color = "green")
PLT.legend()
PLT.xlabel("x₁")
PLT.ylabel("x₂")
if save_fig
    PLT.savefig(joinpath("figs", "manuscript", "pw_inputs_y0.png"), bbox_inches = "tight")
end

PLT.figure(fig_num, figsize = fig_size, dpi = dpi)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs)
PLT.plot(LinRange(lb, ub, Nq), mqs, "*", label = "query positions")
PLT.legend()
PLT.title("Patchwork kriging results")

y_rkhs = f_oracle.(X)
σ² = T(1.0e-8)
θ = θs[1]
model_rkhs = PK.RKHSModel(X, y_rkhs, σ², θ)
mqs_rkhs, vqs_rkhs, _ = query_line_rkhs(lb, ub, y_val, Nq, model_rkhs)

PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs)
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs, "*", label = "query positions")
PLT.legend()
PLT.title("Predictive posterior mean")

fqs = f_oracle.(xqs)
PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), fqs)
PLT.title("Oracle function")

PLT.figure(fig_num, figsize = fig_size, dpi = dpi)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs, "x", label = "Conventional GPR")
PLT.plot(LinRange(lb, ub, Nq), mqs, "o", label = "Patchwork kriging")
PLT.plot(LinRange(lb, ub, Nq), fqs, label = "Oracle")
PLT.legend()
PLT.xlabel("x₁")
PLT.ylabel("Function output")
PLT.title("Predictive posterior mean, x₂ = $(y_val)")
if save_fig
    PLT.savefig(joinpath("figs", "manuscript", "pw_results_y0.png"), bbox_inches = "tight")
end

nothing
