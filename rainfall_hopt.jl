using Pkg
Pkg.activate(".")

import Random
Random.seed!(25)

using LinearAlgebra
using Serialization
using Statistics

import Distances
import LazyGPR as LGP
import SingleLinkagePartitions as SL
import SpatialGSP as GSP

import Metaheuristics as EVO
import DataFrames as DF
import CSV

T = Float64
D = 2

# helper functions.
include("helpers/evo.jl")


# user inputs.

save_results_dir = "results/"
if !ispath(save_results_dir)
    mkpath(save_results_dir)
end

width_factor = 3
f_calls_limit = 1_000

ms_trait = LGP.MarginalLikelihood()

σ² = convert(T, 1.0e-3)

# # Load data

data_path = "data/2023_08_10_CA_stations.csv"
df = CSV.read(data_path, DF.DataFrame)

X0 = collect(
    [
            df.LATITUDE[n];
            df.LONGITUDE[n];
        ]
        for n in eachindex(df.LATITUDE)
)
y0 = convert(Vector{T}, df.PRCP ./ 10)
X, y = SL.avgduplicates(X0, y0, eps(T) * 10)
X_mat = reshape(
    collect(Iterators.flatten(X)),
    D, length(X),
)

axis_config = GSP.AxisSearchConfig{T}(
    kernel_σ_multiplier = convert(T, 3),
    w_lb = convert(T, 0.1),
)

warp_config = GSP.WarpConfig{T}()
smooth_iters = 0
W = LGP.create_warp_samples(
    LGP.UseSpatialGSP(GSP),
    LGP.AxisGraph(),
    X, y,
    axis_config,
    warp_config,
)

import ScatteredInterpolation
warpmap = LGP.create_warp_map(
    LGP.UseScatteredInterpolation(ScatteredInterpolation),
    X, W, 3,
)


ref_dek = LGP.DEKernel(
    LGP.SqExpKernel(one(T)), warpmap, zero(T),
)

a_lb = convert(T, 1.0e-3)
a_ub = convert(T, 60)
κ_ub = maximum(abs.(W)) * 100

######

gp_data = LGP.GPData(σ², X, y)

Random.seed!(25)
println("Running: optimize_kernel_hp_separately")
@show ms_trait
dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep = LGP.optimize_kernel_hp_separately(
    LGP.UseMetaheuristics(EVO),
    ref_dek,
    ms_trait,
    gp_data,
    LGP.HCostConfig(),
    LGP.MetaheuristicsConfig(
        f_calls_limit = f_calls_limit,
    ),
    LGP.KernelOptimConfig{T}(
        a_lb = a_lb,
        a_ub = a_ub,
        κ_ub = κ_ub,
        width_factor = convert(T, width_factor),
    ),
)
@show dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep
println()

##
Random.seed!(25)
println("Running: optimize_kernel_hp, sk")
sk_ref = ref_dek.canonical
p0s = collect([x;] for x in LinRange(a_lb, a_ub, 100))
sk_vars, sk_star = LGP.optimize_kernel_hp(
    LGP.UseMetaheuristics(EVO),
    sk_ref,
    ms_trait,
    gp_data,
    LGP.HCostConfig(),
    LGP.MetaheuristicsConfig(
        f_calls_limit = f_calls_limit,
    ),
    LGP.OptimContainer([a_lb;], [a_ub;], p0s)
)
@show sk_vars, sk_star
println()

##
Random.seed!(25)
println("Running: optimize_kernel_hp, dek")
a0s = collect(LinRange(a_lb, a_ub, 10))
push!(a0s, sk_vars[begin])
κ0s = LinRange(0, κ_ub, 20)
p0s = collect.(
    vec(collect(Iterators.product(a0s, κ0s)))
)
dek_vars, dek_star = LGP.optimize_kernel_hp(
    LGP.UseMetaheuristics(EVO),
    ref_dek,
    ms_trait,
    gp_data,
    LGP.HCostConfig(),
    LGP.MetaheuristicsConfig(
        f_calls_limit = f_calls_limit,
    ),
    LGP.OptimContainer([a_lb; zero(T)], [a_ub; κ_ub;], p0s)
)
@show dek_vars, dek_star
@show a_lb, a_ub, κ_ub, width_factor
println()

tag = "ML"

# using Serialization
# serialize(
#     joinpath(save_results_dir, "CA_rainfall_hp_$(tag)"),
#     (
#         dek_vars_sep, dek_star_sep,
#         sk_vars_sep, sk_star_sep,
#         dek_vars, dek_star,
#         sk_vars, sk_star,
#         a_lb, a_ub, κ_ub, width_factor,
#     )
# )


nothing
