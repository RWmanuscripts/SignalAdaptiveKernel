# this script load kernel hyperparameters, and queries the sourthern Canada rainfall data.

using Pkg
Pkg.activate(".")


save_results_dir = "results/"
if !ispath(save_results_dir)
    mkpath(save_results_dir)
end

import Random
Random.seed!(25)

using LinearAlgebra
using Serialization

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

σ² = convert(T, 1e-3)

tag = "ML"

#################

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
X, y = SL.avgduplicates(X0, y0, eps(T)*10)
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


# # Load hyperparameters

ms_trait = LGP.MarginalLikelihood()

using Serialization
(
    dek_vars_sep, dek_star_sep,
    sk_vars_sep, sk_star_sep,
    dek_vars, dek_star,
    sk_vars, sk_star,
) = deserialize(
    joinpath(save_results_dir, "CA_rainfall_hp_$(tag)"),  
)

# # Set up query region

# ## near Ottawa.
x1_lb, x1_ub = 42.0, 51.0
x2_lb, x2_ub = -80.0, -68.0

Nq1 = 90
Nq2 = 120
Xqrs = (
    LinRange(x1_lb, x1_ub, Nq1),
    LinRange(x2_lb, x2_ub, Nq2),
)
Xq = collect( Iterators.product(Xqrs...) )


# # Query

# ## stationary kernel
sk = LGP.SqExpKernel(sk_vars[begin]) # very narrow bandwidth.
gp_sk = LGP.fitGP(X, y, σ², sk)

sk_Xq = collect(
    LGP.queryGP(collect(x), sk, gp_sk)
    for x in Iterators.product(Xqrs...)
)

mqs_sk = map(xx->xx[begin], sk_Xq)
vqs_sk = map(xx->xx[begin+1], sk_Xq)


# ## DE kernel's canonical kernel
ck = LGP.SqExpKernel(dek_vars[begin])
gp_ck = LGP.fitGP(X, y, σ², ck)

ck_Xq = collect(
    LGP.queryGP(collect(x), ck, gp_ck)
    for x in Iterators.product(Xqrs...)
)

mqs_ck = map(xx->xx[begin], ck_Xq)
vqs_ck = map(xx->xx[begin+1], ck_Xq)

# ## DE kernel
a_star, κ_star = dek_vars
dek = LGP.DEKernel(
    LGP.SqExpKernel(a_star),
    warpmap,
    κ_star,
)
gp_dek = LGP.fitGP(X, y, σ², dek)

dek_Xq = collect(
    LGP.queryGP(collect(x), dek, gp_dek)
    for x in Iterators.product(Xqrs...)
)

mqs_dek = map(xx->xx[begin], dek_Xq)
vqs_dek = map(xx->xx[begin+1], dek_Xq)

serialize(
    joinpath(save_results_dir, "CA_rainfall_queries_$(tag)"),
    (X, Xqrs, y, mqs_sk, vqs_sk, mqs_ck, vqs_ck, mqs_dek, vqs_dek),
)

nothing