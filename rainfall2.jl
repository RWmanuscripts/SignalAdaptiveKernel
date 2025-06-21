# explore hyperparameter selection for rainfall.


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

@assert 1 == 23

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

#####


# # Table for the rainfall dataset

tag = "ML"

sig_digits = 3
using Serialization

# # Load query results.
load_results_dir = "tmp/"
if !ispath(load_results_dir)
    mkpath(load_results_dir)
end

X, Xqrs, y, mqs_sk, vqs_sk, mqs_ck, vqs_ck, mqs_dek, vqs_dek = deserialize(
    joinpath(load_results_dir, "CA_rainfall_queries_$(tag)"),
)

sig_digits = 3

# # Load query results.
load_results_dir = "tmp/"
if !ispath(load_results_dir)
    mkpath(load_results_dir)
end

X, Xqrs, y, mqs_sk, vqs_sk, mqs_ck, vqs_ck, mqs_dek, vqs_dek = deserialize(
    joinpath(load_results_dir, "CA_rainfall_queries_$(tag)"),
)
max_dek, min_dek = round(maximum(mqs_dek), sigdigits = sig_digits), round(minimum(mqs_dek), sigdigits = sig_digits)
max_ck, min_ck = round(maximum(mqs_ck), sigdigits = sig_digits), round(minimum(mqs_ck), sigdigits = sig_digits)
max_sk, min_sk = round(maximum(mqs_sk), sigdigits = sig_digits), round(minimum(mqs_sk), sigdigits = sig_digits)

# # Load hyperparameters
(
    dek_vars_sep, dek_star_sep,
    sk_vars_sep, sk_star_sep,
    dek_vars, dek_star,
    sk_vars, sk_star,
) = deserialize(
    joinpath(load_results_dir, "CA_rainfall_hp_$(tag)"),
)
a2s = aa -> sqrt(1 / (2 * aa)) # a := 1/(2*s^2), solve for s.
bandwidth_can, κ = round(a2s(dek_vars[begin]), sigdigits = sig_digits), round(dek_vars[end], sigdigits = sig_digits)
bandwidth_sk = round(a2s(sk_vars[begin]), sigdigits = sig_digits)

y_min = round(minimum(y), sigdigits = sig_digits)
y_max = round(maximum(y), sigdigits = sig_digits)

# # Assemble summary table
using Markdown, Tables, MarkdownTables

data_mat = [
    "Bandwidth" bandwidth_can bandwidth_sk bandwidth_can "NA";
    "Gain κ" κ "NA" "NA" "NA";
    "Minimum queried mean" min_dek min_sk min_ck y_min;
    "Maximum queried mean" max_dek max_sk max_ck y_max;
];
print(
    markdown_table(
        Tables.table(
            data_mat;
            header = [
                "";
                "DE kernel";
                "Stationary kernel";
                "Canonical kernel";
                "Rainfall data";
            ]
        ), String
    )
)

# # Figure for the rainfall dataset


# # Visualize

#fig_size = VIZ.getaspectratio(size(im_y)) .* 4
dpi = 300
#dpi = 96

#cmap = "Greys"
cmap = "bwr"

#var_cmap = "Reds"
var_cmap = "Greys"

vmax = maximum(y)
vmin = -vmax

# dek
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_dek,
    X, #X,
    "o",
    fig_num,
    "Predictive mean";
    cmap = cmap,
    #symmetric_color_range = true,
    vmin = vmin,
    vmax = vmax,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()
PLT.savefig("tmp/rainfall_DE_mean.png", dpi = dpi, bbox_inches = "tight")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    vqs_dek,
    X,
    "o",
    fig_num,
    "Predictive variance";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()
PLT.savefig("tmp/rainfall_DE_variance.png", dpi = dpi, bbox_inches = "tight")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_sk,
    X, #X,
    "o",
    fig_num,
    "Predictive mean";
    cmap = cmap,
    #symmetric_color_range = true,
    vmin = vmin,
    vmax = vmax,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()
PLT.savefig("tmp/rainfall_sqexp_mean.png", dpi = dpi, bbox_inches = "tight")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    vqs_sk,
    X,
    "o",
    fig_num,
    "Predictive variance";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()
PLT.savefig("tmp/rainfall_sqexp_variance.png", dpi = dpi, bbox_inches = "tight")

# ck
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_ck,
    X, #X,
    "o",
    fig_num,
    "Predictive mean";
    cmap = cmap,
    #symmetric_color_range = true,
    vmin = vmin,
    vmax = vmax,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()
PLT.savefig("tmp/rainfall_canonical_mean.png", dpi = dpi, bbox_inches = "tight")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    vqs_ck,
    X,
    "o",
    fig_num,
    "Predictive variance";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()
PLT.savefig("tmp/rainfall_canonical_variance.png", dpi = dpi, bbox_inches = "tight")


nothing
