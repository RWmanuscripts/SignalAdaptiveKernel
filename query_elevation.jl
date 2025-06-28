# run setup.jl first to setup the elevation data.

PLT.close("all")
fig_num = 1

import Distances
import LazyGPR as LGP
import SingleLinkagePartitions as SL
import SpatialGSP as GSP
import Metaheuristics as EVO

Random.seed!(25)
D = 2

X = deepcopy(U)
y = copy(ys)
σ² = one(T)

ms_trait = LGP.MarginalLikelihood()
#width_factor = 3
width_factor = 1
f_calls_limit = 1_000

gp_data = LGP.GPData(σ², X, y)

# # Warp map
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

PLT.figure(fig_num)
fig_num += 1
PLT.plot3D(x1, x2, W, "x")
PLT.title("reduced dataset: warp samples")

# # Fit hyperparameters
ref_dek = LGP.DEKernel(
    LGP.SqExpKernel(one(T)), warpmap, zero(T),
)

a_lb = convert(T, 1.0e-3)
a_ub = convert(T, 60)
κ_ub = maximum(abs.(W)) * 100

Random.seed!(25)
# println("Running: optimize_kernel_hp_separately")
# @show ms_trait
# @time dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep = LGP.optimize_kernel_hp_separately(
#     LGP.UseMetaheuristics(EVO),
#     ref_dek,
#     ms_trait,
#     gp_data,
#     LGP.HCostConfig(),
#     LGP.MetaheuristicsConfig(
#         f_calls_limit = f_calls_limit,
#     ),
#     LGP.KernelOptimConfig{T}(
#         a_lb = a_lb,
#         a_ub = a_ub,
#         κ_ub = κ_ub,
#         width_factor = convert(T, width_factor),
#     ),
# )
# @show dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep
# println()
# serialize(joinpath("results", "elevation_hp"), (dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep))
# ## 9700X: 1631.433041 seconds (7.19 M allocations: 1.005 TiB, 1.27% gc time, 0.10% compilation time: 5% of which was recompilation)

# # Query

dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep = deserialize(joinpath("results", "elevation_hp"))
sk_vars = sk_vars_sep
dek_vars = dek_vars_sep

println("Querying:")

x1_lb, x1_ub = 445700, 445750
x2_lb, x2_ub = 257675, 257700

Nq1 = 500 # 199
Nq2 = 250 # 50
Xqrs = (
    LinRange(x1_lb, x1_ub, Nq1),
    LinRange(x2_lb, x2_ub, Nq2),
)
Xq = collect(Iterators.product(Xqrs...))

# ## stationary kernel
sk = LGP.SqExpKernel(sk_vars[begin]) # very narrow bandwidth.
gp_sk = LGP.fitGP(X, y, σ², sk)

@time sk_Xq = collect(
    LGP.queryGP(collect(x), sk, gp_sk)
        for x in Iterators.product(Xqrs...)
)
mqs_sk = map(xx -> xx[begin], sk_Xq)
vqs_sk = map(xx -> xx[begin + 1], sk_Xq)
# 9700X: 116.957497 seconds (154.98 k allocations: 1.185 GiB, 0.15% gc time, 0.04% compilation time)

# ## DE kernel
a_star, κ_star = dek_vars
dek = LGP.DEKernel(
    LGP.SqExpKernel(a_star),
    warpmap,
    κ_star,
)
gp_dek = LGP.fitGP(X, y, σ², dek)
# 9700X: 114.471348 seconds (849.16 k allocations: 2.691 GiB, 0.02% gc time, 0.10% compilation time)

@time dek_Xq = collect(
    LGP.queryGP(collect(x), dek, gp_dek)
        for x in Iterators.product(Xqrs...)
)
mqs_dek = map(xx -> xx[begin], dek_Xq)
vqs_dek = map(xx -> xx[begin + 1], dek_Xq)

@time f_shepard_Xq = collect(f_shepard(collect(x)) for x in Iterators.product(Xqrs...))

x1s = map(xx -> xx[1], X)
x2s = map(xx -> xx[2], X)
itp_nn = NN.interpolate(x1s, x2s, y; derivatives = true)

println("Natural neighbors: Sibson, Laplace, Hiyoshi-2, batch query:")
@time f_nn_Xq_s = collect(itp_nn(x[1], x[2]; method = NN.Sibson()) for x in Iterators.product(Xqrs...))
@time f_nn_Xq_l = collect(itp_nn(x[1], x[2]; method = NN.Laplace()) for x in Iterators.product(Xqrs...))
@time f_nn_Xq_h = collect(itp_nn(x[1], x[2]; method = NN.Hiyoshi(2)) for x in Iterators.product(Xqrs...))

serialize(
    joinpath("results", "elevation_queries"),
    (
        U, f_shepard_Xq, f_nn_Xq_s, f_nn_Xq_l, f_nn_Xq_h,
        mqs_dek, vqs_dek, mqs_sk, vqs_sk,
        x1_lb, x1_ub, x2_lb, x2_ub, Nq1, Nq2,
    )
)

# # Visualize
include("figs_elevation.jl")

nothing
