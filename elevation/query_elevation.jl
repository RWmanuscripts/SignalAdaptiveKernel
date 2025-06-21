# run setup.jl first to setup the elevation data.

#PLT.close("all")
#fig_num = 1

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
println("Running: optimize_kernel_hp_separately")
@show ms_trait
@time dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep = LGP.optimize_kernel_hp_separately(
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
serialize("tmp", (dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep))
## 9700X: 1631.433041 seconds (7.19 M allocations: 1.005 TiB, 1.27% gc time, 0.10% compilation time: 5% of which was recompilation)

dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep = deserialize("tmp")

sk_vars = sk_vars_sep
dek_vars = dek_vars_sep

# # Query
println("Querying:")

x1_lb, x1_ub = 445700, 445750
x2_lb, x2_ub = 257675, 257700

Nq1 = 100
Nq2 = 50
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

# ## DE kernel's canonical kernel
ck = LGP.SqExpKernel(dek_vars[begin])
gp_ck = LGP.fitGP(X, y, σ², ck)

@time ck_Xq = collect(
    LGP.queryGP(collect(x), ck, gp_ck)
        for x in Iterators.product(Xqrs...)
)
mqs_ck = map(xx -> xx[begin], ck_Xq)
vqs_ck = map(xx -> xx[begin + 1], ck_Xq)
# 9700X: 116.771353 seconds (154.98 k allocations: 1.185 GiB, 0.14% gc time, 0.04% compilation time)

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

# ## Timing

function timing_sk(Xq::AbstractMatrix{Tuple{T, T}}, sk, gp_sk) where {T}
    buf = zeros(T, 2)
    for i in eachindex(Xq)
        buf[1] = Xq[i][1]
        buf[2] = Xq[i][2]
        LGP.queryGP(buf, sk, gp_sk)
    end
    return nothing
end

function timing_dek(Xq::AbstractMatrix{Tuple{T, T}}, dek, gp_dek) where {T}
    buf = zeros(T, 2)
    for i in eachindex(Xq)
        buf[1] = Xq[i][1]
        buf[2] = Xq[i][2]
        LGP.queryGP(buf, dek, gp_dek)
    end
    return nothing
end
@btime timing_sk(view($Xq, 30:40, 30:40), $sk, $gp_sk)
# 2.830 s (1817 allocations: 29.27 MiB)

@btime timing_dek(view($Xq, 30:40, 30:40), $dek, $gp_dek)
# 2.812 s (4963 allocations: 65.88 MiB)

@btime timing_sk(view($Xq, 30:60, 30:50), $sk, $gp_sk)
# 15.177 s (9767 allocations: 157.49 MiB)

@btime timing_dek(view($Xq, 30:60, 30:50), $dek, $gp_dek)
# 14.917 s (26685 allocations: 354.35 MiB)

# @benchmark timing_sk($Xq, $sk, $gp_sk)
# #

# @benchmark timing_dek($Xq, $dek, $gp_dek)
# #
"""
Running: optimize_kernel_hp_separately
ms_trait = LazyGPR.MarginalLikelihood()
1635.779935 seconds (6.52 M allocations: 1.005 TiB, 1.29% gc time, 0.09% compilation time: 5% of which was recompilation)
(dek_vars_sep, dek_star_sep, sk_vars_sep, sk_star_sep) = ([0.008050425568391638, 9.369090547257047], 50452.20560469506, [0.008050425568391638], 75512.38122364797)

Querying:
114.990419 seconds (602.77 k allocations: 1.207 GiB, 0.19% gc time, 0.17% compilation time)
119.002248 seconds (154.98 k allocations: 1.185 GiB, 0.14% gc time, 0.04% compilation time)
118.773465 seconds (849.15 k allocations: 2.691 GiB, 0.02% gc time, 0.09% compilation time)
2.770 s (1817 allocations: 29.27 MiB)
2.874 s (4963 allocations: 65.88 MiB)
14.899 s (9767 allocations: 157.49 MiB)
15.469 s (26685 allocations: 354.35 MiB)
"""

# # Visualize
var_cmap = "Greys"

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_dek,
    [],
    "o",
    fig_num,
    "Dimension-expansion kernel: mean";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    log.(vqs_dek),
    [],
    "o",
    fig_num,
    "Dimension-expansion kernel: log variance";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_sk,
    [],
    "o",
    fig_num,
    "stationary kernel: mean";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    log.(vqs_sk),
    [],
    "o",
    fig_num,
    "stationary kernel: log variance";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_ck,
    [],
    "o",
    fig_num,
    "canonical kernel: mean";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    log.(vqs_ck),
    [],
    "o",
    fig_num,
    "canonical kernel: log variance";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

# # via pyplot
# xq1 = map(xx -> first(xx), Xq)
# xq2 = map(xx -> last(xx), Xq)

# PLT.figure(fig_num)
# fig_num += 1
# PLT.plot3D(xq1, xq2, mqs_sk, "x")
# PLT.title("sk mean")

# PLT.figure(fig_num)
# fig_num += 1
# PLT.plot3D(xq1, xq2, vqs_sk, "x")
# PLT.title("sk variance")

# PLT.figure(fig_num)
# fig_num += 1
# PLT.plot3D(xq1, xq2, mqs_ck, "x")
# PLT.title("ck mean")

# PLT.figure(fig_num)
# fig_num += 1
# PLT.plot3D(xq1, xq2, vqs_ck, "x")
# PLT.title("ck variance")

# PLT.figure(fig_num)
# fig_num += 1
# PLT.plot3D(xq1, xq2, mqs_dek, "x")
# PLT.title("dek mean")

# PLT.figure(fig_num)
# fig_num += 1
# PLT.plot3D(xq1, xq2, vqs_dek, "x")
# PLT.title("dek variance")


nothing
