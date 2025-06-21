# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

using Pkg
Pkg.activate(".")

using Random, LinearAlgebra, Statistics, BSON, BenchmarkTools, Serialization
import SingleLinkagePartitions as SL

const T = Float64

import PythonPlot as PLT
PLT.close("all")
fig_num = 1

data = BSON.load(joinpath("..", "data", "cache", "shore.bson"))
x1 = data[:x1s]
x2 = data[:x2s]
ys = data[:x3s]


PLT.figure(fig_num)
fig_num += 1
PLT.plot3D(x1, x2, ys, "x")
PLT.title("shore dataset")

X = collect([x1[i]; x2[i]] for i in eachindex(x1, x2))
@show length(X), length(unique(X))

## rounder trees, no repeats
# x1_lb = 445550
# x1_ub = 445600

## sharper trees, has repeats
x1_lb = 445700
x1_ub = 445750

x2_lb = 257675
x2_ub = 257700
inds = findall(
    xx -> ((x1_lb < first(xx) < x1_ub) && (x2_lb < last(xx) < x2_ub)),
    X
)

x1 = x1[inds]
x2 = x2[inds]
ys0 = ys[inds]
U0 = X[inds]
@show length(U0), length(unique(U0))


# collapse duplicate points.
U, ys = SL.replaceduplicates(SL.UseMinimum(), U0, ys0, T(1.0e-5))
x1 = map(xx -> first(xx), U)
x2 = map(xx -> last(xx), U)


PLT.figure(fig_num)
fig_num += 1
PLT.plot3D(x1, x2, ys, "x")
PLT.title("reduced dataset")


PLT.figure(fig_num)
fig_num += 1
PLT.plot(x1, x2, "x")
PLT.title("x-y reduced dataset")

# inverse distance weighting height map here.
import ScatteredInterpolation
# import LazyGPR as LGP
# f_shepard = LGP.create_warp_map(
#     LGP.UseScatteredInterpolation(ScatteredInterpolation),
#     U, ys, 3,
# )

D = 2
X, y = U, ys
X_mat = reshape(collect(Iterators.flatten(X)), D, length(X))
itp_shepard = ScatteredInterpolation.interpolate(
    ScatteredInterpolation.Shepard(2),
    X_mat,
    y,
)
f_shepard = xx -> ScatteredInterpolation.evaluate(itp_shepard, collect(xx))[begin]
#f_shepard = xx -> sinc(norm(xx))

# query
x1_ub, x1_ub = 445700, 445750
x2_lb, x2_ub = 257675, 257700

Nq1 = 1000
Nq2 = 500
Xqrs = (
    LinRange(x1_lb, x1_ub, Nq1),
    LinRange(x2_lb, x2_ub, Nq2),
)
Xq = collect(Iterators.product(Xqrs...))

@time f_shepard_Xq = collect(f_shepard(collect(x)) for x in Iterators.product(Xqrs...))
# 9700X: 41.663823 seconds (14.56 M allocations: 147.760 GiB, 8.41% gc time, 0.11% compilation time)

#using NaturalNeighbours
#sibson_vals = itp(xq, yq; method = Sibson())


# visualize
xq1 = map(xx -> first(xx), Xq)
xq2 = map(xx -> last(xx), Xq)

# PLT.figure(fig_num)
# fig_num += 1
# PLT.imshow(f_shepard_Xq)
# PLT.scatter(x1, x2, "x")
# PLT.title("Sharpard's interpolation")

import VisualizationBag as VIZ

var_cmap = "Greys"
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    f_shepard_Xq,
    X,
    "o",
    fig_num,
    "Inverse distance weighting";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

var_cmap = "Greys"
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    f_shepard_Xq,
    [],
    "o",
    fig_num,
    "Inverse distance weighting";
    cmap = var_cmap,
    horizontal_title = "Longitude",
    vertical_title = "Latitude",
    matrix_mode = true,
)
PLT.gca().invert_yaxis()

nothing
