# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

using Pkg
Pkg.activate(".")

using Random, LinearAlgebra, Statistics, BSON, BenchmarkTools, Serialization
import SingleLinkagePartitions as SL

const T = Float64
const D = 2

import PythonPlot as PLT
PLT.close("all")
fig_num = 1

data = BSON.load(joinpath("data", "cache", "shore.bson"))
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
X, y = U, ys
X_mat = reshape(collect(Iterators.flatten(X)), D, length(X))
itp_shepard = ScatteredInterpolation.interpolate(
    ScatteredInterpolation.Shepard(3),
    X_mat,
    y,
)
struct ShepardWrapper{IT}
    itp::IT
end
function (A::ShepardWrapper)(x::AbstractVector)
    return ScatteredInterpolation.evaluate(itp_shepard, x)[begin]
end
f_shepard = ShepardWrapper(itp_shepard)

# # Query
x1_lb, x1_ub = 445700, 445750
x2_lb, x2_ub = 257675, 257700

# Nq1 = 1000
# Nq2 = 500
# Xqrs = (
#     LinRange(x1_lb, x1_ub, Nq1),
#     LinRange(x2_lb, x2_ub, Nq2),
# )
# Xq = collect(Iterators.product(Xqrs...))

Nq1 = 500
Nq2 = 250
Xqrs = (
    LinRange(x1_lb, x1_ub, Nq1),
    LinRange(x2_lb, x2_ub, Nq2),
)
Xq = collect(Iterators.product(Xqrs...))

println("Inverse distance weighting, batch query:")
@time f_shepard_Xq = collect(f_shepard(collect(x)) for x in Iterators.product(Xqrs...))
println("Inverse distance weighting, single query:")
xq_test = [T(257686.3); T(445728.9)]
f_shepard(xq_test)
q = @benchmark $f_shepard($xq_test)
display(q)
println()

# # Natural Neighbors
import NaturalNeighbours as NN

x1s = map(xx -> xx[1], X)
x2s = map(xx -> xx[2], X)
itp_nn = NN.interpolate(x1s, x2s, y; derivatives = true)

println("Natural neighbors: Sibson, Laplace, Hiyoshi-2, batch query:")
@time f_nn_Xq_s = collect(itp_nn(x[1], x[2]; method = NN.Sibson()) for x in Iterators.product(Xqrs...))
@time f_nn_Xq_l = collect(itp_nn(x[1], x[2]; method = NN.Laplace()) for x in Iterators.product(Xqrs...))
@time f_nn_Xq_h = collect(itp_nn(x[1], x[2]; method = NN.Hiyoshi(2)) for x in Iterators.product(Xqrs...))

println("Sibson, single query:")
itp_nn(xq_test[1], xq_test[2]; method = NN.Sibson())
q = @benchmark $itp_nn($xq_test[1], $xq_test[2]; method = NN.Sibson())
display(q)

println("Laplace, single query:")
itp_nn(xq_test[1], xq_test[2]; method = NN.Laplace())
q = @benchmark $itp_nn($xq_test[1], $xq_test[2]; method = NN.Laplace())
display(q)

println("Hiyoshi-2, single query:")
itp_nn(xq_test[1], xq_test[2]; method = NN.Hiyoshi(2))
q = @benchmark $itp_nn($xq_test[1], $xq_test[2]; method = NN.Hiyoshi(2))
display(q)
println()

nothing
