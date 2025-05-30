# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

using Random, LinearAlgebra, Statistics, BSON, BenchmarkTools

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


import SingleLinkagePartitions as SL
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


nothing
