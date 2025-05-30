# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# run this once: include("a.jl")

using Random, LinearAlgebra, Statistics, PointClouds
#import PythonPlot as PLT

#import Colors
#include("helpers/visualization.jl")

#PLT.close("all")
#fig_num = 1

using GLMakie

const T = Float64
const D = 2

rng = Random.Xoshiro(0)

scene_label = "shore"
#scene_label = "stream"

load_path = joinpath("data", "elevation", "points_stream.laz")

if scene_label == "shore"
    load_path = joinpath("data", "elevation", "points_shore.laz")
end

las = LAS(load_path)

@show getcrs(las) # EPSG:4269
Xs0 = coordinates(las; crs = "EPSG:4269")
Xs = coordinates(las)


# store to file.
x1s = map(xx -> xx[1], Xs)
x2s = map(xx -> xx[2], Xs)
x3s = map(xx -> xx[3], Xs)
using BSON
bson(
    joinpath("data", "cache", "$(scene_label).bson"),
    x1s = x1s, x2s = x2s, x3s = x3s,
)

# use Float32 for Makie.
data = Point3f.(Xs)
x1s = map(xx -> xx[1], data)
x2s = map(xx -> xx[2], data)
x3s = map(xx -> xx[3], data)


# could invert this.

#PLT.scatter3D(x1s, x2s, x3s)
#PLT.scatter(x1s, x2s)

# https://docs.makie.org/stable/reference/plots/scatter#Dealing-with-outline-artifacts-in-GLMakie

# ps = rand(Point3f, 500)
# cs = rand(500)

ps = data
include("helpers/utils.jl")
lb, ub = minimum(x3s), maximum(x3s)
cs = convertcompactdomain(x3s, lb, ub, Float32(0), Float32(1))
#cs = rand(length(x3s))

# ps = ps[1:3]
# cs = cs[1:3]
# fill!(cs, 100)

f = Figure(size = (900, 650))

# Label(f[1, 1], "depthsorting = true", tellwidth = false)
# scatter(f[2, 1], ps, markersize = 5, depthsorting = true)
# Label(f[1, 2], "depthsorting = true", tellwidth = false)
# scatter(f[2, 2], ps, color = cs, markersize = 5, depthsorting = true)
# mesh!(Rect3f(Point3f(0), Vec3f(0.9, 0.9, 0.9)), color = :orange)
Label(f[1, 1], "depthsorting = true1", tellwidth = false)
scatter(f[2, 1], ps, markersize = 5, depthsorting = true)

# Label(f[1, 2], "depthsorting = true2", tellwidth = false)
# #scatter(f[2, 2], ps, color = cs, markersize = 5, fxaa = false)
# scatter(f[2, 2], ps, color = cs, markersize = 5, depthsorting = true)

#mesh!(Rect3f(Point3f(0), Vec3f(0.9, 0.9, 0.9)), color = :orange)
f
