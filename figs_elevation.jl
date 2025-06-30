# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

using LinearAlgebra, Statistics, Serialization, BenchmarkTools

import PythonPlot as PLT

PLT.close("all")
fig_num = 1

# Load elevation query results.
U, f_shepard_Xq, f_nn_Xq_s, f_nn_Xq_l, f_nn_Xq_h,
    mqs_dek, vqs_dek, mqs_sk, vqs_sk,
    x1_lb, x1_ub, x2_lb, x2_ub, Nq1, Nq2 = deserialize(joinpath("results", "elevation_queries"))
#
T = eltype(x1_lb)

# convert to coordinates.
x1_lb = x1_lb / 1_000
x1_ub = x1_ub / 1_000
x2_lb = x2_lb / 1_000
x2_ub = x2_ub / 1_000
U = collect(u ./ 1_000 for u in U)

Xqrs = (
    LinRange(x1_lb, x1_ub, Nq1),
    LinRange(x2_lb, x2_ub, Nq2),
)
Xq = collect(Iterators.product(Xqrs...))

# generate figures for the manuscript.

var_vmin = min(minimum(vqs_dek), minimum(vqs_sk))
var_vmax = max(maximum(vqs_dek), maximum(vqs_sk))

vmin = min(
    minimum(f_nn_Xq_s),
    minimum(f_shepard_Xq),
    minimum(f_nn_Xq_l),
    minimum(f_nn_Xq_h),
    minimum(mqs_dek),
    minimum(mqs_sk),
)
vmax = max(
    maximum(f_nn_Xq_s),
    maximum(f_shepard_Xq),
    maximum(f_nn_Xq_l),
    maximum(f_nn_Xq_h),
    maximum(mqs_dek),
    maximum(mqs_sk),
)
var_cmap = "Greys_r"
#var_cmap = "summer"

function plot_subfig(
        PLT, # PythonPlot
        x_ranges_inp::Vector{RT},
        Y::Matrix{T};
        marker_locations::Vector = [],
        marker_symbol::String = "x",
        marker_color = "black",
        cmap = "Greys_r", # see https://matplotlib.org/stable/gallery/color/colormap_reference.html
        vmin = minimum(Y), # color bar range's minimum.
        vmax = maximum(Y), # color bar range's maximum.
        matrix_mode::Bool = false, # flip the vertical axis.
        color_bar_params = (0.085, 0.04),
        display_color_bar = true,
        symmetric_color_range = false, # if active, overrides vmin and vmax.
        vcenter = NaN, # uses TwoSlopeNorm
        rasterized = false,
    ) where {T <: Real, RT <: AbstractRange}

    if symmetric_color_range
        z = max(abs(minimum(Y)), abs(maximum(Y)))
        vmin = -z
        vmax = z
    end

    x_ranges = x_ranges_inp
    markers = marker_locations
    if matrix_mode
        # first dimension is row, which should be the second (i.e. the veritical) dimension for pcolormesh().
        x_ranges = reverse(x_ranges_inp)
        markers = reverse.(marker_locations)
    end

    #
    @assert length(x_ranges) == 2
    x_coords = collect(collect(x_ranges[d]) for d in 1:2)

    #@show vmin, vmax, vcenter
    ph = []
    if isfinite(vcenter)
        div_norm = PLT.matplotlib.colors.TwoSlopeNorm(vmin = vmin, vcenter = vcenter, vmax = vmax)
        ph = PLT.pcolormesh(x_coords[1], x_coords[2], Y, cmap = cmap, shading = "auto", norm = div_norm, rasterized = rasterized)
    else
        ph = PLT.pcolormesh(x_coords[1], x_coords[2], Y, cmap = cmap, shading = "auto", vmin = vmin, vmax = vmax, rasterized = rasterized)
    end

    if !isempty(markers)
        p1 = map(xx -> xx[1], markers)
        p2 = map(xx -> xx[2], markers)
        PLT.plot(p1, p2, marker_symbol, color = marker_color)
    end


    if display_color_bar
        if isempty(color_bar_params)
            PLT.colorbar()
        else
            # control the size of the color bar.
            fraction, pad = color_bar_params
            PLT.colorbar(fraction = fraction, pad = pad)
        end
    end

    PLT.axis("scaled")

    if matrix_mode
        PLT.gca().invert_yaxis()
    end
    return nothing
end

save_dpi = 150

PLT.figure(fig_num, figsize = (9, 12), dpi = 96)
fig_num += 1

PLT.subplot(231)
plot_subfig(
    PLT,
    collect(Xqrs),
    f_nn_Xq_s;
    marker_locations = [],
    marker_symbol = "o",
    cmap = var_cmap,
    matrix_mode = true,
    vmin = vmin,
    vmax = vmax,
    display_color_bar = true,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("Sibson")

PLT.subplot(232)
plot_subfig(
    PLT,
    collect(Xqrs),
    mqs_dek;
    marker_locations = [],
    marker_symbol = "o",
    cmap = var_cmap,
    matrix_mode = true,
    vmin = vmin,
    vmax = vmax,
    display_color_bar = true,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("DEK: mean")

PLT.subplot(233)
plot_subfig(
    PLT,
    collect(Xqrs),
    log.(vqs_dek);
    marker_locations = [],
    marker_symbol = "o",
    cmap = var_cmap,
    matrix_mode = true,
    display_color_bar = true,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("DEK: log variance")


PLT.subplot(234)
plot_subfig(
    PLT,
    collect(Xqrs),
    f_shepard_Xq;
    marker_locations = [],
    marker_symbol = "o",
    cmap = var_cmap,
    matrix_mode = true,
    vmin = vmin,
    vmax = vmax,
    display_color_bar = true,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("IDW")

PLT.subplot(235)
plot_subfig(
    PLT,
    collect(Xqrs),
    mqs_sk;
    marker_locations = [],
    marker_symbol = "o",
    cmap = var_cmap,
    matrix_mode = true,
    vmin = vmin,
    vmax = vmax,
    display_color_bar = true,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("SK: mean")

PLT.subplot(236)
plot_subfig(
    PLT,
    collect(Xqrs),
    log.(vqs_sk);
    marker_locations = [],
    marker_symbol = "o",
    cmap = var_cmap,
    matrix_mode = true,
    display_color_bar = true,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("SK: log variance")

# adjust margins.
PLT.subplots_adjust(
    left = 0.04, right = 1 - 0.04, top = 1 - 0.03, bottom = 0.03,
    wspace = 0.15, hspace = 0.1,
)
# PLT.xlabel("Longitude")
# PLT.ylabel("Latitude")
PLT.savefig(joinpath("figs", "manuscript", "elevation_r.png"), dpi = save_dpi)

PLT.figure(fig_num, figsize = (9, 7), dpi = 96)
fig_num += 1
plot_subfig(
    PLT,
    collect(Xqrs),
    ones(T, size(f_shepard_Xq)) .* vmax; # force black.
    marker_locations = U,
    marker_symbol = "x",
    cmap = var_cmap,
    matrix_mode = true,
    display_color_bar = false,
    vmin = vmin,
    vmax = vmax,
)
PLT.gca().invert_yaxis()
PLT.axis("off")
PLT.title("Observation positions")
PLT.tight_layout()
PLT.savefig(joinpath("figs", "manuscript", "elevation_l.png"), dpi = save_dpi)

nothing
