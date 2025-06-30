##### revision June 30, 2025

function plot_subfig(
        PLT, # PythonPlot
        x_ranges_inp::Vector{RT},
        Y::Matrix{T};
        marker_locations::Vector = [],
        marker_symbols::Vector = [],
        marker_colors::Vector = [],
        cmap = "Greys_r", # see https://matplotlib.org/stable/gallery/color/colormap_reference.html
        vmin = minimum(Y), # color bar range's minimum.
        vmax = maximum(Y), # color bar range's maximum.
        matrix_mode::Bool = false, # flip the vertical axis.
        color_bar_params = (0.09, 0.04),
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

    if !isempty(markers) && !isempty(marker_colors)
        p1 = map(xx -> xx[1], markers)
        p2 = map(xx -> xx[2], markers)

        for i in eachindex(p1, p2, marker_colors, marker_symbols)
            PLT.plot([p1[i];], [p2[i];], marker_symbols[i], color = marker_colors[i])
        end
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
