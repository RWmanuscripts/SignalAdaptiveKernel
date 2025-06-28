# Visualize the kernel.

down_factor = 2

println()
@show down_factor

# # Setup
# Install dependencies
import Pkg
let
    pkgs = ["PythonPlot"]
    for pkg in pkgs
        if isnothing(Base.find_package(pkg))
            Pkg.add(pkg)
        end
    end
end;

import PythonPlot as PLT
PLT.close("all")
fig_num = 1

import Random
Random.seed!(25)

using LinearAlgebra, Serialization, BenchmarkTools

import Interpolations
import Metaheuristics as EVO
import Images
import LocalFilters
import SpatialGSP as GSP
import LazyGPR as LGP
import Distances

import VisualizationBag as VIZ


const T = Float64
const D = 2;

#Data preparation
data_dir = joinpath("data", "images")
image_file_name = "kodim05_cropped.png"

# data_dir = joinpath("data")
# image_file_name = "kodim23.png"

model_selection_string = "ML" # "ML" means we use marginal likelihood for the hyperparameter optimization objective. Another valid option is "LOOCV", for the leave-one-out cross-validation objective; see  https://doi.org/10.7551/mitpress/3206.001.0001  for details on LOOCV.

#Prameters for smoothing the warp samples before making it into a warp function. The bilateral filter is used.
#If this is larger, then the warp samples are smoother, with increased risk of removing details:
#see compute_kodak_hp() and compute_kodak_warpmap() from the helper script `hopt.jl` to see how `σr_factor` is used to compute `σr`, the bilateral filter intensity standard deviation.
σr_factor = convert(T, 4.0)
σs = one(T); # The spatial standard deviation for the bilateral filter.

# Lazy-evaluation GPR model parameters:
σ² = convert(T, 0.001)
b_x = convert(T, 6);

#For hyperparameter optimization
f_calls_limit = 1_000 # soft-constraint on the number of objective function evaluations during optimization.
N_neighbourhoods = 50 # The maximum number of local datasets we use in the objective function is this number times two. This is `M` in our manuscript.
a_lb = convert(T, 0.001) # lower bound for the bandwidth parameter.
a_ub = one(T)
N_initials_a = 100 # Number of initial guesses for the a, bandwidth parameter.
N_initials_κ = 100; # Similarly for the κ, gain parameter.

#how the warp samples are combined across different graph spectrums.
aggregate_symbol = :sum # An alternative option is :sum_normalized.

if model_selection_string != "ML" &&
        model_selection_string != "LOOCV"

    println("Unknown model_selection_string. Default to Marginal likelihood.")
    model_selection_string = "ML"
end

model_selection_trait = LGP.MarginalLikelihood()
if model_selection_string == "LOOCV"
    model_selection_trait = LGP.LOOCV()
end


# The following helper scripts can be found in `examples/helpers/` from the root repository folder.
include("helpers/hopt.jl")
include("helpers/utils.jl")
include("helpers/image.jl");

# Setup data.
data_path = joinpath(data_dir, image_file_name)
im_y, image_ranges, _ = getdownsampledimage(
    T, data_path, down_factor;
    discard_pixels = 0,
)
Xrs = Tuple(image_ranges);

# adjustment map.
M = floor(Int, b_x) # base this on b_x.
L = M # must be an even positive integer. The larger the flatter.
if isodd(L)
    L = L + 1
end
x0, y0 = convert(T, 0.8 * M), 1 + convert(T, 0.5)
s_map = LGP.AdjustmentMap(x0, y0, b_x, L);

# Warp samples, from the unit grid graph.
warp_config = GSP.WarpConfig{T}(
    aggregate_option = aggregate_symbol,
)
W = LGP.create_grid_warp_samples(
    LGP.UseSpatialGSP(GSP),
    im_y,
    warp_config,
);

sigma_r_factor = T(3)
warpmap = compute_kodak_warpmap(W, sigma_r_factor, Xrs)

# # Setup kernel functions

# Set up marker
mk_red = [T(52); T(80)]
mk_red = [T(53); T(80)]
mk_cyan = [T(24); T(87)]

markers = Vector{Vector{T}}(undef, 2)
markers[1] = mk_red
markers[2] = mk_cyan

marker_colours = Vector{String}(undef, 2)
marker_colours[1] = "cyan"
marker_colours[2] = "cyan"

warp_camp = "bwr"
mk_symbols = ["D"; "*"]

# Load hyperparameters
load_path = joinpath("results", "timing_hp_single_downfactor_$(down_factor)")
down_factor2,
    dek_vars, dek_score, sk_vars, sk_score,
    dek1_vars, dek1_score, sk1_vars, sk1_score, = deserialize(load_path)
#

y = vec(im_y)
X = vec(collect(collect(x) for x in Iterators.product(Xrs...)))

# ## Stationary kernel
sk1 = LGP.SqExpKernel(sk1_vars[begin])
gp_sk1 = LGP.fitGP(X, y, σ², sk1)

# ## DE kernel
kernel_param1, κ1 = dek1_vars[begin], dek1_vars[end]
# κ1 = 9999.9 # debug TODO
canonical_kernel1 = LGP.WendlandSplineKernel(
    LGP.Order2(), kernel_param1, 3,
)
dek1 = LGP.DEKernel(canonical_kernel1, warpmap, κ1)
gp_dek1 = LGP.fitGP(X, y, σ², dek1)

# kernel functions

function expand_dim(x, dek)
    return [x; dek.κ * dek.warpmap(x)]
end

k_sk_red = xx -> LGP.evalkernel(
    Distances.evaluate(LGP.getmetric(sk1), xx, mk_red),
    sk1,
)
k_dek_red = xx -> LGP.evalkernel(
    Distances.evaluate(LGP.getmetric(dek1), expand_dim(xx, dek1), expand_dim(mk_red, dek1)),
    dek1,
)

k_sk_cyan = xx -> LGP.evalkernel(
    Distances.evaluate(LGP.getmetric(sk1), xx, mk_cyan),
    sk1,
)
k_dek_cyan = xx -> LGP.evalkernel(
    Distances.evaluate(LGP.getmetric(dek1), expand_dim(xx, dek1), expand_dim(mk_cyan, dek1)),
    dek1,
)

# ### The red marker's region of interest
ROI_half_size = T(3)

x1_lb_red = mk_red[1] - ROI_half_size
x1_ub_red = mk_red[1] + ROI_half_size
x2_lb_red = mk_red[2] - ROI_half_size
x2_ub_red = mk_red[2] + ROI_half_size
Nq1 = 50
Nq2 = 50
Xqrs_red = (
    LinRange(x1_lb_red, x1_ub_red, Nq1),
    LinRange(x2_lb_red, x2_ub_red, Nq2),
)
Xqs_red = collect(collect(x) for x in Iterators.product(Xqrs_red...))

# the data restricted to this region.
imy_Xq_red = im_y[floor(Int, x1_lb_red):floor(Int, x1_ub_red), floor(Int, x2_lb_red):floor(Int, x2_ub_red)]

# The kernel evaluations
kq_sk_red = k_sk_red.(Xqs_red)
kq_dek_red = k_dek_red.(Xqs_red)

# warp map evaluations
w_Xq_red = dek1.warpmap.(Xqs_red)

# up-conversion result for this region.
println("Querying SK, red region")
@time query_sk_red = collect(
    LGP.queryGP(collect(x), sk1, gp_sk1)
        for x in Iterators.product(Xqrs_red...)
)
mqs_sk_red = map(xx -> xx[1], query_sk_red)
vqs_sk_red = map(xx -> xx[2], query_sk_red)

println("Querying DEK, red region")
@time query_dek_red = collect(
    LGP.queryGP(collect(x), dek1, gp_dek1)
        for x in Iterators.product(Xqrs_red...)
)
mqs_dek_red = map(xx -> xx[1], query_dek_red)
vqs_dek_red = map(xx -> xx[2], query_dek_red)
# 90 sec each.
serialize(joinpath("results", "region_red_upconvert"), (mqs_sk_red, vqs_sk_red, mqs_dek_red, vqs_dek_red))

mqs_sk_red, vqs_sk_red, mqs_dek_red, vqs_dek_red = deserialize(joinpath("results", "region_red_upconvert"))

# ### The cyan marker's region of interest
ROI_half_size = T(3)

x1_lb_cyan = mk_cyan[1] - ROI_half_size
x1_ub_cyan = mk_cyan[1] + ROI_half_size
x2_lb_cyan = mk_cyan[2] - ROI_half_size
x2_ub_cyan = mk_cyan[2] + ROI_half_size
Nq1 = 50
Nq2 = 50
Xqrs_cyan = (
    LinRange(x1_lb_cyan, x1_ub_cyan, Nq1),
    LinRange(x2_lb_cyan, x2_ub_cyan, Nq2),
)
Xqs_cyan = collect(collect(x) for x in Iterators.product(Xqrs_cyan...))

# the data restricted to this region.
imy_Xq_cyan = im_y[floor(Int, x1_lb_cyan):floor(Int, x1_ub_cyan), floor(Int, x2_lb_cyan):floor(Int, x2_ub_cyan)]

# The kernel evaluations
kq_sk_cyan = k_sk_cyan.(Xqs_cyan)
kq_dek_cyan = k_dek_cyan.(Xqs_cyan)

# warp map evaluations
w_Xq_cyan = dek1.warpmap.(Xqs_cyan)

# up-conversion result for this region.
println("Querying SK, cyan region")
@time query_sk_cyan = collect(
    LGP.queryGP(collect(x), sk1, gp_sk1)
        for x in Iterators.product(Xqrs_cyan...)
)
mqs_sk_cyan = map(xx -> xx[1], query_sk_cyan)
vqs_sk_cyan = map(xx -> xx[2], query_sk_cyan)

println("Querying DEK, cyan region")
@time query_dek_cyan = collect(
    LGP.queryGP(collect(x), dek1, gp_dek1)
        for x in Iterators.product(Xqrs_cyan...)
)
mqs_dek_cyan = map(xx -> xx[1], query_dek_cyan)
vqs_dek_cyan = map(xx -> xx[2], query_dek_cyan)
# 90 sec each.
serialize(joinpath("results", "region_cyan_upconvert"), (mqs_sk_cyan, vqs_sk_cyan, mqs_dek_cyan, vqs_dek_cyan))

mqs_sk_cyan, vqs_sk_cyan, mqs_dek_cyan, vqs_dek_cyan = deserialize(joinpath("results", "region_cyan_upconvert"))

# # Visualize

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

save_dpi = 150

PLT.figure(fig_num, figsize = (8, 4), dpi = 150)
fig_num += 1

PLT.subplot(121)
plot_subfig(
    PLT,
    collect(Xrs),
    im_y,
    marker_locations = markers,
    marker_symbols = mk_symbols,
    marker_colors = marker_colours,
    # cmap = var_cmap,
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = true,
    color_bar_params = (0.04, 0.04)
)
#PLT.axis("off")
PLT.title("Data")

PLT.subplot(122)
plot_subfig(
    PLT,
    collect(Xrs),
    W,
    marker_locations = markers,
    marker_symbols = mk_symbols,
    marker_colors = marker_colours,
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    symmetric_color_range = true,
    cmap = warp_camp,
)
#PLT.axis("off")
PLT.title("Warp map")
PLT.subplots_adjust(
    left = 0.06, right = 1 - 0.06, top = 1 - 0.01, bottom = 0.01,
    wspace = 0.2, hspace = 0.01,
)
PLT.savefig(joinpath("figs", "manuscript", "kernel_markers.png"), dpi = 96)

# ## Kernel plots

z = max(abs(minimum(W)), abs(maximum(W)))
vmin_W = -z
vmax_W = z

PLT.figure(fig_num, figsize = (48, 39), dpi = 150)
#PLT.figure(fig_num)
fig_num += 1

X2_red = (
    LinRange(x1_lb_red, x1_ub_red, size(imy_Xq_red, 1)),
    LinRange(x2_lb_red, x2_ub_red, size(imy_Xq_red, 2)),
)
PLT.subplot2grid((3, 4), (0, 0))
plot_subfig(
    PLT,
    collect(X2_red),
    imy_Xq_red,
    marker_locations = markers[1:1],
    marker_symbols = mk_symbols[1:1],
    marker_colors = marker_colours[1:1],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
)
PLT.axis("off")
PLT.title("Data")

PLT.subplot2grid((3, 4), (0, 1))
plot_subfig(
    PLT,
    collect(Xqrs_red),
    w_Xq_red,
    marker_locations = markers[1:1],
    marker_symbols = mk_symbols[1:1],
    marker_colors = marker_colours[1:1],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    symmetric_color_range = false,
    cmap = warp_camp,
    vmin = vmin_W,
    vmax = vmax_W,
)
PLT.axis("off")
PLT.title("Warp map")

X2_cyan = (
    LinRange(x1_lb_cyan, x1_ub_cyan, size(imy_Xq_cyan, 1)),
    LinRange(x2_lb_cyan, x2_ub_cyan, size(imy_Xq_cyan, 2)),
)
PLT.subplot2grid((3, 4), (0, 2))
plot_subfig(
    PLT,
    collect(X2_cyan),
    imy_Xq_cyan,
    marker_locations = markers[2:2],
    marker_symbols = mk_symbols[2:2],
    marker_colors = marker_colours[2:2],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
)
PLT.axis("off")
PLT.title("Data")

PLT.subplot2grid((3, 4), (0, 3))
plot_subfig(
    PLT,
    collect(Xqrs_cyan),
    w_Xq_cyan,
    marker_locations = markers[2:2],
    marker_symbols = mk_symbols[2:2],
    marker_colors = marker_colours[2:2],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    symmetric_color_range = false,
    cmap = warp_camp,
    vmin = vmin_W,
    vmax = vmax_W,
)
PLT.axis("off")
PLT.title("Warp map")

PLT.subplot2grid((3, 4), (1, 0))
plot_subfig(
    PLT,
    collect(Xqrs_red),
    mqs_sk_red,
    marker_locations = markers[1:1],
    marker_symbols = mk_symbols[1:1],
    marker_colors = marker_colours[1:1],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T), vmax = one(T),
)
PLT.axis("off")
PLT.title("SK posterior mean")

PLT.subplot2grid((3, 4), (1, 1))
plot_subfig(
    PLT,
    collect(Xqrs_red),
    mqs_dek_red,
    marker_locations = markers[1:1],
    marker_symbols = mk_symbols[1:1],
    marker_colors = marker_colours[1:1],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T), vmax = one(T),
)
PLT.axis("off")
PLT.title("DEK posterior mean")

PLT.subplot2grid((3, 4), (1, 2))
plot_subfig(
    PLT,
    collect(Xqrs_cyan),
    mqs_sk_cyan,
    marker_locations = markers[2:2],
    marker_symbols = mk_symbols[2:2],
    marker_colors = marker_colours[2:2],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T), vmax = one(T),
)
PLT.axis("off")
PLT.title("SK posterior mean")

PLT.subplot2grid((3, 4), (1, 3))
plot_subfig(
    PLT,
    collect(Xqrs_cyan),
    mqs_dek_cyan,
    marker_locations = markers[2:2],
    marker_symbols = mk_symbols[2:2],
    marker_colors = marker_colours[2:2],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T), vmax = one(T),
)
PLT.axis("off")
PLT.title("DEK posterior mean")


PLT.subplot2grid((3, 4), (2, 0))
plot_subfig(
    PLT,
    collect(Xqrs_red),
    kq_sk_red,
    marker_locations = markers[1:1],
    marker_symbols = mk_symbols[1:1],
    marker_colors = marker_colours[1:1],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T),
    vmax = one(T),
    cmap = "viridis",
)
PLT.axis("off")
PLT.title("SK centered kernel")

PLT.subplot2grid((3, 4), (2, 1))
plot_subfig(
    PLT,
    collect(Xqrs_red),
    kq_dek_red,
    marker_locations = markers[1:1],
    marker_symbols = mk_symbols[1:1],
    marker_colors = marker_colours[1:1],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T),
    vmax = one(T),
    cmap = "viridis",
)
PLT.axis("off")
PLT.title("DEK centered kernel")

PLT.subplot2grid((3, 4), (2, 2))
plot_subfig(
    PLT,
    collect(Xqrs_cyan),
    kq_sk_cyan,
    marker_locations = markers[2:2],
    marker_symbols = mk_symbols[2:2],
    marker_colors = marker_colours[2:2],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T),
    vmax = one(T),
    cmap = "viridis",
)
PLT.axis("off")
PLT.title("SK centered kernel")

PLT.subplot2grid((3, 4), (2, 3))
plot_subfig(
    PLT,
    collect(Xqrs_cyan),
    kq_dek_cyan,
    marker_locations = markers[2:2],
    marker_symbols = mk_symbols[2:2],
    marker_colors = marker_colours[2:2],
    matrix_mode = true,
    display_color_bar = true,
    color_bar_params = (0.04, 0.04),
    vmin = zero(T),
    vmax = one(T),
    cmap = "viridis",
)
PLT.axis("off")
PLT.title("DEK centered kernel")

PLT.subplots_adjust(
    left = 0.04, right = 1 - 0.1, top = 1 - 0.1, bottom = 0.03,
    wspace = 0.15, hspace = 0.3,
)
#PLT.savefig(joinpath("figs", "manuscript", "kernel_evals.png"), dpi = save_dpi)
# save manually from the figure for more control.

nothing
