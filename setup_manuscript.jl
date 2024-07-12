
# First, navigate to the julia_scripts folder, then run in Bash shell:
# bash hopt.sh
# bash kodak.sh
# Then navigate to this directory and run this script in Julia REPL.

do_not_display_figures = true # set to false to have PythonPlot.jl show figures when you run this script.

using Pkg
Pkg.activate(".")
Pkg.Registry.add(RegistrySpec(url = "https://github.com/RoyCCWang/RWPublicJuliaRegistry"))
let
    pkgs = ["JSON3", "LazyGPR", "SpatialGSP", "Interpolations", "ScatteredInterpolation", "PythonPlot", "Revise", "IJulia", "Images", "RieszDSP", "LocalFilters", "VisualizationBag", "StaticArrays",
    "Markdown", "Tables", "MarkdownTables", "FileIO"]
    for pkg in pkgs
        if Base.find_package(pkg) === nothing
            Pkg.add(pkg)
        end
    end
end

import Interpolations
import ScatteredInterpolation as SI
import PythonPlot as PLT
import Images
import RieszDSP as RZ
import LocalFilters
using StaticArrays

import VisualizationBag as VIZ
fig_num = 1
PLT.close("all") # close all existing figures.

import LazyGPR as LGP
import SpatialGSP as GSP

using Markdown, Tables, MarkdownTables, FileIO, Serialization, DelimitedFiles

include("helpers/image.jl");

T = Float64;

# All generated files from this script goes in this folder:
save_folder = "results/manuscript_tmp_files/"
if !ispath(save_folder)
    mkpath(save_folder)
end

# # Set up for figures for one-hop operators

# user inputs.
pics_folder = "data/images/kodak"
image_path = joinpath(pics_folder, "kodim23.png")

img = loadkodakimage(T, image_path; discard_pixels = 1)
x_nD, x_ranges = image2samples(img)
sz_x = size(x_nD)

# name image data.
Xrs = (1:size(x_nD,1), 1:size(x_nD,2))
im_y = x_nD
x = vec(x_nD)

# # Graph laplacian
nbs = GSP.getgridnbs(size(x_nD))
G = GSP.UnitGrid(nbs)

graph = GSP.getgraph(G)
A = GSP.create_adjacency(G)
deg = GSP.create_degree(G)
deg_inv = GSP.create_invdegree(T, G)
L = GSP.create_laplacian(G)

# normalized Laplacian.
Ln = GSP.create_snlaplacian(T, G)
TL = GSP.create_rwlaplacian(T, G)

# apply operators to signal.
Ax = A*x # adjacency
Anx_pre = deg_inv*A*x
Anx_post = A*deg_inv*x
Lx = L*x # combinatorial Laplacian.
Qx = deg\(A*x) # random-walk operator, inv(deg)*A*x.
Tx = x - Qx # random-walk Laplacian, T = I - Q.
Ln_x = Ln*x

# Riesz warp samples.
W_rz = LGP.create_grid_warp_samples(LGP.UseRieszDSP(RZ), im_y)

# segment border
function getcloseup(Xrs, Y, w, h)
    # Y_out = Y[end-h:end, 220:220+w]
    # Xrs_out = (Xrs[begin][end-h:end], Xrs[end][220:220+w])
    Y_out = Y[begin:begin+h, begin:begin+w]
    Xrs_out = (Xrs[begin][begin:begin+h], Xrs[end][begin:begin+w])
    return Xrs_out, Y_out
end

width = 20
height = 10
Xrs_close, x_close = getcloseup(Xrs, x_nD, width, height)
_, Ax_close = getcloseup(Xrs, reshape(Ax, sz_x), width, height)
_, Anx_pre_close = getcloseup(Xrs, reshape(Anx_pre, sz_x), width, height)
_, Anx_post_close = getcloseup(Xrs, reshape(Anx_post, sz_x), width, height)
_, Lx_close = getcloseup(Xrs, reshape(Lx, sz_x), width, height)
_, Lnx_close = getcloseup(Xrs, reshape(Ln_x, sz_x), width, height)
_, Tx_close = getcloseup(Xrs, reshape(Tx, sz_x), width, height)
_, W_rz_close = getcloseup(Xrs, W_rz, width, height)

# # Figures for one-hop operators
fig_size = VIZ.getaspectratio(size(im_y)) .* 4
dpi = 300
#dpi = 96



fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    x_close,
    [],
    "x",
    fig_num,
    "Image data, x";
    cmap = "gray",
    vmin = 0,
    vmax = 1,
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/x_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Ax_close,
    [],
    "x",
    fig_num,
    "Ax";
    cmap = "gray",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    vmin = 0,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/Ax_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Anx_post_close,
    [],
    "x",
    fig_num,
    "AD⁻¹x";
    cmap = "gray",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    vmin = 0,
    vmax = 1,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/Anx_post_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Anx_pre_close,
    [],
    "x",
    fig_num,
    "D⁻¹Ax";
    cmap = "gray",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    vmin = 0,
    vmax = 1,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/Anx_pre_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    W_rz_close,
    [],
    "x",
    fig_num,
    "Warp samples via HWRT";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    symmetric_color_range = true,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/W_rz_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Lx_close,
    [],
    "x",
    fig_num,
    "Lx";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    symmetric_color_range = true,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/Lx_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Lnx_close,
    [],
    "x",
    fig_num,
    "ℒx";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    symmetric_color_range = true,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/Lnx_close.png")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Tx_close,
    [],
    "x",
    fig_num,
    "Tx";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    color_bar_shrink = 0.6,
    symmetric_color_range = true,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/Tx_close.png")

# # Bernstein basis figure

include("helpers/misc.jl")

Bernstein_degree = 5
fig_num = visualizebernsteinbasis(
    one(T), Bernstein_degree, fig_num;
    fig_size = (5,2), dpi = 300, title_string = "Bernstein polynomial basis of degree $Bernstein_degree"
)
PLT.subplots_adjust(right=0.75) # need to do this otherwise PLT.savefig() cuts off the  legend text.
do_not_display_figures || PLT.show()
PLT.savefig("figs/bernstein_basis.png", dpi = 300, bbox_inches = "tight")

# # S-map figure

fig_size = (5, 1)
dpi = 300

# ## Visualize RQ curve
b_x = convert(T, 5)
M = floor(Int, b_x)
L = M # must be an even positive integer. The larger the flatter.
if isodd(L)
    L = L + 1
end
x0, y0 = convert(T, 0.8*M), 1 + convert(T, 0.5)
s = LGP.AdjustmentMap(x0, y0, b_x, L)

# plot.
viz_bound = 0.9
u_range = LinRange(0, viz_bound*M, 1000)
g = uu->LGP.evalsmap(uu, s)

#@assert abs(g(x0)-y0) < eps(T)*100
#@show s, propertynames(s)

PLT.figure(fig_num; figsize = fig_size, dpi = dpi)
fig_num += 1
PLT.plot(u_range, g.(u_range))
PLT.xlabel("τ")
PLT.ylabel("ξ(τ)")
#PLT.title("Adjustment map")
do_not_display_figures || PLT.show()
PLT.savefig("figs/s_map.png", dpi = dpi, bbox_inches = "tight")


# # fig-warp-compare
# user inputs.
D = 2
dpi = 300

img = loadkodakimage(T, image_path; discard_pixels = 1)
x_nD, x_ranges = image2samples(img)
sz_x = size(x_nD)

# name image data.
Xrs = (1:size(x_nD,1), 1:size(x_nD,2))
im_y = x_nD

# Warp samples from RieszDSP.jl
W_rz = LGP.create_grid_warp_samples(LGP.UseRieszDSP(RZ), im_y)
sigma_r_factor = 4
σr = maximum(abs.(W_rz))*sigma_r_factor
σs = 1.0
if σr > 0 && σs > 0
    W_rz = LocalFilters.bilateralfilter(
        W_rz, σr, σs, 2*round(Int,3*σs)+1,
    )
end

# Bernstein filtering, k-nn graph.
knn_config = GSP.KNNConfig{T}(k = 2*D)
warp_config = GSP.WarpConfig{T}(aggregate_option = :sum)
X = collect(
    SVector{D, T}(x) for x in Iterators.product(Xrs...)
)
W_knn_vec = LGP.create_warp_samples(
    LGP.UseSpatialGSP(GSP),
    LGP.KNNGraph(),
    vec(X), vec(im_y), knn_config, warp_config,
)
W_knn = reshape(W_knn_vec, size(im_y))
σr = maximum(abs.(W_knn))*sigma_r_factor
if σr > 0 && σs > 0
    W_knn = LocalFilters.bilateralfilter(
        W_knn, σr, σs, 2*round(Int,3*σs)+1,
    )
end

# Bernstein filtering, grid graph.
W_grid = LGP.create_grid_warp_samples(
    LGP.UseSpatialGSP(GSP),
    im_y,
    warp_config,
)
if σr > 0 && σs > 0
    W_grid = LocalFilters.bilateralfilter(
        W_grid, σr, σs, 2*round(Int,3*σs)+1,
    )
end

# # Visualize
fig_size = VIZ.getaspectratio(size(im_y)) .* 5.5
#@show fig_size

dpi = 300
#dpi = 96

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    im_y,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = 0,
    vmax = 1,
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
    fig_size =fig_size = fig_size,
    dpi = dpi,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/fig-warp-compare-image_data.png", dpi = dpi, bbox_inches = "tight")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    W_rz,
    [],
    "x",
    fig_num,
    "";
    cmap = "bwr",
    symmetric_color_range = true,
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
    fig_size = fig_size,
    dpi = dpi,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/fig-warp-compare-HRWT.png", dpi = dpi, bbox_inches = "tight")

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    W_knn,
    [],
    "x",
    fig_num,
    "";
    cmap = "bwr",
    symmetric_color_range = true,
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
    fig_size = fig_size,
    dpi = dpi,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/fig-warp-compare-bernstein_filtering.png", dpi = dpi, bbox_inches = "tight")

# # Table for the rainfall dataset

tag = "ML"

sig_digits = 3
using Serialization

# # Load query results.
load_results_dir = "results/"
if !ispath(load_results_dir)
    mkpath(load_results_dir)
end

X, Xqrs, y, mqs_sk, vqs_sk, mqs_ck, vqs_ck, mqs_dek, vqs_dek = deserialize(
    joinpath(load_results_dir, "CA_rainfall_queries_$(tag)"),
)

sig_digits = 3

# # Load query results.
load_results_dir = "results/"
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
a2s = aa->sqrt(1/(2*aa)) # a := 1/(2*s^2), solve for s.
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
print(markdown_table(Tables.table(
    data_mat;
    header = [
        "";
        "DE kernel";
        "Stationary kernel";
        "Canonical kernel";
        "Rainfall data";
    ]
), String))

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
do_not_display_figures || PLT.show()
PLT.savefig("figs/rainfall_DE_mean.png", dpi = dpi, bbox_inches = "tight")

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
do_not_display_figures || PLT.show()
PLT.savefig("figs/rainfall_DE_variance.png", dpi = dpi, bbox_inches = "tight")

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
do_not_display_figures || PLT.show()
PLT.savefig("figs/rainfall_sqexp_mean.png", dpi = dpi, bbox_inches = "tight")

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
do_not_display_figures || PLT.show()
PLT.savefig("figs/rainfall_sqexp_variance.png", dpi = dpi, bbox_inches = "tight")

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
do_not_display_figures || PLT.show()
PLT.savefig("figs/rainfall_canonical_mean.png", dpi = dpi, bbox_inches = "tight")

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
do_not_display_figures || PLT.show()
PLT.savefig("figs/rainfall_canonical_variance.png", dpi = dpi, bbox_inches = "tight")

# # Hat figure


# # Visualize

reference_image_folder = "data/images/kodak/"
results_dir = "results/kodak/down2/spline32/"
r_select = 4
ms_string = "ML"

# Hat.
Xrs = (14:242, 162:372) # close up, eye, neck.
scene = "kodim04"
Xrs_hat = Xrs

load_path = joinpath(
    results_dir, "upconvert_$(scene)_$(ms_string)"
)
(
    itp_Xq, mqs_sk, vqs_sk,
    mqs_dek_set, vqs_dek_set,
    dek_vars_set, dek_score_set,
    sk_vars_set, sk_score_set, κ_ub,
) = deserialize(load_path)

mqs_sk = reshape(mqs_sk, size(itp_Xq))
vqs_sk = reshape(vqs_sk, size(itp_Xq))

# means
m_dek = mqs_dek_set[r_select][Xrs...]
m_ck = mqs_sk[Xrs...]
v_dek_hat = vqs_dek_set[r_select][Xrs...]
v_ck_hat = vqs_sk[Xrs...]
itp = itp_Xq[Xrs...]

# oracle
file_path = joinpath(reference_image_folder, "$(scene).png")
img = Images.load(file_path)
gray_img = Images.Gray.(img)
y_nD = convert(Array{T}, gray_img)
im_y = y_nD[Xrs...]

# oracle
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    im_y,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/hat_reference.png", dpi = dpi, bbox_inches = "tight")

# DEK - mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_dek,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/hat_DE_mean.png", dpi = dpi, bbox_inches = "tight")

# bi-cubic interpolation.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    itp,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/hat_spline_itp.png", dpi = dpi, bbox_inches = "tight")

# CK mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_ck,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/hat_canonical_mean.png", dpi = dpi, bbox_inches = "tight")

# # Parrot figure

# # Visualize

reference_image_folder = "data/images/kodak/"
results_dir = "results/kodak/down2/spline32/"
r_select = 4
ms_string = "ML"

# Parrot face.
Xrs = (180:320, 120:300) # close up, eye, neck.
scene = "kodim23"
Xrs_bird = Xrs

load_path = joinpath(
    results_dir, "upconvert_$(scene)_$(ms_string)"
)
(
    itp_Xq, mqs_sk, vqs_sk,
    mqs_dek_set, vqs_dek_set,
    dek_vars_set, dek_score_set,
    sk_vars_set, sk_score_set, κ_ub,
) = deserialize(load_path)

mqs_sk = reshape(mqs_sk, size(itp_Xq))
vqs_sk = reshape(vqs_sk, size(itp_Xq))

# means
m_dek = mqs_dek_set[r_select][Xrs...]
m_ck = mqs_sk[Xrs...]
v_dek_bird = vqs_dek_set[r_select][Xrs...]
v_ck_bird = vqs_sk[Xrs...]
itp = itp_Xq[Xrs...]

# oracle
file_path = joinpath(reference_image_folder, "$(scene).png")
img = Images.load(file_path)
gray_img = Images.Gray.(img)
y_nD = convert(Array{T}, gray_img)
im_y = y_nD[Xrs...]

# oracle
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    im_y,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/parrot_reference.png", dpi = dpi, bbox_inches = "tight")

# DEK - mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_dek,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/parrot_DE_mean.png", dpi = dpi, bbox_inches = "tight")

# bi-cubic interpolation.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    itp,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/parrot_spline_itp.png", dpi = dpi, bbox_inches = "tight")

# CK mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_ck,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/parrot_canonical_mean.png", dpi = dpi, bbox_inches = "tight")

# # Helmet figure

# # Visualize

reference_image_folder = "data/images/kodak/"
results_dir = "results/kodak/down2/spline32/"
r_select = 4
ms_string = "ML"

# Helmet.
Xrs = (30:190, 30:190) # close up, eye, neck.
scene = "kodim05"
Xrs_helmet = Xrs

load_path = joinpath(
    results_dir, "upconvert_$(scene)_$(ms_string)"
)
(
    itp_Xq, mqs_sk, vqs_sk,
    mqs_dek_set, vqs_dek_set,
    dek_vars_set, dek_score_set,
    sk_vars_set, sk_score_set, κ_ub,
) = deserialize(load_path)

mqs_sk = reshape(mqs_sk, size(itp_Xq))
vqs_sk = reshape(vqs_sk, size(itp_Xq))

# means
m_dek = mqs_dek_set[r_select][Xrs...]
m_ck = mqs_sk[Xrs...]
v_dek_helmet = vqs_dek_set[r_select][Xrs...]
v_ck_helmet = vqs_sk[Xrs...]
itp = itp_Xq[Xrs...]

# oracle
file_path = joinpath(reference_image_folder, "$(scene).png")
img = Images.load(file_path)
gray_img = Images.Gray.(img)
y_nD = convert(Array{T}, gray_img)
im_y = y_nD[Xrs...]

# oracle
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    im_y,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/helmet_reference.png", dpi = dpi, bbox_inches = "tight")

# DEK - mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_dek,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/helmet_DE_mean.png", dpi = dpi, bbox_inches = "tight")

# bi-cubic interpolation.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    itp,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/helmet_spline_itp.png", dpi = dpi, bbox_inches = "tight")

# CK mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_ck,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    vmin = zero(T),
    vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/helmet_canonical_mean.png", dpi = dpi, bbox_inches = "tight")



# # Table - Kodak dataset, hyperparameters


include("helpers/kodak_tables.jl")
rs = 1:6
down_factor = 2

# run kodak_results.jl first.
hps, SSIMs_itp, SSIMs_sk, SSIMs_deks = deserialize("results/kodak_down2_ML")
hp_mat = generate_hp_table(hps; sig_digits = 3)

# save to CSV.
writedlm( "tables/kodak_hp.csv",  hp_mat, ',')

scene_names = get_scene_names()

DEK_hp_header = collect(
    "Gain, r: $r" for r in rs
)
print(markdown_table(Tables.table(
    [scene_names hp_mat];
    header = [
        "Scene"; "Scale"; DEK_hp_header;
    ]
), String))

# # table SSIM

scene_names = get_scene_names()

SSIM_mat = generate_SSIM_table(
    SSIMs_itp, SSIMs_sk, SSIMs_deks;
    sig_digits = 5
)
writedlm( "tables/ssim_itp.csv",  SSIMs_itp, ',')
writedlm( "tables/ssim_sk.csv",  SSIMs_sk, ',')
writedlm( "tables/ssim_de.csv",  collect(Iterators.flatten(SSIMs_deks)), ',') # `SSIMs_deks` is a nested Vector. Flatten to 1-D array before storing as CSV.

DEK_r_header = collect(
    "DE r: $r" for r in rs
)

data_mat = [
    scene_names SSIM_mat
];

print(markdown_table(Tables.table(
    data_mat;
    header = [
        "Scene"; "Bi-cubic"; "CK"; DEK_r_header;
    ]
), String))

# # Figure - Kodak images variance

# DEK variance. hat.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_hat),
    v_dek_hat,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/variance_hat_DE.png", dpi = dpi, bbox_inches = "tight")

# CK variance. hat.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_hat),
    v_ck_hat,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/variance_hat_canonical.png", dpi = dpi, bbox_inches = "tight")

# DEK variance. bird.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_bird),
    v_dek_bird,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/variance_parrot_DE.png", dpi = dpi, bbox_inches = "tight")

# CK variance. bird.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_bird),
    v_ck_bird,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/variance_parrot_canonical.png", dpi = dpi, bbox_inches = "tight")

# DEK variance. helmet.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_helmet),
    v_dek_helmet,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/variance_helmet_DE.png", dpi = dpi, bbox_inches = "tight")

# CK variance. helmet.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_helmet),
    v_ck_helmet,
    [],
    "x",
    fig_num,
    "";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
)
do_not_display_figures || PLT.show()
PLT.savefig("figs/variance_helmet_canonical.png", dpi = dpi, bbox_inches = "tight")

nothing