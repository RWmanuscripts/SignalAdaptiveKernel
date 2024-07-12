
import Random
Random.seed!(25)

using Serialization

tag = "ML"

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
    "Minimum value" min_dek min_sk min_ck y_min;
    "Maximum value" max_dek max_sk max_ck y_max;
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


# visualize.

import VisualizationBag as VIZ
import PythonPlot as PLT
fig_num = 1
PLT.close("all")


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
PLT.show()

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
PLT.show()

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
PLT.show()

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
PLT.show()

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
PLT.show()

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
PLT.show()

nothing