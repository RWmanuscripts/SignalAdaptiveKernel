using Pkg
Pkg.activate(".")

# # Visualize

import Images
using Serialization

import VisualizationBag as VIZ
import PythonPlot as PLT
fig_num = 1
PLT.close("all")

T = Float64

reference_image_folder = "data/images/kodak/"

down_factor = 2
#down_factor = 4

results_dir = "results/kodak/down$(down_factor)/spline32/"


r_select = 4

# # # Parrot face.
# #Xrs = (147:362, 59:317) # too large and far.
# Xrs = (180:320, 120:300) # close up, eye, neck.
# scene = "kodim23"

# Helmet.
Xrs = (30:190, 30:190) # close up, eye, neck.
scene = "kodim05"

# # Hat.
# Xrs = (14:242, 162:372) # close up, eye, neck.
# scene = "kodim04"

ms_string = "ML"

#####

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
v_dek = vqs_dek_set[r_select][Xrs...]
v_ck = vqs_sk[Xrs...]
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
    "Oracle image";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)

# DEK - mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_dek,
    [],
    "x",
    fig_num,
    "DEK - mean";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)

# bi-cubic interpolation.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    itp,
    [],
    "x",
    fig_num,
    "Interpolation";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)

# CK mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    m_ck,
    [],
    "x",
    fig_num,
    "CK - mean";
    cmap = "gray",
    #vmin = zero(T),
    #vmax = one(T),
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
)


# DEK variance.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    v_dek,
    [],
    "x",
    fig_num,
    "DEK - variance";
    cmap = "gray",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.6,
)

# CK variance.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    v_ck,
    [],
    "x",
    fig_num,
    "CK - variance";
    cmap = "gray",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.6,
)

nothing
