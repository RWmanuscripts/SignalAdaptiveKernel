save_dpi = 300

PLT.figure(fig_num, figsize = (7, 11), dpi = 150)
fig_num += 1

PLT.subplot2grid((4, 2), (0, 0))
plot_subfig(
    PLT,
    collect(Xrs_bird),
    im_y_bird,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Reference")

PLT.subplot2grid((4, 2), (0, 1))
plot_subfig(
    PLT,
    collect(Xrs_bird),
    m_dek_bird,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Lazy-evaluation with DEK")

PLT.subplot2grid((4, 2), (1, 0))
plot_subfig(
    PLT,
    collect(Xrs_bird),
    itp_bird,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Spline")

PLT.subplot2grid((4, 2), (1, 1))
plot_subfig(
    PLT,
    collect(Xrs_bird),
    m_ck_bird,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Lazy-evaluation with CK")


PLT.subplot2grid((4, 2), (2, 0))
plot_subfig(
    PLT,
    collect(Xrs_helmet),
    im_y_helmet,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Reference")

PLT.subplot2grid((4, 2), (2, 1))
plot_subfig(
    PLT,
    collect(Xrs_helmet),
    m_dek_helmet,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Lazy-evaluation with DEK")

PLT.subplot2grid((4, 2), (3, 0))
plot_subfig(
    PLT,
    collect(Xrs_helmet),
    itp_helmet,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Spline")

PLT.subplot2grid((4, 2), (3, 1))
plot_subfig(
    PLT,
    collect(Xrs_helmet),
    m_ck_helmet,
    marker_locations = [],
    matrix_mode = true,
    vmin = zero(T),
    vmax = one(T),
    display_color_bar = false,
)
PLT.axis("off")
PLT.title("Lazy-evaluation with CK")

PLT.subplots_adjust(
    left = 0.04, right = 1 - 0.04, top = 1 - 0.04, bottom = 0.04,
    wspace = 0.1, hspace = 0.1,
)
PLT.savefig(
    joinpath("figs", "manuscript", "fig-kodak.png"),
    dpi = save_dpi,
    bbox_inches = "tight",
)

nothing
