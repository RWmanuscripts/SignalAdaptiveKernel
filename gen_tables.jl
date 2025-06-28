# # Generate latex table

using Printf, SummaryTables, Serialization, LinearAlgebra, tectonic_jll

# original image dimension is 209 x 189, which is written as (209, 189) by the notations used for the tables.
# downsample by 2: (95, 105)
# by 4: (48, 53)
# by 6: (32, 35)

# # Hyperparameter optimization timing
# b_x = 6.
categories = [
    "Conventional",
    "Conventional",
    "Lazy 25",
    "Lazy 25",
    "Lazy 50",
    "Lazy 50",
    "Lazy 100",
    "Lazy 100",
]
headings = ["SK", "DEK", "SK", "DEK", "SK", "DEK", "SK", "DEK"]

# units: seconds.
times_data = [
    "1142.7" "1458.6" "1.5" "1.5" "2.6" "2.6" "4.9" "5"
    "44.7" "57.2" "1.3" "1.4" "2.3" "2.5" "2.9" "3"
    "6.7" "9.3" "1.3" "1.4" "1.5" "1.5" "1.5" "1.5"
]

size_col = ["(95, 105)" ; "(48, 53)"; "(32, 35)"]

body = [
    hcat(
        Cell(nothing),
        Cell.(categories, bold = true, merge = true, border_bottom = true)',
    )
    hcat(Cell("Size"), Cell.(headings)')
    hcat(
        Cell.(size_col),
        Cell.(times_data)
    )
]

hp_table = Table(body, header = 2)


dir = joinpath("results")
texfile = joinpath(dir, "hp_table.tex")

open(texfile, "w") do io
    # add the necessary packages in the preamble
    println(
        io, raw"""
            \documentclass{article}
            \usepackage{threeparttable}
            \usepackage{multirow}
            \usepackage{booktabs}
            \begin{document}
        """
    )

    # print the table as latex code
    show(io, MIME"text/latex"(), hp_table)

    println(io, raw"\end{document}")
end

# render the tex file to pdf
tectonic_jll.tectonic() do bin
    run(`$bin --chatter=minimal $texfile`)
end


# # Lazy query timing

# mention setup time was 428 us for downsample by 2.

sk_both6, sk_mean6, sk_var6, dek_cache6, dek_both6, dek_mean6, dek_var6 = deserialize(joinpath("results", "time_query_lazy_bx_6.0"))

sk_both8, sk_mean8, sk_var8, dek_cache8, dek_both8, dek_mean8, dek_var8 = deserialize(joinpath("results", "time_query_lazy_bx_8.0"))

sk_both10, sk_mean10, sk_var10, dek_cache10, dek_both10, dek_mean10, dek_var10 = deserialize(joinpath("results", "time_query_lazy_bx_10.0"))

sk_both12, sk_mean12, sk_var12, dek_cache12, dek_both12, dek_mean12, dek_var12 = deserialize(joinpath("results", "time_query_lazy_bx_12.0"))

sk_both14, sk_mean14, sk_var14, dek_cache14, dek_both14, dek_mean14, dek_var14 = deserialize(joinpath("results", "time_query_lazy_bx_14.0"))

function to_table_string(x)
    return @sprintf("%.1f", x)
end

categories = [
    "Radius 6",
    "Radius 6",
    "Radius 8",
    "Radius 8",
    "Radius 10",
    "Radius 10",
    "Radius 12",
    "Radius 12",
    "Radius 14",
    "Radius 14",
]
headings = ["SK", "DEK", "SK", "DEK", "SK", "DEK", "SK", "DEK", "SK", "DEK"]

# units: seconds.
times_data = to_table_string.(
    [
        sk_both6 dek_both6 sk_both8 dek_both8 sk_both10 dek_both10 sk_both12 dek_both12 sk_both14 dek_both14
        sk_mean6 dek_mean6 sk_mean8 dek_mean8 sk_mean10 dek_mean10 sk_mean12 dek_mean12 sk_mean14 dek_mean14
    ]
)

label_col = ["Both" ; "Mean";]

body = [
    hcat(
        Cell(nothing),
        Cell.(categories, bold = true, merge = true, border_bottom = true)',
    )
    hcat(Cell(nothing), Cell.(headings)')
    hcat(
        Cell.(label_col),
        Cell.(times_data)
    )
]

lazy_query_table = Table(body, header = 2)


dir = joinpath("results")
texfile = joinpath(dir, "lazy_query_table.tex")

open(texfile, "w") do io
    # add the necessary packages in the preamble
    println(
        io, raw"""
            \documentclass{article}
            \usepackage{threeparttable}
            \usepackage{multirow}
            \usepackage{booktabs}
            \begin{document}
        """
    )

    # print the table as latex code
    show(io, MIME"text/latex"(), lazy_query_table)

    println(io, raw"\end{document}")
end

# render the tex file to pdf
tectonic_jll.tectonic() do bin
    run(`$bin --chatter=minimal $texfile`)
end


# # Conventional GPR query timing

sk_rkhs2, sk_both2, sk_mean2, sk_var2, dek_rkhs2, dek_both2, dek_mean2, dek_var2 = deserialize(joinpath("results", "time_query_1_down_2"))

sk_rkhs4, sk_both4, sk_mean4, sk_var4, dek_rkhs4, dek_both4, dek_mean4, dek_var4 = deserialize(joinpath("results", "time_query_1_down_4"))

sk_rkhs6, sk_both6, sk_mean6, sk_var6, dek_rkhs6, dek_both6, dek_mean6, dek_var6 = deserialize(joinpath("results", "time_query_1_down_6"))

function to_table_string_from_us(x_us)
    if x_us < 1.0e3
        return @sprintf("%.1f us", x_us) # keep as microseconds
    end

    if x_us < 1.0e6
        return @sprintf("%.1f ms", x_us / 1000) # miliseconds
    end

    return @sprintf("%.1f s", x_us / 1.0e6) # seconds
end
categories = [
    "Size (95, 105)",
    "Size (95, 105)",
    "Size (48, 53)",
    "Size (48, 53)",
    "Size (32, 35)",
    "Size (32, 35)",
]
headings = ["SK", "DEK", "SK", "DEK", "SK", "DEK"]

# units: seconds.
times_data = to_table_string_from_us.(
    [
        sk_rkhs2 dek_rkhs2 sk_rkhs4 dek_rkhs4 sk_rkhs6 dek_rkhs6
        sk_both2 dek_both2 sk_both4 dek_both4 sk_both6 dek_both6
        sk_mean2 dek_mean2 sk_mean4 dek_mean4 sk_mean6 dek_mean6
    ]
)

label_col = ["Setup"; "Both"; "Mean";]

body = [
    hcat(
        Cell(nothing),
        Cell.(categories, bold = true, merge = true, border_bottom = true)',
    )
    hcat(Cell(nothing), Cell.(headings)')
    hcat(
        Cell.(label_col),
        Cell.(times_data)
    )
]

conventional_query_table = Table(body, header = 2)


dir = joinpath("results")
texfile = joinpath(dir, "conventional_query_table.tex")

open(texfile, "w") do io
    # add the necessary packages in the preamble
    println(
        io, raw"""
            \documentclass{article}
            \usepackage{threeparttable}
            \usepackage{multirow}
            \usepackage{booktabs}
            \begin{document}
        """
    )

    # print the table as latex code
    show(io, MIME"text/latex"(), conventional_query_table)

    println(io, raw"\end{document}")
end

# render the tex file to pdf
tectonic_jll.tectonic() do bin
    run(`$bin --chatter=minimal $texfile`)
end


# # Non-GPR timing

function to_table_string_from_ns(x_ns)
    if typeof(x_ns) <: AbstractString
        return x_ns # skip strings
    end

    if x_ns < 1.0e3
        return @sprintf("%.1f ns", x_ns) # keep as nanoseconds
    end

    if x_ns < 1.0e6
        return @sprintf("%.1f us", x_ns / 1000) # microseconds
    end

    return @sprintf("%.1f ms", x_ns / 1.0e6) # miliseconds
end

time_bicubic, time_setup_bicubic, time_idw, time_hiyoshi2, time_laplace, time_sibson, time_setup_nn = deserialize(joinpath("results", "time_non_gpr"))

headings = ["Bi-cubic B-splines", "Inverse Distance Weighting", "Sibson"]
times_data = to_table_string_from_ns.(
    [
        time_bicubic time_sibson time_idw
        time_setup_bicubic "none" time_setup_nn
    ]
)
label_col = ["Query", "Setup"]

body = [
    Cell.(headings)';
    Cell.(times_data)
]

body = [
    hcat(Cell(nothing), Cell.(headings)')
    hcat(
        Cell.(label_col),
        Cell.(times_data)
    )
]

alternatives_table = Table(body, header = 2)


dir = joinpath("results")
texfile = joinpath(dir, "alternatives_table.tex")

open(texfile, "w") do io
    # add the necessary packages in the preamble
    println(
        io, raw"""
            \documentclass{article}
            \usepackage{threeparttable}
            \usepackage{multirow}
            \usepackage{booktabs}
            \begin{document}
        """
    )

    # print the table as latex code
    show(io, MIME"text/latex"(), alternatives_table)

    println(io, raw"\end{document}")
end

# render the tex file to pdf
tectonic_jll.tectonic() do bin
    run(`$bin --chatter=minimal $texfile`)
end


# I am here. load all the results from time_query.sh assemble into table.
# then assemble second table from non-gpr timings.
# then move onto patchwork kriging.

nothing
