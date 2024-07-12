
function visualizebernsteinbasis(
    λ_max::Real,
    L::Integer,
    fig_num::Integer;
    title_string = "Bernstein polynomial basis of degree $L",
    Nq::Integer = 1000,
    fig_size = (3, 2), # width, height.
    dpi = 96,
    )

    #
    evalbern = collect(
        tt->GSP.evalbernsteinpolynomialdirect(i, L, tt/λ_max)
        for i = 0:L
    )

    ts = LinRange(0, λ_max, Nq)

    B_evals = collect( evalbern[i+1].(ts) for i = 0:L )
    sum_B = sum(B_evals)

    PLT.figure(fig_num; figsize = fig_size, dpi = dpi)
    fig_num += 1

    for i = 0:L
        PLT.plot(ts, B_evals[begin+i], label = "Basis $i")
    end

    PLT.plot(ts, sum_B, label = "Sum", "--")
    PLT.title(title_string)
    PLT.legend(loc = "center left", bbox_to_anchor = (1, 0.5)) # legend outside of plot.

    return fig_num
end