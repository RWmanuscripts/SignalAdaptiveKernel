
T = Float64
D = 2;

tag, η_string = ARGS
η = parse(T, η_string)


# # Setup
using Pkg
Pkg.activate(".")
let
    pkgs = ["PlotlyLight", "ColorSchemes", "VisualizationBag", "SpatialGSP"]
    for pkg in pkgs
        if Base.find_package(pkg) === nothing
            Pkg.add(pkg)
        end
    end
end;

# Load packages.
using LinearAlgebra
import PlotlyLight as PLY
import ColorSchemes

import VisualizationBag as VIZ
import SpatialGSP as GSP

import Random
Random.seed!(25);


#src scatter data
N = 20
X = collect( randn(T, D) for _ = 1:N )

#src grid-data
if tag == "grid"
    X0 = vec(collect( Iterators.product(-5:4,-3:6) ))
    X = collect( convert(Vector{T}, collect(x)) for x in X0)
    N = length(X)
end

# # Axis search
kernel_ref = GSP.SqExp(T)
config = GSP.AxisSearchConfig{T}(
    kernel_σ_multiplier = η,
    make_undirected = true,
    remove_null_nbs = true,
    w_lb = convert(T, 0.1)
)
G = GSP.create_axis_graph(config, X)

# ## Visualize.
width = 600 # plot width.
height = 600 # plot height.

# Create edges
edge_h = GSP.getedgecoord(G, X, 1) # horizontal
edge_v = GSP.getedgecoord(G, X, 2) # vertical

# Create nodes
x_h = map(xx->xx[begin], X)
x_v = map(xx->xx[begin+1], X)
N_nbs = GSP.getNnbs(G)

nodes_trace = PLY.Config(
    name = "Node",
    x = x_v,
    y = x_h,
    mode = "markers",
    text = [
        "Node $n, Neighbors: $(N_nbs[n])"
        for n in eachindex(N_nbs)
    ],
    marker = PLY.Config(
        #showscale = true,
        showscale = false,
        colorscale = ColorSchemes.viridis,
        color = N_nbs,
        size = 10,
    ),
)

layout = PLY.Config(
    width = width,
    height = height,
    margin = PLY.Config(
        l = 1,
        r = 1,
        t = 1,
        b = 1,
    ),
    hovermode = "closest",
    title = "",
    titlefont = PLY.Config(
        size = 16,
    ),
    showlegend = false,
    showarrow = false,
    xaxis = PLY.Config(
        showgrid = false,
        zeroline = false,
        showticklabels = false,
    ),
    yaxis = PLY.Config(
        showgrid = false,
        zeroline = false,
        showticklabels = false,
    ),
)

eds, mid_pts_trace = VIZ.get2Dedgetraces_variablewidth(
    PLY, G.edges.srcs, G.edges.dests, G.edges.ws, X,
)
v = vcat(eds...)
push!(v, mid_pts_trace)
push!(v, nodes_trace)
ph = PLY.Plot(v, layout);

# # save.
PLY.save(ph, "figs/axis/axis_graph_$(η)_$(tag).html")

import PlotlyKaleido

PlotlyKaleido.start()
(;data, layout, config) = ph
PlotlyKaleido.savefig(
    ph,
    "figs/axis/axis_graph_$(η)_$(tag).svg";
    width = width,
    height = height,
)
PlotlyKaleido.savefig(
    ph,
    "figs/axis/axis_graph_$(η)_$(tag).png";
    width = width,
    height = height,
)

nothing