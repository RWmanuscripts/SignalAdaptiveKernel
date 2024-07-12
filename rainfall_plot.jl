using Pkg
Pkg.activate(".")

import SingleLinkagePartitions as SL
import PlotlyLight as PLY
import CSV
import DataFrames as DF
using LinearAlgebra

T = Float64

df_ca = CSV.read("data/2023_08_10_CA_stations.csv", DF.DataFrame)

# merge points.
df = df_ca
X0 = collect(
    [
        df.LATITUDE[n];
        df.LONGITUDE[n];
    ]
    for n in eachindex(df.LATITUDE)
)
y0 = convert(Vector{T}, df.PRCP ./ 10) # to milimetres.
X, y = SL.avgduplicates(X0, y0, eps(T)*10)

lon = map(xx->xx[begin+1], X)
lat = map(xx->xx[begin], X)
values = y


# # display bounds.

## Southern Canada.
longitude_range = [-80, -68]
latitude_range = [42, 51] 

aspect_ratio = (latitude_range[end] - latitude_range[begin])/(longitude_range[end] - longitude_range[begin])

region_lons = [
    longitude_range[begin],
    longitude_range[begin],
    longitude_range[end],
    longitude_range[end],
    longitude_range[begin],
]
region_lats = [
    latitude_range[begin]-1,
    latitude_range[end],
    latitude_range[end],
    latitude_range[begin],
    latitude_range[begin],
]

qbox_trace = PLY.Config(
    #type = "scattergeo",
    type = "scatter",
    mode = "lines",
    x = region_lons,
    y = region_lats,
    line = PLY.Config(
        width = 1.5,
        color = "rgba(255,0,0,1)",
    ),
)



using ColorSchemes
color_scheme = ColorSchemes.amp

function createcolorscale(scheme)

    ts = LinRange(0, 1, length(scheme))

    return collect( 
        (ts[i], scheme[i] .* 255) # ColorSchemes.jl uses [0,1] while plotly uses [0,255].
        for i in eachindex(ts)
    )
end
colour_scale = createcolorscale(color_scheme)


plot_data = PLY.Config(
    name = "NA Precipitation",
    type = "scattergeo",
    mode = "markers",
    text = values,
    lon = lon,
    lat = lat,
    marker = PLY.Config(
        color = values,
        colorscale = colour_scale,
        cmin = minimum(values),
        reversescale = false,
        opacity = 0.8,
        size = 7,
        line = PLY.Config(
            width = 0.5, # marker border
            color = "rgba(0,0,0,1)",
        ),
        colorbar = PLY.Config(
            thickness = 10,
            titleside = "right",
            outlinecolor = "rgba(68,68,68,0)",
            ticks = "outside",
            ticklen = 3,
            shoticksuffix = "last",
            ticksuffix = "mm",
            dtick = 5,
        ),
    ),
)


width = 1200
height = 1200
layout = PLY.Config(
    #title = "Southern Canada Rainfall",
    title = "",
    width = width,
    height = height,
    margin = PLY.Config(
        l = 1,
        r = 1,
        t = 1,
        b = 1,
    ),
    font = PLY.Config(
        family = "Droid Serif, serif",
        size = 20
    ),
    titlefont = PLY.Config(
        size = 26
    ),
    geo = PLY.Config(
        scope = "Southern Canada",
        lonaxis = PLY.Config(
            showgrid = true,
            gridwidth = 0.5,
            range = longitude_range,
            dtick = 5
        ),
        lataxis = PLY.Config(
            showgrid = true,
            gridwidth = 0.5,
            range = latitude_range,
            dtick = 5
        ),
        showrivers = true,
        rivercolor = "#fff",
        showlakes = true,
        lakecolor = "rgb(255,255,255)",
        showland = true,
        landcolor = "rgb(212,212,212)",
        countrycolor = "rgb(255,255,255)",
        countrywidth = 1.5,
        subunitcolor = "rgb(255,255,255)",
        showsubunits = true,
        showcountries = true,
        resolution = 50,
        projection = PLY.Config(
            type = "conic conformal",
            rotation = PLY.Config(long = -100),
        )
    )
)
v = plot_data
ph = PLY.Plot(v, layout);
PLY.save(ph, "figs/rainfall/southern_ca_rainfall.html")

import PlotlyKaleido

PlotlyKaleido.start()
(;data, layout, config) = ph
PlotlyKaleido.savefig(
    ph,
    "figs/rainfall/southern_ca_rainfall.svg";
    width = width,
    height = height,
)
PlotlyKaleido.savefig(
    ph,
    "figs/rainfall/southern_ca_rainfall.png";
    width = width,
    height = height,
)

nothing