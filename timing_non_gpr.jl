# run timeing_query.sh first.

down_factor = 2

const T = Float64
time_normalizing_constant = 1 # nano sectonds.

const D = 2

using Random, Images, BenchmarkTools, LinearAlgebra, Serialization
import Interpolations
import ScatteredInterpolation
using NaturalNeighbours

import PythonPlot as PLT
PLT.close("all")
fig_num = 1

# # image
include(joinpath("helpers", "image.jl"))

data_dir = joinpath("data", "images")
image_file_name = "kodim05_cropped.png"
data_path = joinpath(data_dir, image_file_name)

im_y, image_ranges, _ = getdownsampledimage(
    T, data_path, down_factor;
    discard_pixels = 0,
)
y = vec(im_y)

Xrs = Tuple(image_ranges);
Xs = vec(collect(collect(x) for x in Iterators.product(Xrs...)))
x1s = map(xx -> xx[1], Xs)
x2s = map(xx -> xx[2], Xs)

println("Natural neighbors' setup")
itp = interpolate(x1s, x2s, y; derivatives = true)
q = @benchmark interpolate($x1s, $x2s, $y; derivatives = true)
time_setup_nn = median(q.times) / time_normalizing_constant
display(q)


up_factor = down_factor * 2
Xqrs = (
    LinRange(first(Xrs[1]), last(Xrs[1]), round(Int, length(Xrs[1]) * up_factor)),
    LinRange(first(Xrs[2]), last(Xrs[2]), round(Int, length(Xrs[2]) * up_factor)),
)
Nr, Nc = length.(Xqrs);

# specify test point as the mid-point of the query set.
Xqs = collect(collect(x) for x in Iterators.product(Xqrs...))
xq_test = Xqs[round(Int, length(Xqs) / 2)]

println("Natural neighbors query, Sibson:")
itp(xq_test[1], xq_test[2]; method = Sibson())
q = @benchmark itp($xq_test[1], $xq_test[2]; method = Sibson())
time_sibson = median(q.times) / time_normalizing_constant
display(q)

println("Natural neighbors query, Laplace:")
itp(xq_test[1], xq_test[2]; method = Laplace())
q = @benchmark itp($xq_test[1], $xq_test[2]; method = Laplace())
time_laplace = median(q.times) / time_normalizing_constant
display(q)

println("Natural neighbors query, Hiyoshi-2: ")
itp(xq_test[1], xq_test[2]; method = Hiyoshi(2))
q = @benchmark itp($xq_test[1], $xq_test[2]; method = Hiyoshi(2))
time_hiyoshi2 = median(q.times) / time_normalizing_constant
display(q)
println()

# # Inverse function weighting
X_mat = reshape(collect(Iterators.flatten(Xs)), D, length(Xs))
itp_obj = ScatteredInterpolation.interpolate(
    ScatteredInterpolation.Shepard(3),
    X_mat,
    y,
)
println("Inverse distance weighting query:")
ScatteredInterpolation.evaluate(itp_obj, xq_test)
q = @benchmark ScatteredInterpolation.evaluate($itp_obj, $xq_test)
time_idw = median(q.times) / time_normalizing_constant
display(q)


# # Bi-cubic interpolation
# # Query bi-cubic interpolator
#Setup the bi-cubic interpolator.

function setup_etp(im_y::AbstractArray{T}, Xrs) where {T}
    itp = Interpolations.interpolate(
        im_y,
        Interpolations.BSpline(
            Interpolations.Cubic(
                Interpolations.Line(Interpolations.OnGrid()),
            ),
        ),
    )
    scaled_itp = Interpolations.scale(
        itp, Xrs...,
    )
    etp = Interpolations.extrapolate(
        scaled_itp, zero(T),
    )
    return etp
end
etp = setup_etp(im_y, Xrs)
println("Bi-cubic interpolation, setup:")
q = @benchmark setup_etp($im_y, $Xrs)
time_setup_bicubic = median(q.times) / time_normalizing_constant
display(q)

#Evaluate at the query positions.
println("Bi-cubic interpolation, query:")
q = @benchmark $etp($xq_test[1], $xq_test[2])
time_bicubic = median(q.times) / time_normalizing_constant
display(q)

serialize(joinpath("results", "time_non_gpr"), (time_bicubic, time_setup_bicubic, time_idw, time_hiyoshi2, time_laplace, time_sibson, time_setup_nn))

nothing
