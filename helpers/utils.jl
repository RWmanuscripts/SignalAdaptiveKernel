# tensor product filtering for dimension D discrete signal.
function ranges2collection(
        x_ranges::Vector{LinRange{T, L}},
        ::Val{D},
    )::Array{Vector{T}, D} where {T, D, L}

    # set up.
    @assert !isempty(x_ranges)
    @assert length(x_ranges) == D
    N_array = collect(length(x_ranges[d]) for d in 1:D)
    sz_N = tuple(N_array...)

    # Position.
    X_nD = Array{Vector{T}, D}(undef, sz_N)
    for 𝑖 in CartesianIndices(sz_N)
        X_nD[𝑖] = Vector{T}(undef, D)

        for d in 1:D
            X_nD[𝑖][d] = x_ranges[d][𝑖[d]]
        end
    end

    return X_nD
end

"""
    convertcompactdomain(x::Real, a::Real, b::Real, c::Real, d::Real)

converts compact domain x ∈ [a,b] to compact domain out ∈ [c,d].
"""
function convertcompactdomain(x::Real, a::Real, b::Real, c::Real, d::Real)

    return (x - a) * (d - c) / (b - a) + c
end

function convertcompactdomain(x::AbstractVector{<:Real}, a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, c::AbstractVector{<:Real}, d::AbstractVector{<:Real})

    return collect(convertcompactdomain(x[i], a[i], b[i], c[i], d[i]) for i in eachindex(x, a, b, c, d))
end

function convertcompactdomain(x::Vector{<:Real}, a::Real, b::Real, c::Real, d::Real)

    return collect(convertcompactdomain(x[i], a, b, c, d) for i in eachindex(x))
end

function inputs2mat(X::Array{Vector{T}, D}) where {T <: AbstractFloat, D}
    V = vec(X)

    Y = zeros(T, D, length(X))
    for i in eachindex(V)
        Y[:, i] = V[i]
    end

    return Y
end

function inputs2mat(X::Array{NTuple{D, T}, D}) where {T <: AbstractFloat, D}
    V = vec(X)

    Y = zeros(T, D, length(X))
    for i in eachindex(V)
        Y[:, i] .= V[i]
    end

    return Y
end


### visualization.

function vizmeshgrid(
        x_ranges::Vector{LinRange{T, L}},
        Y::Matrix{T},
        marker_locations::Vector,
        marker_symbol::String,
        fig_num::Int,
        title_string::String;
        x1_title_string::String = "Dimension 1",
        x2_title_string::String = "Dimension 2",
        cmap = "Greens_r",
    ) where {T <: Real, L}

    #
    @assert length(x_ranges) == 2
    x_coords = collect(collect(x_ranges[d]) for d in 1:2)

    PLT.figure(fig_num)
    fig_num += 1
    PLT.pcolormesh(x_coords[1], x_coords[2], Y, cmap = cmap, shading = "auto")
    PLT.xlabel(x1_title_string)
    PLT.ylabel(x2_title_string)
    PLT.title(title_string)

    for i in 1:length(marker_locations)
        #pt = reverse(marker_locations[i])
        pt = marker_locations[i]
        PLT.annotate(marker_symbol, xy = pt, xycoords = "data")
    end

    PLT.colorbar()
    PLT.axis("scaled")

    return fig_num
end


### delete folder contents.
function clearfolder(tmp_storage_folder::String)

    file_list = readdir(tmp_storage_folder)
    for i in eachindex(file_list)
        rm(joinpath(tmp_storage_folder, file_list[i]))
    end

    return nothing
end
