
function loadkodakimage(::Type{T}, file_path::String; discard_pixels::Integer = 0) where T <: AbstractFloat
    
    img = Images.load(file_path)
    gray_img = Images.Gray.(img)
    y_nD = convert(Array{T}, gray_img)
    
    # some kodak images have border artefacts. remove pixels.
    a = discard_pixels
    y_nD = y_nD[begin+a:end-a, begin+a:end-a]    

    return y_nD
end

function image2samples(y_nD::Matrix{T}) where T

    Nr, Nc = size(y_nD)
    x_ranges = getmatrixranges(Nr, Nc)

    return y_nD, x_ranges
end

function getmatrixranges(Nr::Integer, Nc::Integer)
    
    v_range = LinRange(1, Nr, Nr)
    h_range = LinRange(1, Nc, Nc)
    

    x_ranges = Vector{LinRange{T,Int}}(undef, 2)
    x_ranges[1] = v_range
    x_ranges[2] = h_range

    return x_ranges
end

function getdownsampledimage(
    ::Type{T},
    image_path,
    down_factor;
    discard_pixels = 1) where T
    
    img = loadkodakimage(T, image_path; discard_pixels = discard_pixels)
    im_y_ref = convert(Array{T}, Images.Gray.(img))
    
    
    im_y = im_y_ref[begin:down_factor:end, begin:down_factor:end]
    image_ranges = getmatrixranges(size(im_y)...)

    return im_y, image_ranges, im_y_ref
end

function loadparrotface(
    b_x::T;
    down_factor::Integer = 2,
    up_factor::Real = 4.5,
    ) where T <: AbstractFloat
    #
    #down_factor = 2
    im_y, image_ranges, _ = getdownsampledimage(
        T, "data/kodim23.png", down_factor;
        discard_pixels = 1,
    )
    
    query_b_box = LGP.BoundingBox(
        round.(Int, (200, 190) ./ down_factor),
        round.(Int, (275, 261) ./ down_factor),
    )

    xq_ranges = LGP.getqueryranges(
        up_factor, query_b_box, step.(image_ranges),
    )
    
    # noise.
    M = floor(Int, b_x) # base this on b_x.
    L = M # must be an even positive integer. The larger the flatter.
    if isodd(L)
        L = L + 1
    end
    x0, y0 = convert(T, 0.8*M), 1 + convert(T, 0.5)
    s_map = LGP.AdjustmentMap(x0, y0, b_x, L)

    return im_y, Tuple(image_ranges), s_map, Tuple(xq_ranges)
end

# rescale the values of X to 01.
function scale01(X::Matrix)

    lb = minimum(X)
    ub = maximum(X)
    return collect(
        convertcompactdomain(x, lb, ub, zero(T), one(T))
        for x in X
    )
end