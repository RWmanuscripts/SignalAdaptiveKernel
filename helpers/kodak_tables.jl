
################# SSIM and image output.

function get_scene_names()

    out = Vector{String}(undef, 24)
    for i = 1:24

        image_name = "kodim0$i"
        if i >= 10
            image_name = "kodim$i"
        end

        out[i] = image_name
    end

    return out
end

function transform_img(A::Matrix, transform)
    
    out = A
    if transform == "clip"
    
        out = map(Images.clamp01nan, out)
        
    elseif transform == "scale"

        out = scale01(out)
        
    end
    return out
end

struct ImageHP{T}
    dek_vars_set::Vector{Vector{T}} # each entry is for a different σr value.
    dek_score_set::Vector{T} # each entry is for a different σr value.
    sk_vars::Vector{T}
    sk_score::T
    κ_ub::T
end

function process_image_result(
    im_y,
    load_path;
    #verbose = true,
    save_dir = [],
    image_name = "",
    transform = "none", # "none", "clip", "scale"
    )

    (
        itp_Xq, mqs_sk, vqs_sk,
        mqs_dek_set, vqs_dek_set,
        dek_vars_set, dek_score_set,
        sk_vars_set, sk_score_set, κ_ub,
    ) = deserialize(load_path)

    mqs_sk = reshape(mqs_sk, size(itp_Xq)) # remove later.
    vqs_sk = reshape(vqs_sk, size(itp_Xq)) # remove later.

    # sanity check.
    ref = sk_vars_set[begin]
    for i in eachindex(sk_vars_set)
        residual = norm(ref - sk_vars_set[i])
        @assert isapprox(residual, 0)
    end

    ref2 = sk_score_set[begin]
    for i in eachindex(sk_score_set)
        residual = abs(ref2 - sk_score_set[i])
        @assert isapprox(residual, 0)
    end

    # stationary kernel, interpolation.
    mqs_sk2 = transform_img(mqs_sk, transform)
    itp_Xq2 = transform_img(itp_Xq, transform)

    if !isempty(save_dir) && !isempty(image_name)
        save_path = joinpath(save_dir, "$(image_name)_itp")
        FileIO.save(File{format"PNG"}(save_path), mqs_sk2)
        
        save_path = joinpath(save_dir, "$(image_name)_sk")
        FileIO.save(File{format"PNG"}(save_path), itp_Xq2)
    end

    SSIM_itp = Images.assess_ssim(itp_Xq2, im_y)
    SSIM_sk = Images.assess_ssim(mqs_sk2, im_y)

    # DE kernels.
    SSIM_deks = zeros(T, length(mqs_dek_set))
    min_deks = zeros(T, length(mqs_dek_set))
    max_deks = zeros(T, length(mqs_dek_set))
    for (k,r) in Iterators.enumerate(eachindex(mqs_dek_set))
        mqs_dek = mqs_dek_set[r]
        mqs_dek2 = transform_img(mqs_dek, transform)

        if !isempty(save_dir) && !isempty(image_name)
            save_path = joinpath(save_dir, "$(image_name)_dek_$k")
            FileIO.save(File{format"PNG"}(save_path), mqs_dek2)
        end

        SSIM_deks[r] = Images.assess_ssim(mqs_dek2, im_y)
        min_deks[r] = minimum(mqs_dek2)
        max_deks[r] = maximum(mqs_dek2)
    end

    # hyperparameters
    return ImageHP(
        dek_vars_set, dek_score_set,
        sk_vars_set[begin], sk_score_set[begin], κ_ub,
    ),
    SSIM_itp, SSIM_sk, SSIM_deks
end


function process_kodak_results(
    ::Type{T},
    reference_image_folder,
    load_dir;
    save_dir = [], # empty if don't want to save to png.
    model_select_string = "", # save file name postfix.
    transform = "none", # "none", "clip", "scale"
    scenes = 1:24,
    ) where T

    M = length(scenes)
    SSIMs_itp = Vector{T}(undef, M)
    SSIMs_sk = Vector{T}(undef, M)
    SSIMs_deks = Vector{Vector{T}}(undef, M)

    hps = Vector{ImageHP{T}}(undef, M)

    for (k, i) in Iterators.enumerate(scenes)

        image_name = "kodim0$i"
        if i >= 10
            image_name = "kodim$i"
        end

        image_path = joinpath(reference_image_folder, "$(image_name).png")
        im_y, _ = getdownsampledimage(
            T, image_path, 1; discard_pixels = 0,
        )

        load_path = joinpath(load_dir, "upconvert_$(image_name)_$(model_select_string)")
        hps[k], SSIMs_itp[k], SSIMs_sk[k], SSIMs_deks[k] = process_image_result(
            im_y,
            load_path;
            save_dir = save_dir,
            transform = transform,
            image_name = "$(image_name)_$(model_select_string)",
        )

    end

    return hps, SSIMs_itp, SSIMs_sk, SSIMs_deks
end

# show only maximum
function generate_SSIM_table(
    SSIMs_itp, SSIMs_sk, SSIMs_deks;
    sig_digits = 5
    )

    dmat_deks = hcat(SSIMs_deks...)
    dmat = [SSIMs_itp SSIMs_sk dmat_deks']

    # rankings.
    tmp = collect(
        sortperm(dmat[r,:], rev = true) for r in axes(dmat,1)
    )
    rankings = map(xx->tuple(xx...), tmp)

    # largest.
    bold_cols = map(xx->xx[begin], rankings)

    # smallest.
    italics_cols = map(xx->xx[end], rankings)

    f = xx->round(xx, sigdigits = sig_digits)
    dmat3 = string.(f.(dmat))
    for r in axes(dmat3,1)
        x = dmat3[r,:]

        # bold.
        entry = x[bold_cols[r]]
        x[bold_cols[r]]  = "**$(entry)**"

        # italics.
        entry = x[italics_cols[r]]
        x[italics_cols[r]]  = "*$(entry)*"
        
        # save.
        dmat3[r,:] = x
    end
    
    return dmat3
end

# hyperparameters

function generate_hp_table(hps::Vector{ImageHP{T}}; sig_digits = 3) where T

    # first column is for the canonical kernel.
    out = Matrix{String}(undef, length(hps), 1+length(hps[begin].dek_vars_set))
    for r in eachindex(hps)
        
        tmp = hcat(hps[r].dek_vars_set...)
        κ_rs = vec(tmp[end,:])

        a_r = hps[r].sk_vars[begin]
        
        f = xx->round(xx, sigdigits = sig_digits)
        tmp = string.(map(f, [a_r; κ_rs]))

        out[r,:] = tmp
    end

    return out
end
