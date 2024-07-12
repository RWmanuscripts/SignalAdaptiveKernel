# for use with kodak.sh

# activate the current virtual environment
using Pkg
Pkg.activate(".")

# # Setup
import JSON3
import Interpolations
import Metaheuristics as EVO
import Images
import LocalFilters
import SpatialGSP as GSP

#using Revise
import LazyGPR as LGP

const T = Float64
const D = 2

include("helpers/hopt.jl")

# # from load from command line.
save_dir, image_file_name, 
N_workers_string, model_selection_string,
down_factor_string = ARGS

if !ispath(save_dir)
    mkpath(save_dir)
end

# load config.
inp = JSON3.read("kodak_config.json")

data_dir = inp[:data_dir]
σ² = convert(T, inp[:noise_variance])
b_x = convert(T, inp[:local_radius])

aggregate_string = inp[:aggregate_option]
sigma_rs = inp[:bilateral_filter_sigma_r]

f_calls_limit = convert(Int, inp[:f_calls_limit])
N_neighbourhoods = convert(Int, inp[:half_max_size_S])

a_lb = convert(T, inp[:a_lb])
a_ub = convert(T, inp[:a_ub])
N_initials_a = convert(Int, inp[:N_initials_a])
N_initials_κ = convert(Int, inp[:N_initials_kappa])

width_factor = 1
σs = one(T)

# process inputs.
aggregate_symbol = :sum_normalized
if aggregate_string == "sum"
    aggregate_symbol = :sum
end

if model_selection_string != "ML" 

    println("Unknown model_selection_string. Default to Marginal likelihood.")
    model_selection_string = "ML"
end

model_selection_trait = LGP.MarginalLikelihood()

N_workers = parse(Int, N_workers_string)
down_factor = parse(Int, down_factor_string)
σr_factors = convert(Vector{T}, sigma_rs)

# ## Set up distributed environments
using Distributed

clear_tmp = true

worker_list = ones(Int, 1) # if this is the first time we're launching.
if nprocs() > 1
    worker_list = workers() # if we're re-using the existing workers.
end
if nworkers() < N_workers
    worker_list = LGP.create_local_procs(
        N_workers;
        pkg_load_list = [
            # :(using Revise);
            :(import Interpolations);
            :(import Metaheuristics);
        ],
        #verbose = true,
        verbose = false,
    )
end
@show worker_list

import Random
Random.seed!(25)

using LinearAlgebra
using Serialization

include("helpers/utils.jl")
include("helpers/image.jl")

# set up data.
data_path = joinpath(data_dir, image_file_name)
im_y, image_ranges, _ = getdownsampledimage(
    T, data_path, down_factor;
    discard_pixels = 0,
)
Xrs = Tuple(image_ranges)

# adjustment map.
M = floor(Int, b_x) # base this on b_x.
L = M # must be an even positive integer. The larger the flatter.
if isodd(L)
    L = L + 1
end
x0, y0 = convert(T, 0.8*M), 1 + convert(T, 0.5)
s_map = LGP.AdjustmentMap(x0, y0, b_x, L)

# Warp map
W = LGP.create_grid_warp_samples(
    LGP.UseSpatialGSP(GSP),
    im_y,
    GSP.WarpConfig{T}(
        aggregate_option = aggregate_symbol,
    ),
)


# # Kerenl hyperparameter fitting

# ## construct cost function for hyperparameter optimization
# assemble container for model.
model = LGP.LazyGP(
    b_x, s_map,
    LGP.GPData(σ², Xrs, im_y),
)
# kernel selection.
# 3D, order 2 spline kernel.
ref_can_kernel = LGP.WendlandSplineKernel(
    LGP.Order2(), one(T), 3,
)

# tuning parameters.
max_abs_W = maximum(abs.(W))
κ_ub = convert(T, (b_x*3) * 1/max_abs_W)
V = im_y

lazy_hopt_config = LGP.LazyHCostConfig( # lazy-evaluation hyperparameter optimization config
    worker_list, N_neighbourhoods, true, V,
)

solver_config = LGP.MetaheuristicsConfig( # solver_config
    f_calls_limit = f_calls_limit,
)

optim_config = LGP.KernelOptimConfig{T}( # optim_config
    a_lb = a_lb,
    a_ub = a_ub,
    κ_ub = κ_ub,
    width_factor = 1, # carry over sk.
)

out = collect(
    compute_kodak_hp(
        W, r, Xrs, ref_can_kernel,
        model_selection_trait,
        LGP.UseMetaheuristics(EVO),
        lazy_hopt_config,
        solver_config, optim_config,
    )
    for r in σr_factors
)
# dek_vars, dek_star, sk_vars, sk_star, dek_ref
dek_vars_set = map(xx->xx[begin], out)
dek_score_set = map(xx->xx[begin+1], out)
sk_vars_set = map(xx->xx[begin+2], out)
sk_score_set = map(xx->xx[begin+3], out)

@show model_selection_trait
dmat = [dek_vars_set dek_score_set sk_vars_set sk_score_set]
println("[dek_vars_set dek_score_set sk_vars_set sk_score_set]")
display(dmat)
println()

# # Set up query options.
options = LGP.QueryOptions()

up_factor = down_factor
Xqrs = (
    LinRange(first(Xrs[1]), last(Xrs[1]), round(Int, length(Xrs[1])*up_factor) ),
    LinRange(first(Xrs[2]), last(Xrs[2]), round(Int, length(Xrs[2])*up_factor) ),
)
Nr, Nc = length.(Xqrs)

# # Stationary/canonical.
mqs_sk_vec, vqs_sk_vec = upconvert_kodak_sk(
    sk_vars_set[begin], worker_list, model, options, Xqrs,
)
mqs_sk = reshape(mqs_sk_vec, Nr, Nc)
vqs_sk = reshape(vqs_sk_vec, Nr, Nc)

# # DE kernel.
q_dek = collect(
    upconvert_kodak_dek(
        W, r, Xrs,
        dek_vars_set[i],
        worker_list, model, options,
        Xqrs,
    )
    for (i,r) in Iterators.enumerate(σr_factors)
)
mqs_dek_set = map(xx->reshape(xx[begin], Nr, Nc), q_dek)
vqs_dek_set = map(xx->reshape(xx[begin+1], Nr, Nc), q_dek)

# # Reference method: cubic itp.
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
) # zero outside interp range.

itp_Xq = collect(
    etp(x...)
    for x in Iterators.product(Xqrs...)
)

# # reference image.
# im_ref, _ = getdownsampledimage(
#     T, data_path, 1;
#     discard_pixels = 0,
# )

out_disk = (
    itp_Xq, mqs_sk, vqs_sk,
    mqs_dek_set, vqs_dek_set,
    dek_vars_set, dek_score_set,
    sk_vars_set, sk_score_set, κ_ub,
)


save_name = replace(
    "$image_file_name",
    ".png"=>"_$(model_selection_string)",
)
serialize(
    joinpath(save_dir, "upconvert_$save_name"),
    out_disk
)

nothing