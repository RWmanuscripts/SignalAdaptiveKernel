# Timing:

const T = Float64
time_normalizing_constant = 1000 # micro sectonds.

b_x_inp = ARGS[1]
b_x = parse(Float64, b_x_inp)

# # for non-command line use.
#b_x = convert(T, 6);

@show b_x

down_factor = 2

# # Setup
# Install dependencies
import Pkg
let
    pkgs = ["PythonPlot"]
    for pkg in pkgs
        if isnothing(Base.find_package(pkg))
            Pkg.add(pkg)
        end
    end
end;

import Random
Random.seed!(25)

using LinearAlgebra, Serialization, BenchmarkTools

import Interpolations
import Metaheuristics as EVO
import Images
import LocalFilters
import SpatialGSP as GSP
import LazyGPR as LGP

import VisualizationBag as VIZ

const D = 2;

#Data preparation
data_dir = joinpath("data", "images")
image_file_name = "kodim05_cropped.png"

# data_dir = joinpath("data")
# image_file_name = "kodim23.png"

model_selection_string = "ML" # "ML" means we use marginal likelihood for the hyperparameter optimization objective. Another valid option is "LOOCV", for the leave-one-out cross-validation objective; see  https://doi.org/10.7551/mitpress/3206.001.0001  for details on LOOCV.

#Prameters for smoothing the warp samples before making it into a warp function. The bilateral filter is used.
#If this is larger, then the warp samples are smoother, with increased risk of removing details:
#see compute_kodak_hp() and compute_kodak_warpmap() from the helper script `hopt.jl` to see how `σr_factor` is used to compute `σr`, the bilateral filter intensity standard deviation.
σr_factor = convert(T, 4.0)
σs = one(T); # The spatial standard deviation for the bilateral filter.

# Lazy-evaluation GPR model parameters:
σ² = convert(T, 0.001)
N_neighbourhoods = 50 # The maximum number of local datasets we use in the objective function is this number times two. This is `M` in our manuscript.

#For hyperparameter optimization
f_calls_limit = 1_000 # soft-constraint on the number of objective function evaluations during optimization.
a_lb = convert(T, 0.001) # lower bound for the bandwidth parameter.
a_ub = one(T)
N_initials_a = 100 # Number of initial guesses for the a, bandwidth parameter.
N_initials_κ = 100; # Similarly for the κ, gain parameter.

#how the warp samples are combined across different graph spectrums.
aggregate_symbol = :sum # An alternative option is :sum_normalized.

if model_selection_string != "ML" &&
        model_selection_string != "LOOCV"

    println("Unknown model_selection_string. Default to Marginal likelihood.")
    model_selection_string = "ML"
end

model_selection_trait = LGP.MarginalLikelihood()
if model_selection_string == "LOOCV"
    model_selection_trait = LGP.LOOCV()
end


# The following helper scripts can be found in `examples/helpers/` from the root repository folder.
include("helpers/hopt.jl")
include("helpers/utils.jl")
include("helpers/image.jl");

# Setup data.
data_path = joinpath(data_dir, image_file_name)
im_y, image_ranges, _ = getdownsampledimage(
    T, data_path, down_factor;
    discard_pixels = 0,
)
Xrs = Tuple(image_ranges);

# adjustment map.
M = floor(Int, b_x) # base this on b_x.
L = M # must be an even positive integer. The larger the flatter.
if isodd(L)
    L = L + 1
end
x0, y0 = convert(T, 0.8 * M), 1 + convert(T, 0.5)
s_map = LGP.AdjustmentMap(x0, y0, b_x, L);

# Warp samples, from the unit grid graph.
warp_config = GSP.WarpConfig{T}(
    aggregate_option = aggregate_symbol,
)
W = LGP.create_grid_warp_samples(
    LGP.UseSpatialGSP(GSP),
    im_y,
    warp_config,
);

# # Kerenl hyperparameter fitting

# Assemble model container.
model = LGP.LazyGP(
    b_x, s_map,
    LGP.GPData(σ², Xrs, im_y),
);

# # Query
load_path = joinpath(
    "results",
    "timing_hp_lazy_downfactor_$(down_factor)_neighbourhood_$(N_neighbourhoods)",
)
down_factor2, N_neighbourhoods, dek_vars, dek_score, sk_vars, sk_score = deserialize(load_path)

# sanity-check
@assert down_factor == down_factor2

# # Query GPR Models
# Setup options
options = LGP.QueryOptions();

up_factor = down_factor * 2
Xqrs = (
    LinRange(first(Xrs[1]), last(Xrs[1]), round(Int, length(Xrs[1]) * up_factor)),
    LinRange(first(Xrs[2]), last(Xrs[2]), round(Int, length(Xrs[2]) * up_factor)),
)
Nr, Nc = length.(Xqrs);

# specify test point as the mid-point of the query set.
Xqs = collect(collect(x) for x in Iterators.product(Xqrs...))
xq_test = Xqs[round(Int, length(Xqs) / 2)]

# lazy GPR: Canonical Kernel Model

sk = LGP.WendlandSplineKernel(
    LGP.Order2(), sk_vars[1], 3,
)
out = LGP.lazyquery2(
    Xqs[begin], model, sk,
    LGP.QueryOptions(
        compute_mean = LGP.Enable(),
        compute_variance = LGP.Enable(),
    ),
    nothing,
)
println("Lazy query: stationary kernel, mean and variance:")
q = @benchmark LGP.lazyquery2(
    $xq_test, $model, $sk,
    LGP.QueryOptions(
        compute_mean = LGP.Enable(),
        compute_variance = LGP.Enable(),
    ),
    nothing,
)
sk_both = median(q.times) / time_normalizing_constant
display(q)
println("Lazy query: stationary kernel, mean-only:")
q = @benchmark LGP.lazyquery2(
    $xq_test, $model, $sk,
    LGP.QueryOptions(
        compute_mean = LGP.Enable(),
        compute_variance = LGP.Disable(),
    ),
    nothing,
)
sk_mean = median(q.times) / time_normalizing_constant
display(q)
println("Lazy query: stationary kernel, variance-only:")
q = @benchmark LGP.lazyquery2(
    $xq_test, $model, $sk,
    LGP.QueryOptions(
        compute_mean = LGP.Disable(),
        compute_variance = LGP.Enable(),
    ),
    nothing,
)
sk_var = median(q.times) / time_normalizing_constant
display(q)
println()


# DE Kernel Model
sigma_r_factor = T(3)

warpmap = compute_kodak_warpmap(W, sigma_r_factor, Xrs)

kernel_param, κ = dek_vars[1], dek_vars[2]
canonical_kernel = LGP.WendlandSplineKernel(
    LGP.Order2(), kernel_param, 3,
)
dek = LGP.DEKernel(canonical_kernel, warpmap, κ)

println("Lazy query: DE kernel, setup cache:")
cvars = LGP.computecachevars(dek, model)
q = @benchmark LGP.computecachevars(dek, model)
dek_cache = median(q.times) / time_normalizing_constant
display(q)

out = LGP.lazyquery2(
    Xqs[begin], model, dek,
    LGP.QueryOptions(
        compute_mean = LGP.Enable(),
        compute_variance = LGP.Enable(),
    ),
    cvars,
)

println("Lazy query: DE kernel, mean and variance:")
q = @benchmark LGP.lazyquery2(
    $xq_test, $model, $dek,
    LGP.QueryOptions(
        compute_mean = LGP.Enable(),
        compute_variance = LGP.Enable(),
    ),
    cvars,
)
dek_both = median(q.times) / time_normalizing_constant
display(q)
println("Lazy query: DE kernel, mean-only:")
q = @benchmark LGP.lazyquery2(
    $xq_test, $model, $dek,
    LGP.QueryOptions(
        compute_mean = LGP.Enable(),
        compute_variance = LGP.Disable(),
    ),
    cvars,
)
dek_mean = median(q.times) / time_normalizing_constant
display(q)
println("Lazy query: DE kernel, variance-only:")
q = @benchmark LGP.lazyquery2(
    $xq_test, $model, $dek,
    LGP.QueryOptions(
        compute_mean = LGP.Disable(),
        compute_variance = LGP.Enable(),
    ),
    cvars,
)
dek_var = median(q.times) / time_normalizing_constant
display(q)
println()

serialize(joinpath("results", "time_query_lazy_bx_$(b_x)"), (sk_both, sk_mean, sk_var, dek_cache, dek_both, dek_mean, dek_var))

nothing
