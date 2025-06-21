# Timing:

down_factor_inp = ARGS[1]
down_factor = parse(Int, down_factor_inp)
#down_factor = 10

println()
@show down_factor

# # Setup
# Install dependencies
import Pkg
let
    pkgs = ["PythonPlot"]
    for pkg in pkgs
        if Base.find_package(pkg) === nothing
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


const T = Float64
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
b_x = convert(T, 6);

#For hyperparameter optimization
f_calls_limit = 1_000 # soft-constraint on the number of objective function evaluations during optimization.
N_neighbourhoods = 50 # The maximum number of local datasets we use in the objective function is this number times two. This is `M` in our manuscript.
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
load_path = joinpath("results", "timing_hp_downfactor_$(down_factor)")
down_factor2,
    dek_vars, dek_score, sk_vars, sk_score,
    dek1_vars, dek1_score, sk1_vars, sk1_score, = deserialize(load_path)

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
display(q)
println()

#@assert 1 == 32

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
display(q)
println()

# # Non-lazy evaluation GPR
y = vec(im_y)
X = vec(collect(collect(x) for x in Iterators.product(Xrs...)))

# ## Stationary kernel
sk1 = LGP.SqExpKernel(sk1_vars[begin])

println("Non-lazy GPR, stationary kernel, RKHS system solve: ")
gp_sk1 = LGP.fitGP(X, y, σ², sk1)
q = @benchmark LGP.fitGP($X, $y, $σ², $sk1)
display(q)

println("Non-lazy GPR, stationary kernel, query mean and variance: ")
LGP.queryGP(xq_test, sk1, gp_sk1)
q = @benchmark LGP.queryGP($xq_test, $sk1, $gp_sk1)
display(q)

buf = zeros(T, length(X))
println("Non-lazy GPR, stationary kernel, query mean-only: ")
LGP.querymean(xq_test, sk1, gp_sk1)
q = @benchmark LGP.querymean($xq_test, $sk1, $gp_sk1)
display(q)

println("Non-lazy GPR, stationary kernel, query variance-only: ")
LGP.queryGPvariance!(buf, xq_test, sk1, gp_sk1)
q = @benchmark LGP.queryGPvariance!($buf, $xq_test, $sk1, $gp_sk1)
display(q)

#@assert 3 == 4444

# ## Stationary kernel
kernel_param1, κ1 = dek1_vars[begin], dek1_vars[end]

canonical_kernel1 = LGP.WendlandSplineKernel(
    LGP.Order2(), kernel_param, 3,
)
dek1 = LGP.DEKernel(canonical_kernel1, warpmap, κ1)

println("Non-lazy GPR, stationary kernel, RKHS system solve: ")
gp_dek1 = LGP.fitGP(X, y, σ², dek1)
q = @benchmark LGP.fitGP($X, $y, $σ², $dek1)
display(q)

println("Non-lazy GPR, stationary kernel, query mean and variance: ")
LGP.queryGP(xq_test, dek1, gp_dek1)
q = @benchmark LGP.queryGP($xq_test, $dek1, $gp_dek1)
display(q)

buf = zeros(T, length(X))
println("Non-lazy GPR, stationary kernel, query mean-only: ")
LGP.querymean(xq_test, dek1, gp_dek1)
q = @benchmark LGP.querymean($xq_test, $dek1, $gp_dek1)
display(q)

println("Non-lazy GPR, stationary kernel, query variance-only: ")
LGP.queryGPvariance!(buf, xq_test, dek1, gp_dek1)
q = @benchmark LGP.queryGPvariance!($buf, $xq_test, $dek1, $gp_dek1)
display(q)

println()

nothing
