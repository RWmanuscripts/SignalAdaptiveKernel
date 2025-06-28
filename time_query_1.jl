# Timing:
const T = Float64
time_normalizing_constant = 1000 # micro sectonds.

down_factor_inp = ARGS[1]
down_factor = parse(Int, down_factor_inp)

# downsample/up-conversion factor. # for non-command line use.
#down_factor = 10

@show down_factor

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
σ² = convert(T, 0.001)

data_path = joinpath(data_dir, image_file_name)
im_y, image_ranges, _ = getdownsampledimage(
    T, data_path, down_factor;
    discard_pixels = 0,
)
Xrs = Tuple(image_ranges);

# Warp samples, from the unit grid graph.
warp_config = GSP.WarpConfig{T}(
    aggregate_option = aggregate_symbol,
)
W = LGP.create_grid_warp_samples(
    LGP.UseSpatialGSP(GSP),
    im_y,
    warp_config,
);

sigma_r_factor = T(3)
warpmap = compute_kodak_warpmap(W, sigma_r_factor, Xrs)

# # Query
load_path = joinpath("results", "timing_hp_single_downfactor_$(down_factor)")
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

# # Non-lazy evaluation GPR
y = vec(im_y)
X = vec(collect(collect(x) for x in Iterators.product(Xrs...)))

# ## Stationary kernel
sk1 = LGP.SqExpKernel(sk1_vars[begin])

println("Non-lazy GPR, stationary kernel, RKHS system solve: ")
gp_sk1 = LGP.fitGP(X, y, σ², sk1)
q = @benchmark LGP.fitGP($X, $y, $σ², $sk1)
sk_rkhs = median(q.times) / time_normalizing_constant
display(q)

println("Non-lazy GPR, stationary kernel, query mean and variance: ")
LGP.queryGP(xq_test, sk1, gp_sk1)
q = @benchmark LGP.queryGP($xq_test, $sk1, $gp_sk1)
sk_both = median(q.times) / time_normalizing_constant
display(q)

buf = zeros(T, length(X))
println("Non-lazy GPR, stationary kernel, query mean-only: ")
LGP.querymean(xq_test, sk1, gp_sk1)
q = @benchmark LGP.querymean($xq_test, $sk1, $gp_sk1)
sk_mean = median(q.times) / time_normalizing_constant
display(q)

println("Non-lazy GPR, stationary kernel, query variance-only: ")
LGP.queryGPvariance!(buf, xq_test, sk1, gp_sk1)
q = @benchmark LGP.queryGPvariance!($buf, $xq_test, $sk1, $gp_sk1)
sk_var = median(q.times) / time_normalizing_constant
display(q)


# ## DE kernel
kernel_param1, κ1 = dek1_vars[begin], dek1_vars[end]

canonical_kernel1 = LGP.WendlandSplineKernel(
    LGP.Order2(), kernel_param1, 3,
)
dek1 = LGP.DEKernel(canonical_kernel1, warpmap, κ1)

println("Non-lazy GPR, DE kernel, RKHS system solve: ")
gp_dek1 = LGP.fitGP(X, y, σ², dek1)
q = @benchmark LGP.fitGP($X, $y, $σ², $dek1)
dek_rkhs = median(q.times) / time_normalizing_constant
display(q)

println("Non-lazy GPR, DE kernel, query mean and variance: ")
LGP.queryGP(xq_test, dek1, gp_dek1)
q = @benchmark LGP.queryGP($xq_test, $dek1, $gp_dek1)
dek_both = median(q.times) / time_normalizing_constant
display(q)

buf = zeros(T, length(X))
println("Non-lazy GPR, DE kernel, query mean-only: ")
LGP.querymean(xq_test, dek1, gp_dek1)
q = @benchmark LGP.querymean($xq_test, $dek1, $gp_dek1)
dek_mean = median(q.times) / time_normalizing_constant
display(q)

println("Non-lazy GPR, DE kernel, query variance-only: ")
LGP.queryGPvariance!(buf, xq_test, dek1, gp_dek1)
q = @benchmark LGP.queryGPvariance!($buf, $xq_test, $dek1, $gp_dek1)
dek_var = median(q.times) / time_normalizing_constant
display(q)
println()

serialize(joinpath("results", "time_query_1_down_$(down_factor)"), (sk_rkhs, sk_both, sk_mean, sk_var, dek_rkhs, dek_both, dek_mean, dek_var))

nothing
