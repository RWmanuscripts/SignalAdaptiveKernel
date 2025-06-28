# In this demo, we go over an image-upconversion example. We fit the hyperparameters for a lazy-evaluation GPR, querying it to up-convert an image.

down_factor_inp = ARGS[1]
down_factor = parse(Int, down_factor_inp)

# downsample/up-conversion factor. # for non-command line use.
#down_factor = 2


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

using LinearAlgebra, Serialization

import Interpolations
import Metaheuristics as EVO
import Images
import LocalFilters
import SpatialGSP as GSP
import LazyGPR as LGP

import VisualizationBag as VIZ

import PythonPlot as PLT
PLT.close("all")
fig_num = 1

const T = Float64
const D = 2;

#increase to use distributed computing.
N_workers = 7
@assert N_workers > 1 #this script was designed for multi-process distributed computing.

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

# ## Setup distributed environments
using Distributed

worker_list = ones(Int, 1) #if this is the first time we're launching.
if nprocs() > 1
    worker_list = workers() #if we're re-using the existing workers.
end
if nworkers() < N_workers
    worker_list = LGP.create_local_procs(
        N_workers;
        pkg_load_list = [
            :(import Interpolations);
            :(import Metaheuristics);
        ],
        verbose = false,
    )
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

# kernel selection: we use a 3D, order 2 Wendland spline kernel.
ref_can_kernel = LGP.WendlandSplineKernel(
    LGP.Order2(), one(T), 3,
);

# Optimization tuning parameters.
max_abs_W = maximum(abs.(W))
κ_ub = convert(T, (b_x * 3) * 1 / max_abs_W)
V = im_y; # The variances for each local dataset.

#lazy-evaluation hyperparameter optimization config
lazy_hopt_config = LGP.LazyHCostConfig(
    worker_list, N_neighbourhoods, true, V,
);

#optimization algorithm config:
solver_config = LGP.MetaheuristicsConfig(
    f_calls_limit = f_calls_limit,
);

#optimization problem (e.g. bounds) config:
optim_config = LGP.KernelOptimConfig{T}( # optim_config
    a_lb = a_lb,
    a_ub = a_ub,
    κ_ub = κ_ub,
    width_factor = 1, # carry over sk.
);

# Fit the hyperparameters.
dek_vars, dek_score, sk_vars, sk_score = compute_kodak_hp(
    model, W, σr_factor, Xrs, ref_can_kernel,
    model_selection_trait,
    LGP.UseMetaheuristics(EVO),
    lazy_hopt_config,
    solver_config, optim_config,
);
sleep(3)
compute_kodak_hp_timing(
    model, W, σr_factor, Xrs, ref_can_kernel,
    model_selection_trait,
    LGP.UseMetaheuristics(EVO),
    lazy_hopt_config,
    solver_config, optim_config,
);
sleep(3)

# The gain κ and canonical kernel bandwidth a, for the DE kernel:
κ, a_DE = dek_vars

# The DE kernel's optimization solution's objective score:
dek_score

# The stationary kernel's bandwidth (should be the same as the canonical kernel if width_factor = 1 for KernelOptimConfig())
sk_vars

# The stationary kernel's optimization solution's objective score:
sk_score

# ## Timing for non-lazy evaluation GPR
# This implies the compute work is done only on a single process.
y = vec(im_y)
X = vec(collect(collect(x) for x in Iterators.product(Xrs...)))

println("Single GPR hyperparameter optimization")
gp_data = LGP.GPData(σ², X, y)

Random.seed!(25)
# sk1_vars, sk1_star = LGP.optimize_kernel_hp_separately_timing(
#     LGP.UseMetaheuristics(EVO),
#     ref_can_kernel,
#     model_selection_trait,
#     gp_data,
#     LGP.HCostConfig(),
#     solver_config,
#     optim_config,
# )
# @show sk1_vars, sk1_star
dek1_vars, dek1_score, sk1_vars, sk1_score = compute_kodak_hp(
    gp_data, W, σr_factor, Xrs, ref_can_kernel,
    model_selection_trait,
    LGP.UseMetaheuristics(EVO),
    LGP.HCostConfig(),
    solver_config, optim_config,
);
sleep(3)
compute_kodak_hp_timing(
    gp_data, W, σr_factor, Xrs, ref_can_kernel,
    model_selection_trait,
    LGP.UseMetaheuristics(EVO),
    LGP.HCostConfig(),
    solver_config, optim_config,
);
@show dek1_vars, dek1_score, sk1_vars, sk1_score
sleep(3)
println()

# # Save hyperparameters
save_path = joinpath("results", "timing_hp_single_downfactor_$(down_factor)")
serialize(
    save_path,
    (
        down_factor,
        dek_vars, dek_score, sk_vars, sk_score,
        dek1_vars, dek1_score, sk1_vars, sk1_score,
    ),
)

nothing
