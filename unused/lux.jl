# based on https://lux.csail.mit.edu/stable/tutorials/beginner/2_PolynomialFitting

using Lux, ADTypes, Optimisers, Printf, Random, Reactant, Statistics

function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end

const T = Float64
rng = MersenneTwister()
Random.seed!(rng, 12345)

function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end

#(x, y) = generate_data(rng)
# use x1s, x2s, x3s
#x = [ convert(Vector{T}, x1s .- 445550); convert(Vector{T}, x2s .- 257534) ]
#x = [ convert(Vector{T}, x1s)'; convert(Vector{T}, x2s)' ]
#y = convert(Vector{T}, x3s)'

x = [ x1s'; x2s' ]
y = Matrix{Float32}(x3s')

D = size(x, 1)
model = Chain(
    Dense(D => 256, relu),
    Dense(256 => 256, relu),
    Dense(256 => 256, relu),
    Dense(256 => 64, relu),
    Dense(64 => 1),
)


opt = Adam(0.03f0)
const loss_function = MSELoss()
const cdev = cpu_device()
const xdev = reactant_device()

ps, st = xdev(Lux.setup(rng, model))
tstate = Training.TrainState(model, ps, st, opt)
vjp_rule = AutoEnzyme()

function main(tstate::Training.TrainState, vjp, data, epochs)
    data = xdev(data)
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 50_000)
forward_pass = @compile Lux.apply(
    tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)
)

y_pred = cdev(
    first(
        forward_pass(
            tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states),
        ),
    ),
)

#f2 = Figure(size = (900, 650))

function prep_makie(x, y_pred)

    vec_tuples = collect(
        (x[1, j], x[2, j], y_pred[j]) for j in eachindex(axes(x, 2), y_pred)
    )

    return Point3f.(vec_tuples)
end
ps2 = prep_makie(x, y_pred)

# lb, ub = minimum(x3s), maximum(x3s)
# cs = convertcompactdomain(x3s, lb, ub, Float32(0), Float32(1))
# cs = rand(length(x3s))

Label(f[1, 2], "depthsorting = true1, pred", tellwidth = false)
scatter(f[2, 2], ps2, markersize = 5, depthsorting = true)
f
