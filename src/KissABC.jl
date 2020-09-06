module KissABC

macro reexport(ex) # taken from Reexport.jl
    modules = Any[e.args[end] for e in ex.args]
    esc(Expr(:toplevel, ex, [:(eval(Expr(:export, names($mod)...))) for mod in modules]...))
end

import AbstractMCMC
import AbstractMCMC: sample, step, MCMCThreads, MCMCDistributed
using Random
import Base.length

@reexport using MonteCarloMeasurements
@reexport using Distributions

include("priors.jl")
include("types.jl")
include("transition.jl")
include("smc.jl")

struct AIS <: AbstractMCMC.AbstractSampler
    nparticles::Int
end

struct AISState{S,L}
    "Sample of the Affine Invariant Sampler."
    sample::S
    "Log-likelihood of the sample."
    loglikelihood::L
    "Current particle"
    i::Int
    AISState(s::S, l::L, i = 1) where {S,L} = new{S,L}(s, l, i)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AIS;
    retry_sampling::Int = 100,
    kwargs...,
)
    nparticles = spl.nparticles
    nparticles < length(model) + 5 && error(
        "nparticles = ",
        nparticles,
        " is insufficient, set number of particles in AIS(⋅) atleast to ",
        length(model) + 5,
    )

    particles = [op(float, unconditional_sample(rng, model)) for i = 1:nparticles]
    logdensity = [loglike(model, push_p(model, particles[i])) for i = 1:nparticles]
    retrys = retry_sampling * nparticles
    for i = 1:nparticles
        while !is_valid_logdensity(model, logdensity[i])
            particles[i] = op(float, unconditional_sample(rng, model))
            logdensity[i] = loglike(model, push_p(model, particles[i]))
            retrys -= 1
            retrys < 0 &&
                error("Prior leads to ∞ costs too often, tune the prior or increase `retry_sampling`.")
        end
    end

    push_p(model, particles[end]), AISState(particles, logdensity)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AIS,
    state::AISState;
    ntransitions::Int = 1,
    kwargs...,
)
    i = state.i
    for reps = 1:ntransitions
        transition!(model, state.sample, state.loglikelihood, i, rng)
    end
    push_p(model, state.sample[i]),
    AISState(state.sample, state.loglikelihood, 1 + (i % spl.nparticles))
end

function AbstractMCMC.bundle_samples(
    samples::Vector{T},
    ::AbstractMCMC.AbstractModel,
    ::AIS,
    ::Any,
    ::Type;
    kwargs...,
) where {T<:Particle}
    l = length(samples[1].x)
    P = map(x -> Particles(x), getindex.(getfield.(samples, :x), i) for i = 1:l)
    length(P) == 1 && return P[1]
    return P
end

function AbstractMCMC.chainsstack(c::Vector{Vector{T}}) where {T<:Particles}
    nc = length(c)
    pl = length(c[1])
    return [Particles(reduce(vcat, c[n][i].particles for n = 1:nc)) for i = 1:pl]
end

function AbstractMCMC.chainsstack(c::Vector{T}) where {T<:Particles}
    return Particles(reduce(vcat, c[i].particles for i in eachindex(c)))
end

"""
    sample(model, AIS(N), Ns[; optional args])
    sample(model, AIS(N), MCMCThreads(), Ns, Nc[; optional keyword args])
    sample(model, AIS(N), MCMCDistributed(), Ns, Nc[; optional keyword args])

# Generalities

This function will run an Affine Invariant MCMC sampler, and will return an `Particles` object for each parameter,
the mandatory parameters are:

`model`: a subtype of `AbstractDensity`, look at `ApproxPosterior`, `ApproxKernelizedPosterior`, `CommonLogDensity`.

`N`: number of particles in the ensemble, this particles will be evolved to generate new samples.

`Ns`: total number of samples which must be recorded.

`Nc`: total number of chains to run in parallel if MCMCThreads or MCMCDistributed is enabled.


the optional arguments available are:


`discard_initial`: number of mcmc particles to discard before saving any sample.

`ntransitions`: number of mcmc steps per particle between each sample.

`retry_sampling`: number of maximum attempts to resample an initial particle whose cost (or log-density) is ±∞ or NaN.

`progress`: a boolean to disable verbosity

# Minimal Example for `CommonLogDensity`:

```julia
using KissABC
D = CommonLogDensity(
    2, #number of parameters
    rng -> randn(rng, 2), # initial sampling strategy
    x -> -100 * (x[1] - x[2]^2)^2 - (x[2] - 1)^2, # rosenbrock banana log-density
)
res = sample(D, AIS(50), 1000, ntransitions = 100, discard_initial = 500, progress = false)
println(res)
```

output:
```
Particles{Float64,1000}[1.43 ± 1.4, 0.99 ± 0.67]
```

# Minimal Example for `ApproxKernelizedPosterior` (`ApproxPosterior`)

```julia
using KissABC
prior = Uniform(-10, 10) # prior distribution for parameter
sim(μ) = μ + rand((randn() * 0.1, randn())) # simulator function
cost(x) = abs(sim(x) - 0.0) # cost function to compare simulations to target data, in this case simply '0'
plan = ApproxPosterior(prior, cost, 0.01) # Approximate model of log-posterior density (ABC)
#                                           ApproxKernelizedPosterior can be used in the same fashion
res = sample(plan, AIS(100), 2000, discard_initial = 10000, progress = false)
println(res)
```

output:
```
0.0 ± 0.46
```

"""
sample

export sample, AIS, MCMCThreads, MCMCDistributed
end
