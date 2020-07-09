module KissABC

import AbstractMCMC
import AbstractMCMC: sample, step, MCMCThreads, MCMCDistributed
using Distributions
using Random
import Base.length

include("priors.jl")
include("types.jl")
include("transition.jl")

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
"""
    AISChain(chains::NTuple{N,Vector})
    AISChain(chain::Vector) = AISChain((chain,))

this type is useful for gathering the results of `sample`,

# Example

```julia
genchain() = [ (rand((1,2,3)), randn()) for i in 1:100] # simple useless generator of chains
C=AISChain((genchain(),genchain(),genchain(),genchain())) # 4 chains
```

output:

```
Object of type AISChain (total samples 400)
number of samples: 100
number of parameters: 2
number of chains: 4
┌─────────┬────────────────────┬─────────────────────┬────────────────────┬────────────────────┬───────────────────┐
│         │               2.5% │               25.0% │              50.0% │              75.0% │             97.5% │
├─────────┼────────────────────┼─────────────────────┼────────────────────┼────────────────────┼───────────────────┤
│ Param 1 │                1.0 │                 1.0 │                2.0 │                3.0 │               3.0 │
│ Param 2 │ -1.830019623768731 │ -0.5840379394249825 │ 0.1015397387884777 │ 0.7357170602574647 │ 1.752198316412034 │
└─────────┴────────────────────┴─────────────────────┴────────────────────┴────────────────────┴───────────────────┘
```

individual samples can be accessed in an 3d-array like fashion:

```julia
C[1:90, 1, 1:2] # we are taking from the samples `1:90` of the chains `1:2`, only the parameter `1` 
```

output:

```
90×2 Array{Real,2}:
 2  1
 2  2
 2  3
 1  2
 1  1
 ⋮
 2  3
 1  3
 3  3
 2  2
```
"""
struct AISChain{T<:Union{Tuple,Vector}} <: AbstractArray{Real,3}
    samples::T
    AISChain(s::T) where {T} = new{T}(s)
end
include("printing.jl")
import Base: size, getindex, IndexStyle
size(x::AISChain{<:Vector}) = (length(x.samples), length(x.samples[1]), 1)
size(x::AISChain{<:Tuple}) =
    (length(x.samples[1]), length(x.samples[1][1]), length(x.samples))
@inline IndexStyle(::AISChain) = IndexCartesian()
getindex(x::AISChain{<:Tuple}, i::Int, j::Int, k::Int) = x.samples[k][i][j]
getindex(x::AISChain{<:Vector}, i::Int, j::Int, k::Int) = x.samples[i][j]

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AIS;
    burnin::Int = 0,
    retry_sampling::Int =100,
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
    retrys=retry_sampling*nparticles
    for i = 1:nparticles
        while !is_valid_logdensity(model,logdensity[i])
            particles[i]=op(float, unconditional_sample(rng, model))
            logdensity[i]=loglike(model, push_p(model, particles[i]))
            retrys-=1
            retrys < 0 && error("Prior leads to ∞ costs too often, tune the prior or increase `retry_sampling`.")
        end
    end

    for reps = 1:burnin, i = 1:nparticles
        transition!(model, particles, logdensity, i, rng)
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
    samples::Vector{<:Particle},
    ::AbstractMCMC.AbstractModel,
    ::AIS,
    ::Any,
    ::Type;
    kwargs...,
)
    return AISChain([samples[i].x for i in eachindex(samples)])
end

boxchain(a::AbstractVector) = (a,)
boxchain(a::Tuple) = a

function AbstractMCMC.chainscat(a::AISChain, b::AISChain)
    return AISChain((boxchain(a.samples)..., boxchain(b.samples)...))
end

"""
    sample(model, AIS(N), Ns[; optional args])
    sample(model, AIS(N), MCMCThreads(), Ns, Nc[; optional keyword args])
    sample(model, AIS(N), MCMCDistributed(), Ns, Nc[; optional keyword args])

# Generalities

This function will run an Affine Invariant MCMC sampler, and will return an `AISChain` object with all the parameter samples,
the mandatory parameters are:

`model`: a subtype of `AbstractDensity`, look at `ApproxPosterior`, `ApproxKernelizedPosterior`, `CommonLogDensity`.

`N`: number of particles in the ensemble, this particles will be evolved to generate new samples.

`Ns`: total number of samples which must be recorded.

`Nc`: total number of chains to run in parallel if MCMCThreads or MCMCDistributed is enabled.


the optional arguments available are:


`burnin`: number of mcmc steps per particle prior to saving any sample.

`ntransitions`: number of mcmc steps per particle between each sample.

`progress`: a boolean to disable verbosity

# Minimal Example for `CommonLogDensity`:

```julia
using KissABC
D = CommonLogDensity(
    2, #number of parameters
    rng -> randn(rng, 2), # initial sampling strategy
    x -> -100 * (x[1] - x[2]^2)^2 - (x[2] - 1)^2, # rosenbrock banana log-density
)
res = sample(D, AIS(50), 1000, ntransitions = 100, burnin = 500, progress = false)
println(res)
```

output:
```
number of samples: 1000
number of parameters: 2
number of chains: 1
┌─────────┬───────────────────────┬─────────────────────┬────────────────────┬───────────────────┬────────────────────┐
│         │                  2.5% │               25.0% │              50.0% │             75.0% │              97.5% │
├─────────┼───────────────────────┼─────────────────────┼────────────────────┼───────────────────┼────────────────────┤
│ Param 1 │ -0.025648264131516257 │  0.3219940894353638 │ 0.9721286048546971 │ 2.041743999647929 │ 5.6520319210700825 │
│ Param 2 │   -0.7226487177325958 │ 0.48611230863899335 │ 0.9604278578610763 │ 1.418519267388806 │  2.385312701114671 │
└─────────┴───────────────────────┴─────────────────────┴────────────────────┴───────────────────┴────────────────────┘
```

# Minimal Example for `ApproxKernelizedPosterior` (`ApproxPosterior`)

```julia
using KissABC, Distributions
prior = Uniform(-10, 10) # prior distribution for parameter
sim(μ) = μ + rand((randn() * 0.1, randn())) # simulator function
cost(x) = abs(sim(x) - 0.0) # cost function to compare simulations to target data, in this case simply '0'
plan = ApproxPosterior(prior, cost, 0.01) # Approximate model of log-posterior density (ABC)
#                                           ApproxKernelizedPosterior can be used in the same fashion
res = sample(plan, AIS(100), 2000, burnin = 10000, progress = false)
println(res)
```

output:
```
Object of type AISChain (total samples 2000)
number of samples: 2000
number of parameters: 1
number of chains: 1
┌─────────┬─────────────────────┬─────────────────────┬──────────────────────┬─────────────────────┬────────────────────┐
│         │                2.5% │               25.0% │                50.0% │               75.0% │              97.5% │
├─────────┼─────────────────────┼─────────────────────┼──────────────────────┼─────────────────────┼────────────────────┤
│ Param 1 │ -1.8692526364678272 │ -0.1390312701192662 │ 0.023740627510271367 │ 0.20440332127121577 │ 1.1844931848164661 │
└─────────┴─────────────────────┴─────────────────────┴──────────────────────┴─────────────────────┴────────────────────┘
```

"""
sample

export sample, AIS, AISChain, MCMCThreads, MCMCDistributed
end
