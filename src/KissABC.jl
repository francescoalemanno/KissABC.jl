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

struct AISChain{T<:Union{Tuple,Vector}} <: AbstractArray{Real,3}
    samples::T
    AISChain(s::T) where {T} = new{T}(s)
end

import Base: size, getindex, IndexStyle
size(x::AISChain{<:Vector}) = (length(x.samples),length(x.samples[1]),1)
size(x::AISChain{<:Tuple}) = (length(x.samples[1]),length(x.samples[1][1]),length(x.samples))
@inline IndexStyle(::AISChain) = IndexCartesian()
getindex(x::AISChain{<:Tuple},i::Int,j::Int,k::Int) = x.samples[k][i][j]
getindex(x::AISChain{<:Vector},i::Int,j::Int,k::Int) = x.samples[i][j]

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AIS;
    burnin::Int = 0,
    kwargs...,
)
    particles = [op(float, unconditional_sample(rng, model)) for i = 1:spl.nparticles]
    nparticles = length(particles)
    nparticles < 2 * length(model) + 10 && error(
        "nparticles = ",
        nparticles,
        " is insufficient, set number of particles in AIS(⋅) atleast to ",
        2 * length(model) + 10,
    )

    logdensity = [loglike(model, push_p(model, particles[i])) for i = 1:spl.nparticles]

    sep = nparticles ÷ 2
    ensembles = ((1:sep, (sep+1):nparticles), ((sep+1):nparticles, 1:sep))
    for reps = 1:burnin, (active, inactive) in ensembles
        for i in active
            transition!(model, particles, logdensity, inactive, i, rng)
        end
    end
    push_p(model, particles[end]), AISState(particles, logdensity)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AIS,
    state::AISState;
    kwargs...,
)
    i = state.i
    sep = spl.nparticles ÷ 2
    inactives = (1:sep, (sep+1):spl.nparticles)
    inactive = ifelse(i <= sep, inactives[2], inactives[1])
    transition!(model, state.sample, state.loglikelihood, inactive, i, rng)
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
function AbstractMCMC.chainscat(a::AISChain{<:Vector}, b::AISChain{<:Vector})
    return AISChain((a.samples, b.samples))
end
function AbstractMCMC.chainscat(a::AISChain{<:Vector}, b::AISChain{<:Tuple})
    return AISChain((a.samples, b.samples...))
end
function AbstractMCMC.chainscat(a::AISChain{<:Tuple}, b::AISChain{<:Vector})
    return AISChain((a.samples..., b.samples))
end
function AbstractMCMC.chainscat(a::AISChain{<:Tuple}, b::AISChain{<:Tuple})
    return AISChain((a.samples..., b.samples...))
end

export sample, AIS, AISChain, MCMCThreads, MCMCDistributed
end
