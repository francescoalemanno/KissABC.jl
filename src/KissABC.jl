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
    kwargs...,
)
    particles = [op(float, unconditional_sample(rng, model)) for i = 1:spl.nparticles]
    nparticles = length(particles)
    nparticles < length(model) + 5 && error(
        "nparticles = ",
        nparticles,
        " is insufficient, set number of particles in AIS(⋅) atleast to ",
        length(model) + 5,
    )

    logdensity = [loglike(model, push_p(model, particles[i])) for i = 1:spl.nparticles]

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

export sample, AIS, AISChain, MCMCThreads, MCMCDistributed
end
