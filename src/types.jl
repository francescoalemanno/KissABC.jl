abstract type AbstractDensity <: AbstractMCMC.AbstractModel end
abstract type AbstractApproxPosterior <: AbstractDensity end
#=
unconditional_sample(rng::AbstractRNG,density::AbstractDensity) = error("define a method to cover unconditional_sample(rng::AbstractRNG,density::"*repr(typeof(density))*").")
loglike(density::AbstractDensity,sample) = error("define a method to cover logpdf(density::"*repr(typeof(density))*",sample). must return a named tuple (logprior = ?, loglikelihood = ?)")
length(density::AbstractDensity) = error("define a method to cover length(density::"*repr(typeof(density))*").")
accept(density::AbstractDensity,rng::AbstractRNG,old_ld,new_ld) = error("define a method to cover accept(density::"*repr(typeof(density))*",rng::AbstractRNG,old_ld,new_ld). must return boolean to accept or reject a transition from old_ld → new_ld")
=#

struct Particle{Xt}
    x::Xt
    Particle(x::T) where {T} = new{T}(x)
end

op(f, a::Particle, b::Particle) = Particle(op(f, a.x, b.x))
op(f, a::Particle, b::Number) = Particle(op(f, a.x, b))
op(f, a::Number, b::Particle) = Particle(op(f, a, b.x))
op(f, a::Number, b::Number) = f(a, b)
op(f, a, b) = op.(Ref(f), a, b)

op(f, a::Particle) = Particle(op(f, a.x))
op(f, a::Number) = f(a)
op(f, a) = op.(Ref(f), a)

op(f, args...) = foldl((x, y) -> op(f, x, y), args)

push_p(density::AbstractDensity, p::Particle) = p
push_p(density::AbstractApproxPosterior, p::Particle) = Particle(push_p(density.prior, p.x))
push_p(density::Factored, p) = push_p.(density.p, p)
push_p(density::Distribution, p) = push_p.(Ref(density), p)
push_p(density::ContinuousDistribution, p::Number) = float(p)
push_p(density::DiscreteDistribution, p::Number) = round(Int, p)

unconditional_sample(rng::AbstractRNG, density::AbstractApproxPosterior) =
    Particle(rand(rng, density.prior))

length(density::AbstractApproxPosterior) = length(density.prior)


struct ApproxKernelizedPosterior{P<:Distribution,C,S<:Real} <: AbstractApproxPosterior
    prior::P
    cost::C
    scale::S
    ApproxKernelizedPosterior(
        prior::T1,
        cost::T2,
        target_average_cost::T3,
    ) where {T1,T2,T3} = new{T1,T2,T3}(prior, cost, target_average_cost)
end

function loglike(density::ApproxKernelizedPosterior, sample::Particle)
    lp = logpdf(density.prior, sample.x)
    ll = lp
    if isfinite(lp)
        ll = -0.5 * abs2(density.cost(sample.x) / density.scale)
    end
    (logprior = lp, loglikelihood = ll)
end

is_valid_logdensity(density::ApproxKernelizedPosterior, ld) = isfinite(sum(ld))

function accept(
    density::ApproxKernelizedPosterior,
    rng::AbstractRNG,
    old_ld,
    new_ld,
    ld_correction,
)
    isfinite(ld_correction) || error("ld_correction is invalid")
    is_valid_logdensity(density, old_ld) || error("starting sample invalid.")
    is_valid_logdensity(density, new_ld) || return false

    lW = ld_correction + sum(new_ld) - sum(old_ld)
    return -randexp(rng) <= lW
end
struct ApproxPosterior{P<:Distribution,C,S<:Real} <: AbstractApproxPosterior
    prior::P
    cost::C
    maxcost::S
    ApproxPosterior(prior::T1, cost::T2, max_cost::T3) where {T1,T2,T3} =
        new{T1,T2,T3}(prior, cost, max_cost)
end

function loglike(density::ApproxPosterior, sample::Particle)
    lp = logpdf(density.prior, sample.x)
    cs = -lp
    if isfinite(lp)
        cs = density.cost(sample.x)
    end
    (logprior = lp, cost = cs)
end

is_valid_logdensity(density::ApproxPosterior, ld) =
    isfinite(ld.cost) && isfinite(ld.logprior)

function accept(density::ApproxPosterior, rng::AbstractRNG, old_ld, new_ld, ld_correction)
    isfinite(ld_correction) || error("ld_correction is invalid")
    is_valid_logdensity(density, old_ld) || error("starting sample invalid.")
    is_valid_logdensity(density, new_ld) || return false

    lW = ld_correction + new_ld.logprior - old_ld.logprior
    lW2 = max(density.maxcost, old_ld.cost) - new_ld.cost
    (-randexp(rng) <= lW) && lW2 >= 0
end
struct CommonLogDensity{N,A,B} <: AbstractDensity
    sample_init::A
    lπ::B
    CommonLogDensity(nparameters::Int, sample_init::A, lπ::B) where {A,B} =
        new{nparameters,A,B}(sample_init, lπ)
end

unconditional_sample(rng::AbstractRNG, density::CommonLogDensity) =
    Particle(density.sample_init(rng))

length(density::CommonLogDensity{N}) where {N} = N

function loglike(density::CommonLogDensity, sample::Particle)
    density.lπ(sample.x)
end

is_valid_logdensity(density::CommonLogDensity, ld) = isfinite(ld)

function accept(density::CommonLogDensity, rng::AbstractRNG, old_ld, new_ld, ld_correction)
    isfinite(ld_correction) || error("ld_correction is invalid")
    is_valid_logdensity(density, old_ld) || error("starting sample invalid.")
    is_valid_logdensity(density, new_ld) || return false
    return -randexp(rng) <= ld_correction + new_ld - old_ld
end

"""
    ApproxKernelizedPosterior(
        prior::Distribution,
        cost::Function,
        target_average_cost::Real
    )
this function will return a type which can be used in the `sample` function as an ABC density,
this type works by assuming Gaussianly distributed errors 𝒩(0,ϵ), ϵ is specified in the variable `target_average_cost`.
"""
ApproxKernelizedPosterior
"""
    ApproxPosterior(
        prior::Distribution,
        cost::Function,
        max_cost::Real
    )
this function will return a type which can be used in the `sample` function as an ABC density,
this type works by assuming uniformly distributed errors in [-ϵ,ϵ], ϵ is specified in the variable `max_cost`.
"""
ApproxPosterior

"""
    CommonLogDensity(nparameters, sample_init, lπ)
this function will return a type for performing classical MCMC via the `sample` function.

`nparameters`: total number of parameters per sample.

`sample_init`: function which accepts an `RNG::AbstractRNG` and returns a sample for `lπ`.

`lπ`: function which accepts a sample, and returns a log-density float value.
"""
CommonLogDensity

export CommonLogDensity, ApproxPosterior, ApproxKernelizedPosterior
