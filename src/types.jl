abstract type AbstractApproxDensity end
#=
unconditional_sample(rng::AbstractRNG,density::AbstractApproxDensity) = error("define a method to cover unconditional_sample(rng::AbstractRNG,density::"*repr(typeof(density))*").")
loglike(density::AbstractApproxDensity,sample) = error("define a method to cover logpdf(density::"*repr(typeof(density))*",sample). must return a named tuple (logprior = ?, loglikelihood = ?)")
length(density::AbstractApproxDensity) = error("define a method to cover length(density::"*repr(typeof(density))*").")
accept(density::AbstractApproxDensity,rng::AbstractRNG,old_ld,new_ld) = error("define a method to cover accept(density::"*repr(typeof(density))*",rng::AbstractRNG,old_ld,new_ld). must return boolean to accept or reject a transition from old_ld → new_ld")
=#

abstract type AbstractPerturbator end
struct Perturbator <: AbstractPerturbator
    rng::AbstractRNG
end

function (a::Perturbator)()
    randn(a.rng)
end

@inline handleperturbator(x) = x
@inline handleperturbator(x::AbstractPerturbator) = x()
@inline boxperturbator(x::AbstractPerturbator) = Ref(x)
@inline boxperturbator(x) = x
const MultiLike = Union{AbstractPerturbator,Tuple,Array,Number}
const SingleLike = Union{AbstractPerturbator,Number}

for op in (:+, :-, :*, :/)
    opp = Symbol(op, "′")
    @eval begin
        $opp(a::A, b::B) where {A<:MultiLike,B<:MultiLike} =
            $opp.(boxperturbator(a), boxperturbator(b))
        $opp(a::A, b::B) where {A<:SingleLike,B<:SingleLike} =
            $op(handleperturbator(a), handleperturbator(b))
    end
end

triangle_abs(a::T, b::T, c::T) where {T<:SingleLike} =
    (abs(a - b) + abs(c - b) + abs(c - a)) / 3
triangle_abs(a::T, b::T, c::T) where {T<:MultiLike} = triangle_abs.(a, b, c)


sqrt′(a::T) where {T<:SingleLike} = sqrt(a)
sqrt′(a::T) where {T<:MultiLike} = sqrt′.(a)

abstract type AbstractApproxPosterior <: AbstractApproxDensity end
unconditional_sample(rng::AbstractRNG, density::AbstractApproxPosterior) =
    rand(rng, density.prior)
length(density::AbstractApproxPosterior) = length(density.prior)


densitytypes(S::AbstractApproxPosterior) = densitytypes(S.prior)
densitytypes(S::ContinuousDistribution) = Float64
densitytypes(S::DiscreteDistribution) = Int64

floatize(S) = floatize.(S)
floatize(S::Number) = convert(Float64, S)

tostartingsupport(S, A::T) where {T<:MultiLike} = tostartingsupport.(S, A)
tostartingsupport(S::Type{<:Integer}, A::T) where {T<:SingleLike} = round(S, A)
tostartingsupport(S::Type{<:AbstractFloat}, A::T) where {T<:SingleLike} = A

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

function loglike(density::ApproxKernelizedPosterior, sample)
    lp = logpdf(density.prior, sample)
    isfinite(lp) || return (logprior = lp, loglikelihood = lp)
    (logprior = lp, loglikelihood = -0.5 * abs2(density.cost(sample) / density.scale))
end

accept(
    density::ApproxKernelizedPosterior,
    rng::AbstractRNG,
    old_ld,
    new_ld,
    ld_correction,
) = -randexp(rng) <= ld_correction + sum(new_ld) - sum(old_ld)

struct ApproxPosterior{P<:Distribution,C,S<:Real} <: AbstractApproxPosterior
    prior::P
    cost::C
    maxcost::S
    ApproxPosterior(prior::T1, cost::T2, max_cost::T3) where {T1,T2,T3} =
        new{T1,T2,T3}(prior, cost, max_cost)
end

function loglike(density::ApproxPosterior, sample)
    lp = logpdf(density.prior, sample)
    isfinite(lp) || return (logprior = lp, cost = -lp)
    (logprior = lp, cost = density.cost(sample))
end

accept(density::ApproxPosterior, rng::AbstractRNG, old_ld, new_ld, ld_correction) =
    (-randexp(rng) <= ld_correction + new_ld.logprior - old_ld.logprior) &&
    new_ld.cost <= max(density.maxcost, old_ld.cost)


"""
    ApproxKernelizedPosterior(
        prior::Distribution,
        cost::Function,
        target_average_cost::Real
    )
this function will return a type which can be used in the `mcmc` function as an ABC density,
this type works by assuming Gaussianly distributed errors 𝒩(0,ϵ), ϵ is specified in the variable `target_average_cost`.
"""
ApproxKernelizedPosterior
"""
    ApproxPosterior(
        prior::Distribution,
        cost::Function,
        max_cost::Real
    )
this function will return a type which can be used in the `mcmc` function as an ABC density,
this type works by assuming uniformly distributed errors in [-ϵ,ϵ], ϵ is specified in the variable `max_cost`.
"""
ApproxKernelizedPosterior

export ApproxPosterior, ApproxKernelizedPosterior
