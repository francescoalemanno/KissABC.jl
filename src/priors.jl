import Distributions.pdf, Distributions.logpdf, Random.rand, Base.length

struct MixedSupport <: ValueSupport; end

"""
    Factored{N} <: Distribution{Multivariate, MixedSupport}
a `Distribution` type that can be used to combine multiple `UnivariateDistribution`'s and sample from them.
Example: it can be used as `prior = Factored(Normal(0,1), Uniform(-1,1))`
"""
struct Factored{N}<:Distribution{Multivariate,MixedSupport}
    p::NTuple{N,UnivariateDistribution}
    Factored(args::UnivariateDistribution...) = new{length(args)}(args)
end
"""
    pdf(d::Factored, x) = begin
Function to evaluate the pdf of a `Factored` distribution object
"""
function pdf(d::Factored{N},x) where N
    s=pdf(d.p[1],x[1])
    for i in 2:N
        s*=pdf(d.p[i],x[i])
    end
    s
end

"""
    logpdf(d::Factored, x) = begin
Function to evaluate the logpdf of a `Factored` distribution object
"""
function logpdf(d::Factored{N},x) where N
    s=logpdf(d.p[1],x[1])
    for i in 2:N
        s+=logpdf(d.p[i],x[i])
    end
    s
end

"""
    rand(rng::AbstractRNG, factoreddist::Factored)
function to sample one element from a `Factored` object
"""
rand(rng::AbstractRNG,factoreddist::Factored{N}) where N = ntuple(i->rand(rng,factoreddist.p[i]),Val(N))

"""
    length(p::Factored) = begin
returns the number of distributions contained in `p`.
"""
length(p::Factored{N}) where N = N

densitytypes(S::Factored) = densitytypes.(S.p)

export Factored
