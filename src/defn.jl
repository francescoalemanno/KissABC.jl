
"""
    ABCplan(prior, simulation, data, distance; params=())

Builds a type `ABCplan` which holds

# Arguments:
- `prior`: a `Distribution` to use for sampling candidate parameters
- `simulation`: simulation function `sim(prior_sample, constants) -> data` that accepts a prior sample and the `params` constant and returns a simulated dataset
- `data`: target dataset which must be compared with simulated datasets
- `distance`: distance function `dist(x,y)` that return the distance (a scalar value) between `x` and `y`
- `params`: an optional set of constants to be passed as second argument to the simulation function
"""
struct ABCplan{T1,T2,T3,T4,T5}
    prior::T1
    simulation::T2
    data::T3
    distance::T4
    params::T5
    ABCplan(prior::T1,simulation::T2,data::T3,distance::T4;params::T5=()) where {T1,T2,T3,T4,T5} = new{T1,T2,T3,T4,T5}(prior,simulation,data,distance,params)
end

macro cthreads(condition::Symbol,loop) #does not work well because of #15276, but seems to work on Julia v0.7
    return esc(quote
        if $condition
            Threads.@threads $loop
        else
            $loop
        end
    end)
end

macro extract_params(S,params...)
    c=:()
    for p in params
        c=quote
            $c
            $p = $S.$p
        end
    end
    esc(c)
end

import Distributions.pdf, Random.rand, Base.length
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
pdf(d::Factored,x) = prod(i->pdf(d.p[i],x[i]),eachindex(x))

"""
    rand(rng::AbstractRNG, factoreddist::Factored)

function to sample one element from a `Factored` object
"""
rand(rng::AbstractRNG,factoreddist::Factored) = rand.(Ref(rng),factoreddist.p)

"""
    length(p::Factored) = begin

returns the number of distributions contained in `p`.
"""
length(p::Factored) = sum(length.(p.p))
