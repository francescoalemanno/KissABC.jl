"""
    KissABC
Module to perform approximate bayesian computation,

Simple Example:
inferring the mean of a `Normal` distribution
```julia
using KissABC
using Distributions

prior=Normal(0,1)
data=randn(1000) .+ 1
sim(μ,other)=randn(1000) .+ μ
dist(x,y) = abs(mean(x) - mean(y))

plan=ABCplan(prior, sim, data, dist)
μ_post,Δ = ABCDE(plan, 1e-2)
@show mean(μ_post) ≈ 1.0
```

for more complicated code examples look at `https://github.com/francescoalemanno/KissABC.jl/`
"""
module KissABC

using Base.Threads
using Distributions
using Random

include("defn.jl")
include("ABCREJ.jl")
include("DE.jl")
include("SMCPR.jl")

export ABCplan, ABC, ABCSMCPR, ABCDE, Factored, sample_plan

end
