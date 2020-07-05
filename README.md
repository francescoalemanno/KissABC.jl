# KissABC

![CI](https://github.com/JuliaApproxInference/KissABC.jl/workflows/CI/badge.svg?branch=master)
[![Coverage](http://codecov.io/github/JuliaApproxInference/KissABC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproxInference/KissABC.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaapproxinference.github.io/KissABC.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaapproxinference.github.io/KissABC.jl/stable/)

Table of Contents
=================

  * [Beginners Usage Guide](#usage-guide)

## Usage guide

The ingredients you need to use Approximate Bayesian Computation:

1. A simulation which depends on some parameters, able to generate datasets similar to your target dataset if parameters are tuned
1. A prior distribution over such parameters
1. A distance function to compare generated dataset to the true dataset

We will start with a simple example, we have a dataset generated according to an Normal distribution whose parameters are unknown
```julia
tdata=randn(1000).*0.04.+2
```
we are ofcourse able to simulate normal random numbers, so this constitutes our simulation
```julia
sim((μ,σ)) = randn(1000) .* σ .+ μ
```
The second ingredient is a prior over the parameters μ and σ
```julia
using Distributions
using KissABC
prior=Factored(Uniform(1,3), Truncated(Normal(0,0.1), 0, 100))
```
we have chosen a uniform distribution over the interval [1,3] for μ and a normal distribution truncated over ℝ⁺ for σ.

Now all that we need is a distance function to compare the true dataset to the simulated dataset, for this purpose comparing mean and variance is optimal,
```julia
function cost((μ,σ)) 
    x=sim((μ,σ))
    y=tdata
    d1 = mean(x) - mean(y)
    d2 = std(x) - std(y)
    hypot(d1, d2 * 50)
end
```
Now we are all set, we can use `AIS` which is an Affine Invariant MC algorithm via the `sample` function, to simulate the posterior distribution for this model, inferring `μ` and `σ`
```julia
approx_density = ApproxKernelizedPosterior(prior,cost,0.005)
res = sample(approx_density,AIS(10),1000,ntransitions=100)
```

the repl output is:

```TTY
Sampling 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
Object of type AISChain (total samples 1000)
number of samples: 1000
number of parameters: 2
number of chains: 1
┌─────────┬─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┬──────────────────────┐
│         │                2.5% │               25.0% │               50.0% │               75.0% │                97.5% │
├─────────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤
│ Param 1 │  1.9889104510710525 │  1.9958478705688778 │   1.999126578901652 │   2.002886778724749 │   2.0095096952698044 │
│ Param 2 │ 0.03864543664435372 │ 0.03979188808422054 │ 0.04040515336501156 │ 0.04103079769548673 │ 0.042449787918102805 │
└─────────┴─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┴──────────────────────┘
```

We chose a tolerance on distances equal to `0.005`, a number of particles equal to `10`, we chose a number of steps per sample `ntransitions = 100` and we acquired `1000` samples.
For comparison let's extract some prior samples
```julia
prsample=[rand(prior) for i in 1:5000] #some samples from the prior for comparison
```
plotting prior and posterior side by side we get:

![plots of the Inference Results](images/inf_normaldist.png "Inference Results")

we can see that the algorithm has correctly inferred both parameters, this exact recipe will work for much more complicated models and simulations, with some tuning.
