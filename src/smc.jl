
macro cthreads(condition::Symbol, loop) #does not work well because of #15276, but seems to work on Julia v0.7
    return esc(quote
        ($condition) && (Threads.@threads $loop; true) || ($loop; true)
    end)
end

function ess(w)
    sum(w)^2 / sum(abs2, w)
end

function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer) # taken from Turing.jl
    # Pre-allocate array for resampled particles
    indices = Vector{Int}(undef, num_particles)

    # deterministic assignment
    residuals = similar(w)
    i = 1
    @inbounds for j = 1:length(w)
        x = num_particles * w[j]
        floor_x = floor(Int, x)
        for k = 1:floor_x
            indices[i] = j
            i += 1
        end
        residuals[j] = x - floor_x
    end

    # sampling from residuals
    if i <= num_particles
        residuals ./= sum(residuals)
        rand!(Categorical(residuals), view(indices, i:num_particles))
    end

    return indices
end

"""
Adaptive SMC from P. Del Moral 2012, with Affine invariant proposal mechanism, faster that `AIS` for `ABC` targets.
```julia
function smc(
    prior::Distribution,
    cost::Function;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    nparticles::Int = 100,
    M::Int = 1,
    alpha = 0.95,
    mcmc_retrys::Int = 0,
    mcmc_tol = 0.015,
    epstol = 0.0,
    r_epstol = (1 - alpha) / 50,
    min_r_ess = 0.55,
    verbose::Bool = false,
    parallel::Bool = false,
)
```

- `prior`: a Distribution object representing the parameters prior.
- `cost`: a function that given a prior sample returns the cost for said sample (e.g. a distance between simulated data and target data).
- `rng`: an AbstractRNG object which is used by SMC for inference (it can be useful to make an inference reproducible).
- `nparticles`: number of total particles to use for inference.
- `M`: number of cost evaluations per particle, increasing this can reduce the chance of rejecting good particles. 
- `alpha` - used for adaptive tolerance, by solving `ESS(n,ϵ(n)) = α ESS(n-1, ϵ(n-1))` for `ϵ(n)` at step `n`.
- `mcmc_retrys` - if set > 0, whenever the fraction of accepted particles drops below the tolerance `mcmc_tol` the MCMC step is repeated (no more than `mcmc_retrys` times).
- `mcmc_tol` - stopping condition for SMC, if the fraction of accepted particles drops below `mcmc_tol` the algorithm terminates.
- `epstol` - stopping condition for SMC, if the adaptive cost threshold drops below `epstol` the algorithm has converged and thus it terminates.
- `min_r_ess` - whenever the fractional effective sample size drops below `min_r_ess`, a systematic resampling step is performed.
- `verbose` - if set to `true`, enables verbosity.
- `parallel` - if set to `true`, threaded parallelism is enabled, keep in mind that the cost function must be Thread-safe in such case.

# Example

```Julia
using KissABC
prior=Factored(Normal(0,5), Normal(0,5))
cost((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2
results = smc(prior, cost, alpha=0.5, nparticles=5000).P
```

output:
```TTY
2-element Array{Particles{Float64,5000},1}:
 1.0 ± 0.029
 0.999 ± 0.012
```
"""
function smc(
    prior::Tprior,
    cost;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    nparticles::Int = 100,
    M::Int = 1,
    alpha = 0.95,
    mcmc_retrys::Int = 0,
    mcmc_tol = 0.015,
    epstol = 0.0,
    r_epstol = (1 - alpha) / 50,
    min_r_ess = 0.55,
    verbose::Bool = false,
    parallel::Bool = false,
) where {Tprior<:Distribution}
    M >= 1 || error("M must be >= 1.")
    min_r_ess > 0 || error("min_r_ess must be > 0.")
    mcmc_retrys >= 0 || error("mcmc_retrys must be >= 0.")
    alpha > 0 || error("alpha must be > 0.")
    r_epstol >= 0 || error("r_epstol must be >= 0")
    Np=length(prior)
    min_nparticles = ceil(
        Int,
        1.5 * (1 + ifelse(parallel, 1, 0)) * Np / (min(alpha, min_r_ess)),
    )
    nparticles >= min_nparticles || error("nparticles must be >= $min_nparticles.")
    θs = [op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
    Xs = parallel ?
        fetch.([
        Threads.@spawn cost(push_p(prior, θs[$i].x)) for i = 1:nparticles, m = 1:M
    ]) :
        [cost(push_p(prior, θs[i].x)) for i = 1:nparticles, m = 1:M]

    lπs = [logpdf(prior, push_p(prior, θs[i].x)) for i = 1:nparticles]
    good = -Inf .< Xs .< Inf
    ϵ = maximum(Xs[good])
    Ws = collect(vec(sum(good, dims = 2)))
    Ws = Ws ./ sum(Ws)
    Ia = collect(vec(sum(x -> (x <= ϵ) && (-Inf<x<Inf), Xs, dims = 2)))
    ESS = ess(Ws)
    α = alpha
    iteration = 0
    # Step 1 - adaptive threshold
    while true
        iteration += 1
        ϵv = ϵ
        let
            tol = 1 / (4nparticles)
            target = α * ESS
            rϵ = (minimum(Xs), ϵ)
            p = 0.5
            Δ = 0.25
            while true
                ϵn = rϵ[1] * p + rϵ[2] * (1 - p)
                Ian = vec(sum(x -> x <= ϵn, Xs, dims = 2))
                Wsn = Ws .* (Ian) ./ (Ia .+ 1e-15)
                dest = ess(Wsn)
                if dest <= target
                    p -= Δ
                else
                    p += Δ
                end
                Δ /= 2
                if Δ <= tol
                    Ia = collect(Ian)
                    Ws = Wsn ./ sum(Wsn)
                    ϵ = ϵn
                    ESS = dest
                    verbose && (@show iteration, ϵ, dest, target)
                    break
                end
            end
        end

        # Step 2 - Resampling
        if ESS * α <= nparticles * min_r_ess
            idx = resample_residual(Ws, nparticles)
            θs = θs[idx]
            Xs = Xs[idx, :]
            lπs = lπs[idx]
            Ia = Ia[idx]
            Ws .= 1 / nparticles
            ESS = nparticles
        end

        # Step 3 - MCMC
        accepted = parallel ? Threads.Atomic{Int}(0) : 0
        retry_N = 1 + mcmc_retrys

        widx = (1:nparticles)[Ws .> 0]
        L = length(widx)
        cut = L ÷ 2
        s1 = widx[1:cut]
        s2 = widx[(cut+1):L]

        for r = 1:retry_N
            for (A,B) in ((s1,s2),(s2,s1))
                new_p = map(A) do i
                    a = rand(rng,B)
                    Z = sample_g(rng, 2.0)
                    W = op(*, op(-, θs[i], θs[a]), Z)
                    (log(rand(rng)), op(+, θs[a], W), (Np - 1) * log(Z))
                end
                @cthreads parallel for ir = eachindex(A) # non-ideal parallelism
                    i=A[ir]
                    lprob, θp, logcorr = new_p[ir]
                    lπp = logpdf(prior, push_p(prior, θp.x))
                    lπp < 0 && (!isfinite(lπp)) && continue
                    Xp = [cost(push_p(prior, θp.x)) for m = 1:M]
                    Ip = sum(Xp .<= ϵ)
                    Ip == 0 && continue
                    lM = min(lπp - lπs[i] + log(Ip) - log(Ia[i]) + logcorr, 0.0)
                    if lprob < lM
                        θs[i] = θp
                        Xs[i, :] .= Xp
                        Ia[i] = Ip
                        lπs[i] = lπp
                        parallel && (Threads.atomic_add!(accepted, 1); true) || (accepted += 1; true)
                    end
                end
            end
            accepted[] >= mcmc_tol * nparticles && break
        end
        if abs(ϵv - ϵ) < r_epstol * abs(ϵ) ||
           ϵ <= epstol ||
           accepted[] < mcmc_tol * nparticles
           break
        end
    end

    filter = vec((Ws .> 0) .& (sum(Xs .<= ϵ, dims = 2) .> 0))
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles][filter]

    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    W = Particles(Ws[filter])
    (P = P, W = W, ϵ = ϵ)
end

export smc

#=
using KissABC
pp=Factored(Normal(0,5), Normal(0,5))
cc((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2

R=smc(pp,cc,verbose=true,alpha=0.5,nparticles=5000).P
using PyPlot
pygui(true)
sP=Particles(sigmapoints(mean(R),cov(R)))
cc((sP[1],sP[2]))
scatter(R[1].particles,R[2].particles)

scatter(sP[1].particles,sP[2].particles)
hist(R[1].particles,20)

Particles(y)

hist(y,20,weights=Ws)



using Random, KissABC

function costfun((u1, p1); raw=false)
    n=10^6
    A=randexp(n)
    B=rand(n)
    u2 = (1.0 - u1*p1)/(1.0 - p1)
    x = A .* ifelse.(B .< p1, u1, u2)
    sqrt(sum(abs2,[std(x)-2.2, median(x)-0.4]./[2.2,0.4]))
end

@time R=smc(Factored(Uniform(0,1), Uniform(0.5,1)), costfun, nparticles=100, M=1, verbose=true, alpha=0.8,epstol=0.01,parallel=true)

using PyPlot
pygui(true)
scatter(R.P[1].particles,R.P[2].particles)


cov(R.P)

=#
