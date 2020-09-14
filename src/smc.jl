macro cthreads(condition::Symbol, loop) #does not work well because of #15276, but seems to work on Julia v0.7
    return esc(quote
        if $condition 
            Threads.@threads $loop
        else
            $loop
        end
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
    r_epstol = (1 - alpha)^1.5 / 50,
    min_r_ess = alpha^2,
    max_stretch = 2.0,
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
- `max_stretch` - the proposal distribution of `smc` is the stretch move of Foreman-Mackey et al. 2013, the larger the parameters the wider becomes the proposal distribution.
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
    alpha = 0.95,
    mcmc_retrys::Int = 0,
    mcmc_tol = 0.015,
    epstol = 0.0,
    r_epstol = (1 - alpha)^1.5 / 50,
    min_r_ess = alpha^2,
    max_stretch = 2.0,
    verbose::Bool = false,
    parallel::Bool = false,
) where {Tprior<:Distribution}
    min_r_ess > 0 || error("min_r_ess must be > 0.")
    mcmc_retrys >= 0 || error("mcmc_retrys must be >= 0.")
    alpha > 0 || error("alpha must be > 0.")
    r_epstol >= 0 || error("r_epstol must be >= 0")
    mcmc_tol >= 0 || error("mcmc_tol must be >= 0")
    max_stretch > 1 || error("max_stretch must be > 1")
    Np=length(prior)
    min_nparticles = ceil(
        Int,
        3 * Np / (min(alpha, min_r_ess)),
    )
    nparticles >= min_nparticles || error("nparticles must be >= $min_nparticles.")
    θs = [op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
    Xs = parallel ?
        fetch.([
        Threads.@spawn cost(push_p(prior, θs[$i].x)) for i = 1:nparticles]) :
        [cost(push_p(prior, θs[i].x)) for i = 1:nparticles]

    lπs = [logpdf(prior, push_p(prior, θs[i].x)) for i = 1:nparticles]
    α = alpha
    ϵ = Inf
    alive = fill(true,nparticles)
    iteration = 0
    # Step 1 - adaptive threshold
    while true
        iteration += 1
        ϵv = ϵ
        ϵ = quantile(Xs[alive],α)
        flag=false
        if ϵ > minimum(Xs[alive])
            alive = Xs .< ϵ
        else
            alive = Xs .<= ϵ
            flag=true
        end
        ESS = sum(alive)
        verbose && @show iteration, ϵ, ESS
        # Step 2 - Resampling
        if α*ESS <= nparticles * min_r_ess
            idxalive = (1:nparticles)[alive]
            idx=repeat(idxalive,ceil(Int,nparticles/length(idxalive)))[1:nparticles]
            θs = θs[idx]
            Xs = Xs[idx]
            lπs = lπs[idx]
            ESS = nparticles
            alive .= true
        end

        # Step 3 - MCMC
        accepted = parallel ? Threads.Atomic{Int}(0) : 0
        retry_N = 1 + mcmc_retrys

        for r = 1:retry_N
                new_p = map(1:nparticles) do i
                    a = b = i
                    alive[i] || return (nothing,nothing,nothing)
                    while a==i; a = rand(rng,1:nparticles); end
                    while b==i || b==a; b = rand(rng,1:nparticles); end
                    W = op(*, op(-, θs[b], θs[a]), max_stretch*randn(rng)/sqrt(Np))
                    (log(rand(rng)), op(+, θs[i], W), 0.0)
                end
                @cthreads parallel for i = 1:nparticles # non-ideal parallelism
                    alive[i] || continue
                    lprob, θp, logcorr = new_p[i]
                    isnothing(lprob) && continue
                    lπp = logpdf(prior, push_p(prior, θp.x))
                    lπp < 0 && (!isfinite(lπp)) && continue
                    lM = min(lπp - lπs[i] + logcorr, 0.0)
                    if lprob < lM 
                        Xp = cost(push_p(prior, θp.x))
                        if flag
                            Xp > ϵ && continue
                        else
                            Xp >= ϵ && continue
                        end
                        θs[i] = θp
                        Xs[i] = Xp
                        lπs[i] = lπp
                        if parallel 
                            Threads.atomic_add!(accepted, 1)
                        else
                            accepted += 1
                        end
                    end
                end
            accepted[] >= mcmc_tol * nparticles && break
        end
        if 2*abs(ϵv - ϵ) < r_epstol * (abs(ϵv)+abs(ϵ)) ||
           ϵ <= epstol ||
           accepted[] < mcmc_tol * nparticles
           break
        end
    end
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles][alive]

    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    length(P)==1 && (P=first(P))
    (P = P, C = Xs, ϵ = ϵ)
end

export smc

#=
using KissABC

prior = Uniform(-10, 10)
sim(μ) = μ + rand((randn() * 0.1, randn()))
cost(x) = abs(sim(x) - 0.0)
R=smc(prior,cost,verbose=true,nparticles=10000,alpha=0.99,mcmc_tol=0.0001).P
hist(R.particles,80,density=true)


pp=Normal(0,5)

cc(x) = 50*(x+randn()*0.01-1)^2

R=smc(pp,cc,verbose=true,alpha=0.95,nparticles=500).P


using KissABC
pp=Factored(Normal(0,5), Normal(0,5))
cc((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2

R=smc(pp,cc,verbose=true,alpha=0.95,nparticles=500).P
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

@time R=smc(Factored(Uniform(0,1), Uniform(0.5,1)), costfun, nparticles=100,alpha=0.75, verbose=true, parallel=true)

plan=ApproxPosterior(Factored(Uniform(0,1), Uniform(0.5,1)), costfun, 0.01)

@time res = sample(plan, AIS(25),MCMCThreads(),25,4,discard_initial=2500)

using PyPlot
pygui(true)
scatter(R.P[1].particles,R.P[2].particles)


Particles(sigmapoints(mean(R.P),cov(R.P)))

=#



function pfilter(prior, cost, N; rng=Random.GLOBAL_RNG, q=0.7, eff_tol = 0.1, epstol=-Inf, max_iters = Inf, proposal_width=0.75, verbose=false, parallel=false)
    lowN=4*length(prior)
    if N*q<=lowN
        N=ceil(Int,(lowN+1)/q)
    end
    sample=[op(float, Particle(rand(rng, prior))) for i = 1:N]
    logπ = [logpdf(prior, push_p(prior,sample[i].x)) for i = 1:N]
    C = fill(cost(sample[1].x),N)
    @cthreads parallel for i = 1:N
        trng=rng
        parallel && (trng=Random.default_rng(Threads.threadid());)
        if isfinite(logπ[i])
            C[i] = cost(sample[i].x)
        end
        while (!isfinite(C[i])) || (!isfinite(logπ[i]))
            sample[i]=op(float, Particle(rand(trng, prior)))
            logπ[i] = logpdf(prior, push_p(prior,sample[i].x))
            C[i] = cost(sample[i].x)
        end
    end

    iters = 0
    while true
        iters += 1  
        ϵ = quantile(C,q)
        filter_bad=C .> ϵ
        idxok=(1:N)[.!filter_bad]
        idxbad=(1:N)[filter_bad]
        nreps= Threads.Atomic{Int}(0)
        @cthreads parallel for i in idxbad
            trng=rng
            parallel && (trng=Random.default_rng(Threads.threadid());)
            localreps=0
            @label resample
            b=c=d=rand(trng,idxok)
            while c==b; c=rand(trng,idxok); end
            while d==b || d==c; d=rand(trng,idxok); end
            p=op(+,sample[b],op(*,op(-,sample[d],sample[c]), randn(trng)*proposal_width))
            localreps += 1

            ll = logpdf(prior, push_p(prior,p.x))
            if log(rand(trng)) > min(0.0,ll-logπ[i])
                @goto resample
            end
            Cp=cost(p.x)
            if Cp > ϵ
                @goto resample
            end
            C[i] = Cp
            sample[i] = p
            logπ[i] = ll
            Threads.atomic_add!(nreps,localreps)
        end
        eff=length(idxbad)/nreps[]
        verbose && @show iters, ϵ, eff
        eff<eff_tol && break
        ϵ<epstol && break
        iters > max_iters && break
    end

    θs = [push_p(prior, sample[i].x) for i = 1:N]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    length(P)==1 && (P=first(P))
    (P=P, C=Particles(C))
end


export pfilter



function ABCDE(prior, cost, ϵ_target; nparticles=50, generations=20, α=0, parallel=false, earlystop=false, verbose=true, rng=Random.GLOBAL_RNG, proposal_width=1.0)
    @assert 0<=α<1 "α must be in 0 <= α < 1."
    θs =[op(float, Particle(rand(rng, prior))) for i = 1:nparticles]

    logπ = [logpdf(prior, push_p(prior,θs[i].x)) for i = 1:nparticles]
    Δs = fill(cost(θs[1].x),nparticles)

    @cthreads parallel for i = 1:nparticles
        trng=rng
        parallel && (trng=Random.default_rng(Threads.threadid());)
        if isfinite(logπ[i])
            Δs[i] = cost(θs[i].x)
        end
        while (!isfinite(Δs[i])) || (!isfinite(logπ[i]))
            θs[i]=op(float, Particle(rand(trng, prior)))
            logπ[i] = logpdf(prior, push_p(prior,θs[i].x))
            Δs[i] = cost(θs[i].x)
        end
    end

    nsims = zeros(Int,nparticles)
    γ = proposal_width*2.38/sqrt(2*length(prior))
    iters=0
    complete=1-sum(Δs.>ϵ_target)/nparticles
    while iters<generations
        iters+=1
        nθs = identity.(θs)
        nΔs = identity.(Δs)
        nlogπ=identity.(logπ)
        ϵ_l, ϵ_h = extrema(Δs)
        if earlystop
            ϵ_h<=ϵ_target && break
        end
        ϵ_pop = max(ϵ_target,ϵ_l + α * (ϵ_h - ϵ_l))
        @cthreads parallel for i in 1:nparticles
            if earlystop
                Δs[i] <= ϵ_target && continue
            end
            trng=rng
            parallel && (trng=Random.default_rng(Threads.threadid());)
            s = i
            ϵ = ifelse(Δs[i] <= ϵ_target, ϵ_target, ϵ_pop)
            if Δs[i] > ϵ
                s=rand(trng,(1:nparticles)[Δs .<= Δs[i]])
            end
            a = s
            while a == s
                a = rand(trng,1:nparticles)
            end
            b = a
            while b == a || b == s
                b = rand(trng,1:nparticles)
            end
            θp = op(+,θs[s],op(*,op(-,θs[a],θs[b]), γ))
            lπ= logpdf(prior, push_p(prior,θp.x))
            w_prior = lπ - logπ[i]
            log(rand(trng)) > min(0,w_prior) && continue
            nsims[i]+=1
            dp = cost(θp.x)
            if dp <= max(ϵ, Δs[i])
                nΔs[i] = dp
                nθs[i] = θp
                nlogπ[i] = lπ
            end
        end
        θs = nθs
        Δs = nΔs
        logπ = nlogπ
        ncomplete = 1 - sum(Δs .> ϵ_target) / nparticles
        if verbose && (ncomplete != complete || complete >= (nparticles - 1) / nparticles)
            @info "Finished run:" completion=ncomplete nsim = sum(nsims) range_ϵ = extrema(Δs)
        end
        complete=ncomplete
    end
    conv=maximum(Δs) <= ϵ_target
    if verbose
        @info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs)
    end
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    length(P)==1 && (P=first(P))
    (P=P, C=Particles(Δs), reached_ϵ=conv)
end


export ABCDE