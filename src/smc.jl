
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

function smc_propose(rng::AbstractRNG, density, particles::AbstractVector, i::Int)
    a = i
    bluegreen = isodd(i)
    while i == a && (isodd(a) == bluegreen)
        a = rand(rng, eachindex(particles))
    end
    Z = sample_g(rng, 2.0)
    W = op(*, op(-, particles[i], particles[a]), Z)
    op(+, particles[a], W), (length(density) - 1) * log(Z)
end

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
    min_nparticles = ceil(
        Int,
        1.5 * (1 + ifelse(parallel, 1, 0)) * length(prior) / (min(alpha, min_r_ess)),
    )
    nparticles >= min_nparticles || error("nparticles must be >= $min_nparticles.")
    θs = [op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
    Xs = parallel ?
        fetch.([
        Threads.@spawn cost(push_p(prior, θs[$i].x)) for i = 1:nparticles, m = 1:M
    ]) :
        [cost(push_p(prior, θs[i].x)) for i = 1:nparticles, m = 1:M]
    lπs = [logpdf(prior, push_p(prior, θs[i].x)) for i = 1:nparticles]
    Ws = [1 / nparticles for i = 1:nparticles]
    ϵ = maximum(Xs)
    Ia = collect(vec(sum(x -> x <= ϵ, Xs, dims = 2)))
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
        for r = 1:retry_N
            new_p = map(1:nparticles) do i
                Ws[i] <= 0 && return (0, 0, 0)
                (log(rand(rng)), smc_propose(rng, prior, θs, i)...)
            end
            @cthreads parallel for i = 1:nparticles # non-ideal parallelism
                Ws[i] == 0 && continue
                lprob, θp, logcorr = new_p[i]
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
                    if parallel
                        Threads.atomic_add!(accepted, 1)
                    else
                        accepted += 1
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

@time R=smc(Factored(Uniform(0,1), Uniform(0.5,1)), costfun, nparticles=100, M=1, verbose=true, alpha=0.9,epstol=0.01,parallel=true)

using PyPlot
pygui(true)
scatter(R.P[1].particles,R.P[2].particles)


cov(R.P)

=#
