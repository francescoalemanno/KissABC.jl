using Random, Distributions
using MonteCarloMeasurements

function ess(w)
    sum(w)^2 / sum(abs2, w)
end

function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer) #taken from Turing.jl
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
    Z = sample_g(rng, 3.0)
    W = op(*, op(-, particles[i], particles[a]), Z)
    op(+, particles[a], W), (length(density) - 1) * log(Z)
end

function smc(
    prior,
    cost;
    rng = Random.GLOBAL_RNG,
    nparticles = 1000,
    M = 10,
    retrys = 0,
    alpha = 0.95,
    mcmc_tol = 0.015,
    epstol = 0.0,
    r_epstol = 1e-3,
    verbose = false,
)
    θs = [op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
    Xs = [cost(push_p(prior, θs[i].x)) for i = 1:nparticles, m = 1:M]
    lπs = [logpdf(prior, push_p(prior, θs[i].x)) for i = 1:nparticles]
    Ws = [1 / nparticles for i = 1:nparticles]
    ϵ = maximum(Xs)
    Ia = collect(vec(sum(Xs .<= ϵ, dims = 2)))
    ESS = ess(Ws)
    α = alpha
    iteration = 0

    # Step 1 - adaptive threshold
    @label step1
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
            Ian = vec(sum(Xs .<= ϵn, dims = 2))
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
    if abs(ϵv - ϵ) < r_epstol * abs(ϵ)
        @goto results
    end

    # Step 2 - Resampling
    if ESS <= nparticles * max(0.75, α^2.5)
        idx = resample_residual(Ws, nparticles)
        θs = θs[idx]
        Xs = Xs[idx, :]
        lπs = lπs[idx]
        Ia = Ia[idx]
        Ws .= 1 / nparticles
        ESS = nparticles
    end

    # Step 3 - MCMC
    accepted = 0
    retry_N = 1 + retrys
    for r = 1:retry_N
        Threads.@threads for i = 1:nparticles #non-ideal parallelism
            Ws[i] == 0 && continue
            θp, logcorr = smc_propose(rng, prior, θs, i)
            lπp = logpdf(prior, push_p(prior, θp.x))
            lπp < 0 && (!isfinite(lπp)) && continue
            Xp = [cost(push_p(prior, θp.x)) for m = 1:M]
            Ip = sum(Xp .<= ϵ)
            Ip == 0 && continue
            lM = min(lπp - lπs[i] + log(Ip) - log(Ia[i]) + logcorr, 0.0)
            if log(rand(rng)) < lM
                θs[i] = θp
                Xs[i, :] .= Xp
                Ia[i] = Ip
                lπs[i] = lπp
                accepted += 1
            end
        end

        if accepted > mcmc_tol * nparticles
            if ϵ > epstol
                @goto step1
            else
                @goto results
            end
        end
    end

    @label results
    filter = vec((Ws .> 0) .& (sum(Xs .<= ϵ, dims = 2) .> 0))
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles][filter]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    W = Particles(Ws[filter])
    (P = P, W = W, ϵ = ϵ)
end

export smc

#=
pp=Factored(Normal(0,5), Normal(0,5))
cc((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2

R=smc(pp,cc,verbose=true,alpha=0.99,nparticles=5000,retrys=0).P
using PyPlot
pygui(true)
sP=Particles(sigmapoints(mean(R),cov(R)))
cc((sP[1],sP[2]))
scatter(R[1].particles,R[2].particles)

scatter(sP[1].particles,sP[2].particles)
hist(R[1].particles,20)

Particles(y)

hist(y,20,weights=Ws)





function makecost(n)
    A=randexp(n)
    B=rand(n)
    function COSTF((u1, p1); raw=false)
        u2 = (1.0 - u1*p1)/(1.0 - p1)
        x = A .* ifelse.(B .< p1, u1, u2)
        sqrt(sum(abs2,[std(x)-2.2, median(x)-0.4]./[2.2,0.4]))
    end
end
costf=makecost(10^6)
@time R=smc(Factored(Uniform(0,1), Uniform(0.5,1)), costf, nparticles=100, M=1, verbose=true, alpha=0.5,epstol=0.001)

scatter(R.P[1].particles,R.P[2].particles)


cov(R.P)

=#
