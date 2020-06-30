module KissABC
using Base.Threads
using Distributions
using Random
using ProgressMeter
import Base.length

macro cthreads(condition::Symbol, loop)
    return esc(quote
        if $condition
            Threads.@threads $loop
        else
            $loop
        end
    end)
end

include("types.jl")
include("priors.jl")

function de_propose(
    rng::AbstractRNG,
    density::AbstractApproxDensity,
    particles::AbstractVector,
    i::Int,
    inactive_particles::AbstractVector,
)
    γ = 2.38 / sqrt(2 * length(density)) * exp(randn(rng) * 0.1)
    b = a = rand(rng, inactive_particles)
    while b == a
        b = rand(rng, inactive_particles)
    end
    W = op(*, op(-, particles[a], particles[b]), γ)
    T = op(
        x -> x / 300 * randn(rng),
        op(
            +,
            op(
                +,
                op(abs, op(-, particles[a], particles[b])),
                op(abs, op(-, particles[i], particles[b])),
            ),
            op(abs, op(-, particles[a], particles[i])),
        ),
    )
    op(+, particles[i], op(+, W, T)), 0.0
end

function ais_walk_propose(
    rng::AbstractRNG,
    density::AbstractApproxDensity,
    particles::AbstractVector,
    i::Int,
    inactive_particles::AbstractVector,
)
    c = b = a = rand(rng, inactive_particles)
    while b == a
        b = rand(rng, inactive_particles)
    end
    while c == a || c == b
        c = rand(rng, inactive_particles)
    end
    Xs = op(/, op(+, particles[a], op(+, particles[b], particles[c])), 3)
    W = op(
        +,
        op(
            +,
            op(*, randn(rng), op(-, particles[a], Xs)),
            op(*, randn(rng), op(-, particles[b], Xs)),
        ),
        op(*, randn(rng), op(-, particles[c], Xs)),
    )
    op(+, particles[i], W), 0.0
end

"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
cdf_g_inv(u, a) = (u * (sqrt(a) - sqrt(1 / a)) + sqrt(1 / a))^2

"Sample from g using inverse transform sampling.  a=2.0 is recommended."
sample_g(rng::AbstractRNG, a) = cdf_g_inv(rand(rng), a)

function stretch_propose(
    rng::AbstractRNG,
    density::AbstractApproxDensity,
    particles::AbstractVector,
    i::Int,
    inactive_particles::AbstractVector,
)
    a = rand(rng, inactive_particles)
    Z = sample_g(rng, 3.0)
    W = op(*, op(-, particles[i], particles[a]), Z)
    op(+, particles[a], W), (length(density) - 1) * log(Z)
end

function propose(
    rng::AbstractRNG,
    density::AbstractApproxDensity,
    particles::AbstractVector,
    i::Int,
    inactive_particles::AbstractVector,
)
    rand(rng) < 2 / 3 &&
        return stretch_propose(rng, density, particles, i, inactive_particles)
    rand(rng) < 2 / 3 && return de_propose(rng, density, particles, i, inactive_particles)
    return ais_walk_propose(rng, density, particles, i, inactive_particles)
end
push_p(density::AbstractApproxPosterior, p::Particle) = Particle(push_p(density.prior, p.x))
push_p(density::Factored, p) = push_p.(density.p, p)
push_p(density::Distribution, p) = push_p.(density, p)
push_p(density::ContinuousDistribution, p::Number) = float(p)
push_p(density::DiscreteDistribution, p::Number) = round(Int, p)
function kernel_mcmc!(
    density::AbstractApproxDensity,
    particles::AbstractVector,
    logdensity::AbstractVector,
    inactive_particles::AbstractVector,
    particle_index::Int,
    rng::AbstractRNG,
)
    p, ld_correction = propose(rng, density, particles, particle_index, inactive_particles)
    ld = loglike(density, push_p(density, p))
    if accept(density, rng, logdensity[particle_index], ld, ld_correction)
        particles[particle_index] = p
        logdensity[particle_index] = ld
        return true
    end
    return false
end

function _mcmc!(
    density::AbstractApproxDensity,
    particles::AbstractVector,
    logdensity::AbstractVector;
    generations,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    parallel = false,
    verbose = 0,
)
    nparticles = length(particles)
    sep = nparticles ÷ 2
    ensembles = ((1:sep, (sep+1):nparticles), ((sep+1):nparticles, 1:sep))
    p = (verbose >= 1) ? Progress(generations, ifelse(verbose > 2, -Inf, 0.25)) : nothing
    accepted = zeros(Int, ifelse(parallel, Threads.nthreads(), 1))
    for reps = 1:generations
        accepted .= 0
        for (active, inactive) in ensembles
            @cthreads parallel for i in active
                ai = ifelse(parallel, Threads.threadid(), 1)
                kernel_mcmc!(density, particles, logdensity, inactive, i, rng) &&
                    (accepted[ai] += 1)
            end
        end
        if verbose >= 1
            stats = [(:generation, reps), (:acceptance_rate, sum(accepted) / nparticles)]
            if verbose >= 2
                μp = foldl(
                    (x, y) -> op(+, x, op(/, y, nparticles)),
                    view(particles, 2:nparticles),
                    init = op(/, particles[1], nparticles),
                )
                σp = op(
                    x -> sqrt(x) * sqrt(nparticles / (nparticles - 1)),
                    foldl(
                        (x, y) -> op(+, x, op(/, op(abs2, op(-, y, μp)), nparticles)),
                        view(particles, 2:nparticles),
                        init = op(/, op(abs2, op(-, particles[1], μp)), nparticles),
                    ),
                )
                stats = [stats..., (:avg_particle, μp.x), (:std_particle, σp.x)]
            end
            if reps == generations
                finish!(p; showvalues = stats)
            else
                next!(p; showvalues = stats)
            end
        end
    end
    nothing
end

function _mcmc(
    density::AbstractApproxDensity,
    particles::AbstractVector;
    generations,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    parallel = false,
    verbose = 0,
)
    logdensity =
        [loglike(density, push_p(density, particles[i])) for i in eachindex(particles)]
    _mcmc!(
        density,
        particles,
        logdensity;
        generations = generations,
        rng = rng,
        parallel = parallel,
        verbose = verbose,
    )
    particles, logdensity
end

"""
    function mcmc(
        density::AbstractApproxDensity;
        nparticles::Int,
        generations,
        rng::AbstractRNG = Random.GLOBAL_RNG,
        parallel = false,
        verbose = 2,
    )
This function will run an Affine Invariant MC sampler on the ABC density defined in `density`,
the ensemble will contain `nparticles` particles, and each particle will evolve for a total number of steps equal to `generations`.
"""
function mcmc(
    density::AbstractApproxDensity;
    nparticles::Int,
    generations,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    parallel = false,
    verbose = 2,
)
    particles, loglikes = _mcmc(
        density,
        [op(float, unconditional_sample(rng, density)) for i = 1:nparticles],
        generations = generations,
        rng = rng,
        parallel = parallel,
        verbose = verbose,
    )
    pushed_particles = [push_p(density, particles[i]).x for i in eachindex(particles)]

    pushed_particles, loglikes
end

export mcmc
end
