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
    perturbator::AbstractPerturbator,
)
    γ = 2.38 / sqrt(2 * length(density)) * exp(randn(rng) * 0.1)
    b = a = rand(rng, inactive_particles)
    while b == a
        b = rand(rng, inactive_particles)
    end
    W =
        ((particles[a] -′ particles[b]) *′ γ) +′ (
            perturbator *′
            triangle_abs(particles[a], particles[b], particles[i]) *′
            (0.001 * γ)
        )
    particles[i] +′ W, 0.0
end

function ais_walk_propose(
    rng::AbstractRNG,
    density::AbstractApproxDensity,
    particles::AbstractVector,
    i::Int,
    inactive_particles::AbstractVector,
    perturbator::AbstractPerturbator,
)
    c = b = a = rand(rng, inactive_particles)
    while b == a
        b = rand(rng, inactive_particles)
    end
    while c == a || c == b
        c = rand(rng, inactive_particles)
    end
    Xs = (particles[a] +′ particles[b] +′ particles[c]) /′ 3
    W =
        (randn(rng) *′ (particles[a] -′ Xs)) +′
        (randn(rng) *′ (particles[b] -′ Xs)) +′
        (randn(rng) *′ (particles[c] -′ Xs))
    particles[i] +′ W, 0.0
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
    perturbator::AbstractPerturbator,
)
    a = rand(rng, inactive_particles)
    Z = sample_g(rng, 3.0)
    W = (particles[i] -′ particles[a]) *′ Z
    particles[a] +′ W, (length(density) - 1) * log(Z)
end

function propose(
    rng::AbstractRNG,
    density::AbstractApproxDensity,
    particles::AbstractVector,
    i::Int,
    inactive_particles::AbstractVector,
    perturbator::AbstractPerturbator,
)
    rand(rng) < 2 / 3 &&
        return stretch_propose(rng, density, particles, i, inactive_particles, perturbator)
    rand(rng) < 2 / 3 &&
        return de_propose(rng, density, particles, i, inactive_particles, perturbator)
    return ais_walk_propose(rng, density, particles, i, inactive_particles, perturbator)
end

function kernel_mcmc!(
    density::AbstractApproxDensity,
    particles::AbstractVector,
    logdensity::AbstractVector,
    perturbator::AbstractPerturbator,
    inactive_particles::AbstractVector,
    particle_index::Int,
    rng::AbstractRNG,
)
    p, ld_correction =
        propose(rng, density, particles, particle_index, inactive_particles, perturbator)
    ld = loglike(density, tostartingsupport(densitytypes(density), p))
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
    pert = Perturbator(rng)
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
                kernel_mcmc!(density, particles, logdensity, pert, inactive, i, rng) &&
                    (accepted[ai] += 1)
            end
        end
        if verbose >= 1
            stats = [(:generation, reps), (:acceptance_rate, sum(accepted) / nparticles)]
            if verbose >= 2
                μp = foldl(
                    (x, y) -> x +′ (y /′ nparticles),
                    view(particles, 2:nparticles),
                    init = particles[1] /′ nparticles,
                )
                σp =
                    sqrt′(foldl(
                        (x, y) -> x +′ (((y -′ μp) *′ (y -′ μp)) /′ nparticles),
                        view(particles, 2:nparticles),
                        init = ((particles[1] -′ μp) *′ (particles[1] -′ μp)) /′ nparticles,
                    )) *′ sqrt(nparticles / (nparticles - 1))
                stats = [stats..., (:avg_particle, μp), (:std_particle, σp)]
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
    logdensity = [
        loglike(density, tostartingsupport(densitytypes(density), particles[i]))
        for i in eachindex(particles)
    ]
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
        [floatize(unconditional_sample(rng, density)) for i = 1:nparticles],
        generations = generations,
        rng = rng,
        parallel = parallel,
        verbose = verbose,
    )
    pushed_particles = [
        tostartingsupport(densitytypes(density), particles[i])
        for i in eachindex(particles)
    ]

    pushed_particles, loglikes
end

export mcmc
end
