
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
        x -> γ * x / 300 * randn(rng),
        op(
            +,
            op(abs, op(-, particles[a], particles[b])),
            op(abs, op(-, particles[i], particles[b])),
            op(abs, op(-, particles[a], particles[i])),
        ),
    )
    op(+, particles[i], W, T), 0.0
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
        op(*, randn(rng), op(-, particles[a], Xs)),
        op(*, randn(rng), op(-, particles[b], Xs)),
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
    p = rand(rng, (1, 1, 1, 1, 2, 2, 3))
    p == 1 && return stretch_propose(rng, density, particles, i, inactive_particles)
    p == 2 && return de_propose(rng, density, particles, i, inactive_particles)
    return ais_walk_propose(rng, density, particles, i, inactive_particles)
end

function transition!(
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
