
function de_propose(
    rng::AbstractRNG,
    density::AbstractDensity,
    particles::AbstractVector,
    i::Int,
)
    γ = 2.38 / sqrt(2 * length(density)) * exp(randn(rng) * 0.1)
    a = b = i
    while a ∈ (i,)
        a = rand(rng, eachindex(particles))
    end
    while b ∈ (a, i)
        b = rand(rng, eachindex(particles))
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
    density::AbstractDensity,
    particles::AbstractVector,
    i::Int,
)
    a = b = c = i
    while a ∈ (i,)
        a = rand(rng, eachindex(particles))
    end
    while b ∈ (a, i)
        b = rand(rng, eachindex(particles))
    end
    while c ∈ (b, a, i)
        c = rand(rng, eachindex(particles))
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
    density::AbstractDensity,
    particles::AbstractVector,
    i::Int,
)
    a = i
    while i == a
        a = rand(rng, eachindex(particles))
    end
    Z = sample_g(rng, 3.0)
    W = op(*, op(-, particles[i], particles[a]), Z)
    op(+, particles[a], W), (length(density) - 1) * log(Z)
end

function propose(
    rng::AbstractRNG,
    density::AbstractDensity,
    particles::AbstractVector,
    i::Int,
)
    p = rand(rng, (1, 1, 1, 1, 2, 2, 3))
    pr = (stretch_propose, de_propose, ais_walk_propose)
    pr[p](rng, density, particles, i)
end

function transition!(
    density::AbstractDensity,
    particles::AbstractVector,
    logdensity::AbstractVector,
    particle_index::Int,
    rng::AbstractRNG,
)
    p, ld_correction = propose(rng, density, particles, particle_index)
    ld = loglike(density, push_p(density, p))
    if accept(density, rng, logdensity[particle_index], ld, ld_correction)
        particles[particle_index] = p
        logdensity[particle_index] = ld
        return true
    end
    return false
end
