using KissABC
using AbstractMCMC
using Statistics
using Test
using Random
Random.seed!(1)

@testset "Factored" begin
    d = Factored(Uniform(0, 1), Uniform(100, 101))
    @test all((0, 100) .<= rand(d) .<= (1, 101))
    @test pdf(d, (0.0, 0.0)) == 0.0
    @test pdf(d, (0.5, 100.5)) == 1.0
    @test logpdf(d, (0.5, 100.5)) == 0.0
    @test logpdf(d, (0.0, 0.0)) == -Inf
    @test length(d) == 2
    m = Factored(Uniform(0.00, 1.0), DiscreteUniform(1, 2))
    sample = rand(m)
    @test 0 < sample[1] < 1
    @test sample[2] == 1 || sample[2] == 2
    @test pdf(m, sample) == 0.5
    @test logpdf(m, sample) ≈ log(0.5)
end

@testset "Push" begin
    push_p = KissABC.push_p
    a /′ b = (typeof(a) == typeof(b)) && all(a .== b)
    @test push_p(Normal(), 1) /′ 1.0
    @test push_p(DiscreteUniform(), 1.0) /′ 1
    @test push_p(Factored(Normal(), DiscreteUniform()), (2, 1.0)) /′ (2.0, 1)
    @test push_p(Product([Normal(), Normal()]), [2, 1]) /′ [2.0, 1.0]
end

@testset "Tiny Data, Approximate Bayesian Computation and the Socks of Karl Broman" begin
    function model((n_socks, prop_pairs), consts)
        n_picked = 11
        n_pairs = round(Int, prop_pairs * floor(n_socks / 2))
        n_odd = n_socks - 2 * n_pairs
        socks = sort([repeat(1:n_pairs, 2); (n_pairs+1):(n_pairs+n_odd)])
        picked_socks = socks[randperm(n_socks)][1:min(n_socks, n_picked)]
        lu = length(unique(picked_socks))
        sample_pairs = min(n_socks, n_picked) - lu
        sample_odds = lu - sample_pairs
        sample_pairs, sample_odds
    end

    prior_mu = 30
    prior_sd = 15
    prior_size = -prior_mu^2 / (prior_mu - prior_sd^2)

    pr_socks = NegativeBinomial(prior_size, prior_size / (prior_mu + prior_size))
    pr_prop = Beta(15, 2)

    pri = Factored(pr_socks, pr_prop)

    tinydata = (0, 11)
    nparticles = 5000
    modelabc = ApproxPosterior(pri, x -> sum(abs, model(x, 0) .- tinydata), 0.1)
    results_st = sample(modelabc, AIS(500), 5000, ntransitions = 100, progress = false)
    @test results_st[1] ≈ 46.2
    @test results_st[2] ≈ 0.866

    P =
        smc(
            pri,
            x -> sum(abs, model(x, 0) .- tinydata),
            nparticles = 5000,
            verbose = false,
            alpha = 0.99,
            r_epstol = 0,
            epstol = 0.01,
        ).P

    @test P[1] ≈ 46.2
    @test P[2] ≈ 0.866
end

@testset "Normal dist -> Dirac Delta inference" begin
    pri = Normal(1, 0.2)
    sim(μ) = μ * μ + 1
    cost(x) = abs(sim(x) - 1.5)
    abc = ApproxKernelizedPosterior(pri, cost, 0.001)
    res = sample(abc, AIS(12), 500, discard_initial = 1000, progress = false)

    @test sim(res) ≈ 1.5
    @test smc(pri, cost, epstol = 0.1).P ≈ 0.707
end

@testset "Normal dist -> Dirac Delta inference, MCMCThreads" begin
    pri = Normal(1, 0.2)
    sim(μ) = μ * μ + 1
    cost(x) = abs(sim(x) - 1.5)
    abc = ApproxKernelizedPosterior(pri, cost, 0.001)
    res = sample(
        abc,
        AIS(12),
        MCMCThreads(),
        100,
        50,
        discard_initial = 50 * 12,
        progress = false,
    )
    @show sim(res)
    @test sim(res) ≈ 1.5
end

@testset "Normal dist + Uniform Distr inference" begin
    pri = Factored(Normal(1, 0.5), DiscreteUniform(1, 10))
    sim((n, du)) = (n * n + du) * (n + randn() * 0.01)
    cost(x) = abs(sim(x) - 5.5)
    model_abc = ApproxPosterior(pri, cost, 0.01)
    res = sample(model_abc, AIS(100), 1000, discard_initial = 10000, progress = false)
    @test sim(Tuple(res)) ≈ 5.5
    @test smc(pri, cost).P[2] ≈ 5
end

function brownianrms((μ, σ), N, samples = 200)
    t = 0:N
    #    rand()<1/20 && sleep(0.001)
    @.(sqrt(μ * μ * t * t + σ * σ * t)) .* (0.95 + 0.1 * rand())
end

@testset "Inference on drifted Wiener Process" begin
    params = (0.5, 2.0)
    tdata = brownianrms(params, 30, 10000)
    prior = Factored(Uniform(0, 1), Uniform(0, 4))
    cost(x) = sum(abs, brownianrms(x, 30) .- tdata) / length(tdata)
    modelabc = ApproxPosterior(prior, cost, 0.1)
    sim = sample(modelabc, AIS(50), 100, discard_initial = 50000, progress = false)
    @test all(sim .≈ params)
    @test all(smc(prior, cost).P .≈ params)
end

@testset "Classical Mixture Model 0.1N+N" begin
    st(res) =
        ((quantile(res, 0.1:0.1:0.9)-reverse(quantile(res, 0.1:0.1:0.9)))/2)[1+(end-1)÷2:end]
    st_n = [
        0.0,
        0.04680825481526908,
        0.1057221226763449,
        0.2682111969397526,
        0.8309228020477986,
    ]

    prior = Uniform(-10, 10)
    sim(μ) = μ + rand((randn() * 0.1, randn()))
    cost(x) = abs(sim(x) - 0.0)
    plan = ApproxPosterior(prior, cost, 0.01)
    res = sample(
        plan,
        AIS(50),
        2000,
        ntransitions = 100,
        discard_initial = 5000,
        progress = false,
    )
    plan = ApproxKernelizedPosterior(prior, cost, 0.01 / sqrt(2))
    resk = sample(
        plan,
        AIS(50),
        2000,
        ntransitions = 100,
        discard_initial = 5000,
        progress = false,
    )
    ressmc = smc(prior, cost, nparticles = 2000, alpha = 0.99, epstol = 0.01).P
    testst(alg, r) = begin
        m = mean(abs, st(r) - st_n)
        println(":", alg, ": testing m = ", m)
        @show r
        m < 0.1
    end
    @test testst("Hard threshold", res)
    @test testst("Kernelized threshold", resk)
    @test testst("SMC", ressmc)
end

@testset "Usecase of issue #10" begin
    plan = ApproxPosterior(Normal(0, 1), x -> abs(x - 1.5), 0.01)
    res = sample(plan, AIS(20), 100, discard_initial = 2000, progress = false)
    @show res
    @test res ≈ 1.5
end

@testset "MVNormal vector test + 4 MCMCThreads" begin
    plan =
        ApproxPosterior(MultivariateNormal(4, 1.0), x -> abs(sum(abs2, x)^0.5 - 1.5), 0.01)
    res = sample(
        plan,
        AIS(20),
        MCMCThreads(),
        100,
        4,
        discard_initial = 10000,
        ntransitions = 40,
        progress = false,
    )
    @test mean(plan.cost(res)) < 0.01
end

@testset "CommonLogDensity: rosenbrock banana density" begin
    D = CommonLogDensity(
        2,
        rng -> randn(rng, 2),
        x -> -100 * (x[1] - x[2]^2)^2 - (x[2] - 1)^2,
    )
    @test length(D) == 2
    @test typeof(KissABC.unconditional_sample(Random.GLOBAL_RNG, D)) <: KissABC.Particle
    res = sample(
        D,
        AIS(50),
        1000,
        ntransitions = 100,
        discard_initial = 2000,
        progress = false,
    )
    @show res
    @test quantile(D.lπ(res), 0.97) > -0.69
end


@testset "Handling of ∞ costs" begin
    D = CommonLogDensity(
        2,
        rng -> rand(2) .* (2, 1) .- (1, 0),
        x -> ifelse(sum(abs2, x) <= 1, 0.0, -Inf),
    )
    D2 = CommonLogDensity(2, rng -> rand(2) .* (2, 1) .- (1, 0), x -> -Inf)
    res = sample(
        D,
        AIS(50),
        1000,
        ntransitions = 100,
        discard_initial = 5000,
        progress = false,
    )
    @test D.lπ(res) == 0
    @test_throws ErrorException sample(D2, AIS(50), 10, progress = false)
end

@testset "SMC" begin
    pp = Factored(Normal(0, 5), Normal(0, 5))
    cc((x, y)) = 50 * (x + randn() * 0.01 - y^2)^2 + (y - 1 + randn() * 0.01)^2

    R = smc(pp, cc, verbose = false, alpha = 0.9, nparticles = 500, epstol = 0.01, parallel=true).P
    @test R[1] ≈ 1
    @test R[2] ≈ 1

    cc2((x, y)) = rand((50 * (x + randn() * 0.01 - y^2)^2 + (y - 1 + randn() * 0.01)^2,Inf))

    R = smc(pp, cc2, verbose = false, alpha = 0.9, nparticles = 1000, epstol = 0.01, parallel=true).P

    @test R[1] ≈ 1
    @test R[2] ≈ 1
end

#benchmark
#=
using KissABC, Distributions, Random
function cost((u1, p1); n=10^6, raw=false)
    u2 = (1.0 - u1*p1)/(1.0 - p1)
    x = randexp(n) .* ifelse.(rand(n) .< p1, u1, u2)
    raw && return x
    sqrt(sum(abs2,[std(x)-2.2, median(x)-0.4]./[2.2,0.4]))
end

plan=ApproxPosterior(Factored(Uniform(0,1), Uniform(0.5,1)), cost, 0.01)

@show res=sample(plan, AIS(100),100,discard_initial=10000)

early stop:
 [0.49006664933267297, 0.49313860531909304, 0.49497013116625105]
 [0.8804136291097875, 0.8819843728641816, 0.8834306754737902]

=#
#plotting stuff

#=
using KissABC
using Distributions

function cost((μ,σ))
    x=randn(1000) .* σ .+ μ
    d1=mean(x)-2.0
    d2=std(x)-0.04
    hypot(d1,d2*50)
end

prior=Factored(Uniform(1,3),Truncated(Normal(0,0.05),0,100))
plan=ApproxKernelizedPosterior(prior,cost,0.005)
res=sample(plan,AIS(10),10000,ntransitions=50)

prsample=[rand(prior) for i in 1:10000]
μ_pr=getindex.(prsample,1)
σ_pr=getindex.(prsample,2)

μ_p=vec(res[:,1,:])
σ_p=vec(res[:,2,:])

mean(μ_p),std(μ_p)
mean(σ_p),std(σ_p)

cd(@__DIR__); pwd()

using PyPlot
pygui(true)
figure(figsize=1.5 .*(7.5,7.5).*(1,(sqrt(5)-1)/2),dpi=200)
subplot(2,2,1)
title("PRIOR")
hist(μ_pr,150,histtype="step",label=L" π(μ)",density=true,range=(0.95,3.05))

legend()
xlabel(L"\mu")
subplot(2,2,2)
title("POSTERIOR")
hist(μ_p,150,histtype="step",label=L" P(μ|{\rm data})",density=true,range=(1.9,2.1))

legend()
xlabel(L"\mu")
subplot(2,2,3)
hist(σ_pr,150,histtype="step",label=L" π(σ)",density=true,range=(-0.005,0.3))

xlabel(L"\sigma")
legend()
subplot(2,2,4)
hist(σ_p,150,histtype="step",label=L" P(σ|{\rm data})",density=true,range=(0.02,0.06))

xlabel(L"\sigma")
legend()
tight_layout()
PyPlot.savefig("../images/inf_normaldist.png")

=#
#=

using ApproxInferenceProblems,Distributions,KissABC
normalizep(X)=X./sum(X)
problem = ApproxInferenceProblem(BlowFly,T=1000,statistics = y -> normalizep(diff([count(x->x<=α,y) for α in 14:16:16014])) )
problem.model(rand(problem.prior))
function cost(X)
    s=problem.model(X)
    costs=0.0
    N=0.0
    for i in eachindex(s)
        if s[i]>0 && problem.data[i]>0
            costs+=abs(s[i]-problem.data[i])/(problem.data[i]+s[i])
            N+=1
        end
    end
    costs/N
end
cost(rand(problem.prior))
approx_density = ApproxPosterior(problem.prior,cost,0.01)

mcmc(approx_density, nparticles=100, generations=5000, parallel=true)

problem.target
=#
