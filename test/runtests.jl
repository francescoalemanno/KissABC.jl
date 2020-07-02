using KissABC
using Distributions

using Statistics
using Test
using Random
@testset "AbstractMCMC interface" begin
    pri = Normal(1, 0.2)
    sim(μ) = μ * μ + 1
    cost(x) = abs(sim(x) - 1.5)
    abc = ApproxKernelizedPosterior(pri, cost, 0.001)

    @test sim.(sample(abc, AIS(10),70000))|>mean ≈ 1.5 atol=0.02
end
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
    results_st =
        mcmc(modelabc; nparticles = nparticles, generations = 500, parallel = false)
    results = mcmc(modelabc; nparticles = nparticles, generations = 500, parallel = true)
    bs_median = [median(rand(getindex.(results[1], 1), nparticles)) for i = 1:500]
    bs_median_st = [median(rand(getindex.(results[1], 1), nparticles)) for i = 1:500]
    μ = mean(bs_median)
    μ_st = mean(bs_median_st)
    @test abs(μ - 43.6) < 1
    @test abs(μ_st - 43.6) < 1
    @test abs(μ - μ_st) / hypot(std(bs_median), std(bs_median_st)) < 3
end

@testset "Normal dist -> Dirac Delta inference" begin
    pri = Normal(1, 0.2)
    sim(μ) = μ * μ + 1
    cost(x) = abs(sim(x) - 1.5)
    abc = ApproxKernelizedPosterior(pri, cost, 0.001)
    res = mcmc(abc, nparticles = 100, generations = 100)
    @show sum(res[1] .> 0)
    @test abs(mean(sim.(res[1])) - 1.5) <= 0.005
end

@testset "Normal dist + Uniform Distr -> inference" begin
    pri = Factored(Normal(1, 0.5), DiscreteUniform(1, 10))
    sim((n, du)) = (n * n + du) * (n + randn() * 0.1)
    cost(x) = abs(sim(x) - 5.5)
    model_abc = ApproxPosterior(pri, cost, 0.01)
    @test abs(mean(sim.(mcmc(model_abc, nparticles = 100, generations = 500)[1])) - 5.5) <
          0.2
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
    sim = mcmc(modelabc, nparticles = 50, generations = 1000, parallel = true)
    @test all(
        abs.(
            ((mean(getindex.(sim[1], 1)), mean(getindex.(sim[1], 2))) .- params) ./ params,
        ) .< (0.1, 0.1),
    )
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
    res, _ = mcmc(plan, nparticles = 2000, generations = 10000)
    plan = ApproxKernelizedPosterior(prior, cost, 0.01 / sqrt(2))
    resk, _ = mcmc(plan, nparticles = 2000, generations = 10000)
    testst(alg, r) = begin
        m = mean(abs, st(r) - st_n)
        println(":", alg, ": testing m = ", m)
        m < 0.1
    end
    @test testst("Hard threshold", res)
    @test testst("Kernelized threshold", resk)
end


@testset "Usecase of issue #10" begin
    plan = ApproxPosterior(Normal(0, 1), x -> abs(x - 1.5), 0.01)
    res = mcmc(plan, nparticles = 20, generations = 100)[1]
    @show mean(res), std(res)
    @test abs(mean(res) - 1.5) <= 0.01
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

@show res=mcmc(plan, nparticles=100,generations=125,parallel=true)

using Statistics
function getCI(x::Vector{<:Number})
    quantile(x,[0.25,0.5,0.75])
end
function getCI(x::Vector{<:Tuple})
    [getCI(getindex.(x,i)) for i in 1:length(x[1])]
end

getCI(res[1])
240 generations:
 [0.48958933397111065, 0.4924062224370781, 0.49559446402487584]
 [0.879783065265908, 0.8816472031816496, 0.8835803050367947]
120 generations:
 [0.4893221894893949, 0.49278449533673585, 0.494863093578758]
 [0.8795982153875357, 0.8816146345915951, 0.8829915018185673]
60 generations:
 [0.4887524655164148, 0.49234470862896673, 0.49502567359353133]
 [0.8796953221457162, 0.8814094610516047, 0.8833899764047788]
30 generations:
 [0.48893164848681747, 0.49163740259340305, 0.4944261757524125]
 [0.8795333045815598, 0.8809134893725196, 0.8829819098348083]
early stop:
 [0.49006664933267297, 0.49313860531909304, 0.49497013116625105]
 [0.8804136291097875, 0.8819843728641816, 0.8834306754737902]

quantile.(DiscreteNonParametric(getindex.(res,2),del),[0.25,0.5,0.75])
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
res,_=mcmc(plan,nparticles=10000,generations=500,parallel=true)

prsample=[rand(prior) for i in 1:10000]
μ_pr=getindex.(prsample,1)
σ_pr=getindex.(prsample,2)

μ_p=getindex.(res,1)
σ_p=getindex.(res,2)

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

mcmc(approx_density,nparticles=100,generations=5000,parallel=true)

problem.target
=#
