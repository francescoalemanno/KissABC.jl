using KissABC
using Distributions
using Statistics
using Test
using Random

Random.seed!(1)

@testset "Tiny Data, Approximate Bayesian Computation and the Socks of Karl Broman" begin
    Random.seed!(1)
    function model((n_socks,prop_pairs),consts)
        n_picked=11
        n_pairs=round(Int,prop_pairs*floor(n_socks/2))
        n_odd=n_socks-2*n_pairs
        socks=sort([repeat(1:n_pairs,2);(n_pairs+1):(n_pairs+n_odd)])
        picked_socks=socks[randperm(n_socks)][1:min(n_socks,n_picked)]
        lu=length(unique(picked_socks))
        sample_pairs = min(n_socks,n_picked)-lu
        sample_odds = lu-sample_pairs
        sample_pairs,sample_odds
    end

    prior_mu = 30
    prior_sd = 15
    prior_size = -prior_mu^2 / (prior_mu - prior_sd^2)

    pr_socks=NegativeBinomial(prior_size,prior_size/(prior_mu+prior_size))
    pr_prop=Beta(15,2)

    pri=Factored(pr_socks,pr_prop)

    dist(x,y)=sum(abs,x.-y)
    tinydata=(0,11)
    nparticles=5000
    T=ABC(pri,model,tinydata,dist,0.05,nparticles=nparticles)
    P,d,ϵ=T
    @show ϵ,length(P)
    @test abs(mean(getindex.(P,1)) -46)/std(getindex.(P,1))<5/sqrt(nparticles)
    @show mean(getindex.(P,1))
    res,Δ=ABCDE(pri,model,tinydata,dist,0.01,nparticles=5000)
    @show mean(getindex.(res,1))
    @test abs(mean(getindex.(res,1)) -46)/std(getindex.(res,1))<5/sqrt(nparticles)
    res2,Δ=ABCSMCPR(pri,model,tinydata,dist,0.05,nparticles=6000)
    @test abs(mean(getindex.(res2,1)) -46)/std(getindex.(res2,1))<5/sqrt(nparticles)

    @test abs(median(getindex.(res,1)) - 44) <= 1
    @test abs(median(getindex.(res2,1)) - 44) <= 1
    @test abs(median(getindex.(P,1)) - 44) <= 1
end

@testset "Normal dist -> Dirac Delta inference" begin
    pri=Normal(1,0.2)
    sim(μ,params)=μ*μ+1
    dist(x,y)=abs(x-y)
    P,w=ABCSMCPR(pri,sim,1.5,dist,0.02,nparticles=2000)
    @test abs((mean(P)-1/sqrt(2))/0.02)<3
    P,w=ABCDE(pri,sim,1.5,dist,0.02,nparticles=2000)
    @test abs((mean(P)-1/sqrt(2))/0.02)<3
end

@testset "Normal dist + Uniform Distr -> inference" begin
    pri=Factored(Normal(1,0.5),DiscreteUniform(1,10))
    sim((n,du),params)=(n*n+du)*(n+randn()*0.1)
    dist(x,y)=abs(x-y)
    P,_ = ABCSMCPR(pri,sim,5.5,dist,0.025)
    stat=[sim(P[i],1) for i in eachindex(P)]
    @show mean(stat)
    @test abs((mean(stat)-5.5)/std(stat)) < 1
    P,_ = ABCDE(pri,sim,5.5,dist,0.025)
    stat=[sim(P[i],1) for i in eachindex(P)]
    @show mean(stat)
    @test abs((mean(stat)-5.5)/std(stat)) < 1
end

function brownian((μ,σ),N)
    x=zeros(2)
    μdir=sincos(rand()*2π)
    traj=zeros(2,N)
    for i in 1:N
        traj[:,i].=x
        x.+=μ.*μdir.+randn(2).*σ
    end
    traj.-traj[:,1:1]
end
function brownianrms((μ,σ),N,samples=100)
    trajsq=zeros(2,N)
    for i in 1:samples
        trajsq .+= brownian((μ,σ),N).^2 ./ samples
    end
    sqrt.(sum(trajsq,dims=1)[1,:])
end

@testset "Inference on skewed brownian model" begin
    tdata=brownianrms((0.5,2.0),30,10000)
    prior=Factored(Uniform(0,1),Uniform(0,4))
    dist(x,y)=sum(abs,x.-y)/length(x)
    res,w=ABCSMCPR(prior,brownianrms,tdata,dist,0.5,params=30,parallel=true)
    @test abs((mean(getindex.(res,2))-2)/std(getindex.(res,2)))<4/sqrt(length(w))
    @test abs((mean(getindex.(res,1))-0.5)/std(getindex.(res,1)))<4/sqrt(length(w))
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    res,w=ABCDE(prior,brownianrms,tdata,dist,0.5,params=30,parallel=true)
    @test abs((mean(getindex.(res,2))-2)/std(getindex.(res,2)))<4/sqrt(length(w))
    @test abs((mean(getindex.(res,1))-0.5)/std(getindex.(res,1)))<4/sqrt(length(w))
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    res,w,ϵ=ABC(prior,brownianrms,tdata,dist,0.03,params=30,parallel=true)
    @show ϵ
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    @test abs((mean(getindex.(res,2))-2)/std(getindex.(res,2)))<7/sqrt(length(w))
    @test abs((mean(getindex.(res,1))-0.5)/std(getindex.(res,1)))<7/sqrt(length(w))
end


#plotting stuff
#=
using StatsBase
function ksdist(x,y)
    p1=ecdf(x)
    p2=ecdf(y)
    r=[x;y]
    maximum(abs.(p1.(r)-p2.(r)))
end


tdata=randn(1000).*0.04.+2

sim((μ,σ),param)=randn(100).*σ.+μ

prior=Factored(Uniform(1,3),Truncated(Normal(0,0.1),0,100))

res,_=ABCDE(prior,sim,tdata,ksdist,0.1,nparticles=5000,parallel=true)

prsample=[rand(prior) for i in 1:5000]
μ_pr=getindex.(prsample,1)
σ_pr=getindex.(prsample,2)

μ_p=getindex.(res,1)
σ_p=getindex.(res,2)

mean(μ_p),std(μ_p)
mean(σ_p),std(σ_p)

cd(@__DIR__); pwd()
function dilateextrema(X)
    E=extrema(X)
    return 1.05.*(E.-mean(E)).+mean(E)
end
using PyPlot
pygui(true)
figure(figsize=(10,10).*(1,(sqrt(5)-1)/2),dpi=150)
subplot(2,2,1)
title("PRIOR")
hist(μ_pr,50,histtype="step",label=L"π(μ)")
xlim(dilateextrema(μ_pr)...)
legend()
xlabel(L"\mu")
subplot(2,2,2)
title("POSTERIOR")
hist(μ_p,50,histtype="step",label=L"P(μ|{\rm data})")
xlim(dilateextrema(μ_pr)...)
legend()
xlabel(L"\mu")
subplot(2,2,3)
hist(σ_pr,50,histtype="step",label=L"π(σ)")
xlim(dilateextrema(σ_pr)...)
xlabel(L"\sigma")
legend()
subplot(2,2,4)
hist(σ_p,50,histtype="step",label=L"P(σ|{\rm data})")
xlim(dilateextrema(σ_pr)...)
xlabel(L"\sigma")
legend()
tight_layout()
savefig("../images/inf_normaldist.png")
=#
