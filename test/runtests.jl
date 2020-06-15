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

    plan=ABCplan(pri,model,tinydata,dist)
    T=ABC(plan,0.05,nparticles=nparticles)
    P,d,ϵ=T
    @show ϵ,length(P)
    @test abs(mean(getindex.(P,1)) -46)/std(getindex.(P,1))<2
    @show mean(getindex.(P,1))
    res,Δ=ABCDE(plan,0.01,nparticles=5000,generations=100,verbose=false)
    @show mean(getindex.(res,1))
    @test abs(mean(getindex.(res,1)) -46)/std(getindex.(res,1))<2
    res2,Δ=ABCSMCPR(plan,0.05,nparticles=6000,verbose=false)
    @test abs(mean(getindex.(res2,1)) -46)/std(getindex.(res2,1))<2

    @test abs(median(getindex.(res,1)) - 44) <= 1
    @test abs(median(getindex.(res2,1)) - 44) <= 1
    @test abs(median(getindex.(P,1)) - 44) <= 1
end

@testset "Normal dist -> Dirac Delta inference" begin
    pri=Normal(1,0.2)
    sim(μ,params)=μ*μ+1
    dist(x,y)=abs(x-y)
    plan=ABCplan(pri,sim,1.5,dist)
    P,w=ABCSMCPR(plan,0.02,nparticles=2000,verbose=false)
    @test abs((mean(P)-1/sqrt(2))/0.02)<3
    P,w=ABCDE(plan,0.02,nparticles=2000,verbose=false)
    @test abs((mean(P)-1/sqrt(2))/0.02)<3
    P,w=KABCDE(plan,0.02,nparticles=2000,generations=1000,verbose=false)
    @test abs((mean(P.*w)/mean(w)-1/sqrt(2))/(3*0.02))<3
end

@testset "Normal dist -> Normal Dist" begin
    pri=Normal(0,1)
    data=ones(1000)
    sim(μ,other)=randn(length(data)).+μ
    dist(x,y)=abs(mean(x)-mean(y))
    plan=ABCplan(pri,sim,data,dist)
    res,Δ=ABCDE(plan,0.25/sqrt(length(data)),verbose=false)
    @test abs((mean(res)-1)/std(res))<4
end

@testset "Normal dist + Uniform Distr -> inference" begin
    pri=Factored(Normal(1,0.5),DiscreteUniform(1,10))
    sim((n,du),params)=(n*n+du)*(n+randn()*0.1)
    dist(x,y)=abs(x-y)

    plan=ABCplan(pri,sim,5.5,dist)

    P,_ = ABCSMCPR(plan,0.025,verbose=false)
    stat=[sim(P[i],1) for i in eachindex(P)]
    @show mean(stat)
    @test abs((mean(stat)-5.5)/std(stat)) < 1
    P,_ = ABCDE(plan,0.025,generations=1000,verbose=false)
    stat=[sim(P[i],1) for i in eachindex(P)]
    @show mean(stat)
    @test abs((mean(stat)-5.5)/std(stat)) < 1
    P,w,d = KABCDE(plan,0.025,generations=1000,verbose=true)
    stat=[sim(P[i],1) for i in eachindex(P)]
    @show mean(stat)
    @test abs((sum(stat.*w)-5.5)/std(stat)) < 1
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
function brownianrms((μ,σ),N,samples=200)
    trajsq=zeros(2,N)
    for i in 1:samples
        trajsq .+= brownian((μ,σ),N).^2 ./ samples
    end
    sqrt.(sum(trajsq,dims=1)[1,:])
end

@testset "Inference on drifted Wiener Process" begin
    tdata=brownianrms((0.5,2.0),30,10000)
    prior=Factored(Uniform(0,1),Uniform(0,4))
    dist(x,y)=sum(abs,x.-y)/length(x)
    plan=ABCplan(prior,brownianrms,tdata,dist,params=30)
    res,w=ABCSMCPR(plan,0.4,parallel=true,verbose=false)
    @test abs((mean(getindex.(res,2))-2)/std(getindex.(res,2)))<4/sqrt(length(w))
    @test abs((mean(getindex.(res,1))-0.5)/std(getindex.(res,1)))<4/sqrt(length(w))
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    res,w=ABCDE(plan,0.3,generations=100,parallel=true,verbose=false)
    @test abs((mean(getindex.(res,2))-2)/std(getindex.(res,2)))<4/sqrt(length(w))
    @test abs((mean(getindex.(res,1))-0.5)/std(getindex.(res,1)))<4/sqrt(length(w))
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    res,w,d=KABCDE(plan,0.3,generations=100,parallel=true,verbose=false)
    @test abs((sum(getindex.(res,2).*w)-2)/std(getindex.(res,2)))<4/sqrt(length(w))
    @test abs((sum(getindex.(res,1).*w)-0.5)/std(getindex.(res,1)))<4/sqrt(length(w))
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    res,w,ϵ=ABC(plan,0.02,parallel=true)
    @show ϵ
    @show mean(getindex.(res,1)),std(getindex.(res,1))
    @show mean(getindex.(res,2)),std(getindex.(res,2))
    @test abs((mean(getindex.(res,2))-2)/std(getindex.(res,2)))<6/sqrt(length(w))
    @test abs((mean(getindex.(res,1))-0.5)/std(getindex.(res,1)))<6/sqrt(length(w))
end

@testset "Classical Mixture Model 0.1N+N" begin
    st(res)=((quantile(res,0.1:0.1:0.9)-reverse(quantile(res,0.1:0.1:0.9)))/2)[1+(end-1)÷2:end]
    st_n=[0.0, 0.04680825481526908, 0.1057221226763449, 0.2682111969397526, 0.8309228020477986]

    prior=Uniform(-10,10)
    sim(μ,other) = μ+rand((randn()*0.1,randn()))
    dist(x,y)=abs(x-y)
    plan=ABCplan(prior,sim,0.0,dist)

    res2,Δ=ABCSMCPR(plan,0.01,nparticles=300,maxsimpp=Inf,verbose=false,c=0.0001)
    res3,δ=ABCDE(plan,0.01,nparticles=300,generations=2000,verbose=false)
    res6,δ=KABCDE(plan,0.01,nparticles=300,generations=2000,verbose=false)
    res4,δ=ABC(plan,0.001,nparticles=300)
    res5,δ=ABCDE(plan,0.01,nparticles=100,generations=2000,verbose=true,earlystop=true)
    testst(alg,r) = begin
        m = mean(abs,st(r)-st_n)
        println(":",alg,": testing m = ",m)
        m<0.1
    end
    @test testst("ABCSMCPR",res2)
    @test testst("ABCDE",res3)
    @test testst("KABCDE",res6)
    @test !testst("ABCDE ES",res5) #do not remove the not operator
    @test testst("ABC",res4)
end

function ABCplan_onlydistance(prior,costfunction;params=())
    fakedist(x,y)=x+y
    ABCplan(prior,costfunction,0.0,fakedist;params=params)
end
@testset "Usecase of issue #10" begin
    plan=ABCplan_onlydistance(Normal(0,1),(x,other)->abs(x-1.5))
    @test abs(mean(ABCDE(plan,0.01,verbose=false)[1])-1.5)<=0.01
end

#benchmark
#=
function sim((u1, p1), params; n=10^6, raw=false)
 u2 = (1.0 - u1*p1)/(1.0 - p1)
 x = randexp(n) .* ifelse.(rand(n) .< p1, u1, u2)
 raw && return x
 [std(x), median(x)]
end

function dist(s, s0)
 sqrt(sum(((s .- s0)./s).^2))
end
plan=ABCplan(Factored(Uniform(0,1), Uniform(0.5,1)), sim, [2.2, 0.4], dist)

res,del,conv=KABCDE(plan, 0.2, nparticles=100,generations=24,parallel=true)
del
using Statistics
function getCI(x::Vector{<:Number})
    quantile(x,[0.25,0.5,0.75])
end
function getCI(x::Vector{<:Tuple})
    [getCI(getindex.(x,i)) for i in 1:length(x[1])]
end

via getCI(res)
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
plan=ABCplan(prior,sim,tdata,ksdist)
res,_=ABCDE(plan,0.1,nparticles=10000,generations=200,parallel=true)

prsample=[rand(prior) for i in 1:10000]
μ_pr=getindex.(prsample,1)
σ_pr=getindex.(prsample,2)

μ_p=getindex.(res,1)
σ_p=getindex.(res,2)

mean(μ_p),std(μ_p)
mean(σ_p),std(σ_p)

cd(@__DIR__); pwd()
function dilateextrema(X)
    E=extrema(X)
    return (1.02,1.08).*(E.-mean(E)).+mean(E)
end
using PyPlot
pygui(false)
figure(figsize=1.5 .*(7.5,7.5).*(1,(sqrt(5)-1)/2),dpi=200)
subplot(2,2,1)
title("PRIOR")
hist(μ_pr,50,histtype="step",label=L" π(μ)",density=true)
xlim(dilateextrema(μ_pr)...)
legend()
xlabel(L"\mu")
subplot(2,2,2)
title("POSTERIOR")
hist(μ_p,50,histtype="step",label=L" P(μ|{\rm data})",density=true)
xlim(dilateextrema(μ_pr)...)
legend()
xlabel(L"\mu")
subplot(2,2,3)
hist(σ_pr,50,histtype="step",label=L" π(σ)",density=true)
xlim(dilateextrema(σ_pr)...)
xlabel(L"\sigma")
legend()
subplot(2,2,4)
hist(σ_p,50,histtype="step",label=L" P(σ|{\rm data})",density=true)
xlim(dilateextrema(σ_pr)...)
xlabel(L"\sigma")
legend()
tight_layout()
PyPlot.savefig("../images/inf_normaldist.png")

=#
