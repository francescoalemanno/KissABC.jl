"""
    deperturb(prior::Distribution, sample, r1, r2, γ)

Function for `ABCDE` whose purpose is computing `sample + γ (r1 - r2) + ϵ` (the perturbation function of differential evolution) in a way suited to the prior.

# Arguments:
- `prior`
- `sample`
- `r1`
- `r2`
"""
deperturb

function deperturb(prior::Factored,sample,r1,r2,γ)
    deperturb.(prior.p,sample,r1,r2,γ)
end

function de_ϵ(sample,r1,r2,γ)
    γ*(abs(r1-r2)+abs(sample-r2)+abs(r1-sample))/100
end

function deperturb(prior::ContinuousUnivariateDistribution,sample,r1,r2,γ)
    p = (r2-r1)*γ + randn()*de_ϵ(sample,r1,r2,γ)
    sample + p
end

function deperturb(prior::DiscreteUnivariateDistribution,sample::T,r1,r2,γ) where T
    p = (r2-r1)*γ + randn()*max(de_ϵ(sample,r1,r2,γ),0.5)
    sp=sign(p)
    ap=abs(p)
    intp=floor(ap)
    floatp=ap-intp
    pprob=(intp+ifelse(rand()>floatp,oftype(p,0),oftype(p,1)))*sp
    sample + round(T,pprob)
end

"""
    ABCDE(plan, ϵ_target; nparticles=100, generations=500, parallel=false, verbose=true)

A sequential monte carlo algorithm inspired by differential evolution, very efficient (simpler version of B.M.Turner 2012, https://doi.org/10.1016/j.jmp.2012.06.004)

# Arguments:
- `plan`: a plan built using the function ABCplan.
- `ϵ_target`: maximum acceptable distance between simulated datasets and the target dataset
- `nparticles`: number of samples from the approximate posterior that will be returned
- `generations`: total number of simulations per particle
- `α`: controls the ϵ for each simulation round as ϵ = m+α*(M-m) where m,M = extrema(distances)
- `parallel`: when set to `true` multithreaded parallelism is enabled
- `verbose`: when set to `true` verbosity is enabled
"""
function ABCDE(plan::ABCplan, ϵ_target; nparticles=100, generations=500, α=0, parallel=false, verbose=true)
    @assert 0<=α<1 "α must be in 0 <= α < 1."
    @extract_params plan prior distance simulation data params
    θs,Δs=sample_plan(plan,nparticles,parallel)
    γ = 2.38/sqrt(2*length(prior))
    iters=0
    complete=1-sum(Δs.>ϵ_target)/nparticles
    while iters<generations
        iters+=1
        nθs=identity.(θs)
        nΔs=identity.(Δs)
        ϵ_l,ϵ_h=extrema(Δs)
        ϵ = max(ϵ_target,ϵ_l + α * (ϵ_h - ϵ_l))
        @cthreads parallel for i in 1:nparticles
            s=i
            if Δs[i]>ϵ
                s=rand(1:nparticles)
            end
            a=s
            while a==s
                a=rand(1:nparticles)
            end
            b=a
            while b==a || b==s
                b=rand(1:nparticles)
            end
            θp=deperturb(prior,θs[s],θs[a],θs[b],γ)
            w_prior=pdf(prior,θp)/pdf(prior,θs[i])
            rand() > min(1,w_prior) && continue
            xp=simulation(θp,params)
            dp=distance(xp,data)
            if dp <= max(ϵ, Δs[i])
                nΔs[i]=dp
                nθs[i]=θp
            end
        end
        θs=nθs
        Δs=nΔs
        ncomplete=1-sum(Δs.>ϵ_target)/nparticles
        if verbose && (ncomplete!=complete || complete>=(nparticles-1)/nparticles)
            @info "Finished run:" completion=ncomplete nsim=iters*nparticles range_ϵ=extrema(Δs)
        end
        complete=ncomplete
    end
    conv=maximum(Δs)<=ϵ_target
    if verbose
        @info "End:" completion=complete converged=conv nsim=generations*nparticles range_ϵ=extrema(Δs)
    end
    θs,Δs,conv
end

export ABCDE
