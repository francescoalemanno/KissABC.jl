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

"""
    ABCDE(plan, ϵ_target; α=1/3, nparticles = 100, maxsimpp = 200, mcmcsteps=0, parallel = false, verbose = true)

A sequential monte carlo algorithm inspired by differential evolution, work in progress, very efficient (simpler version of B.M.Turner 2012, https://doi.org/10.1016/j.jmp.2012.06.004)

# Arguments:
- `plan`: a plan built using the function ABCplan.
- `ϵ_target`: maximum acceptable distance between simulated datasets and the target dataset
- `α`: the adaptive ϵ at every iteration is chosen as `ϵ → m*(1-α)+M*α` where `m` and `M` are respectively minimum and maximum distance of current population.
- `nparticles`: number of samples from the approximate posterior that will be returned
- `maxsimpp`: average maximum number of simulations per particle
- `mcmcsteps`: option to sample more than `1` population of `nparticles`, the end population will contain `(1 + mcmcsteps) * nparticles` total particles
- `parallel`: when set to `true` multithreaded parallelism is enabled
- `verbose`: when set to `true` verbosity is enabled
"""
ABCDE


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

function ABCDE_innerloop(plan::ABCplan,ϵ,θs,Δs,idx,parallel)
    @extract_params plan prior distance simulation data params
    nθs=copy(θs)
    nΔs=copy(Δs)
    nparticles=length(θs)
    γ = 2.38/sqrt(2*length(prior))
    @cthreads parallel for i in idx
        a=i#rand(1:nparticles)
        b=a
        while b==a
            b=rand(1:nparticles)
        end
        c=a
        while c==a || c==b
            c=rand(1:nparticles)
        end
        θp=deperturb(prior,θs[a],θs[b],θs[c],γ)
        w_prior=pdf(prior,θp)/pdf(prior,θs[i])
        w=min(1,w_prior)
        rand()>w && continue
        xp=simulation(θp,params)
        dp=distance(xp,data)
        if dp<max(ϵ,Δs[i])
            nΔs[i]=dp
            nθs[i]=θp
        end
    end
    nθs,nΔs
end

function ABCDE(plan::ABCplan, ϵ_target;
                  nparticles=100, maxsimpp=200, parallel=false, α=1/3, mcmcsteps=0, verbose=true)
    # simpler version of https://doi.org/10.1016/j.jmp.2012.06.004
    @extract_params plan prior distance simulation data params
    @assert 0<α<1 "α must be strictly between 0 and 1."
    θs,Δs=sample_plan(plan,nparticles,parallel)
    nsim=nparticles
    ϵ_current=max(ϵ_target,mean(extrema(Δs)))+1
    while maximum(Δs)>ϵ_target && nsim < maxsimpp*nparticles
        ϵ_past=ϵ_current
        ϵ_current=max(ϵ_target,sum(extrema(Δs).*(1-α,α)))
        idx=(1:nparticles)#[Δs.>ϵ_current]
        θs,Δs=ABCDE_innerloop(plan, ϵ_current, θs, Δs, idx, parallel)
        nsim+=length(idx)
        if verbose && ϵ_current!=ϵ_past
            @info "Finished run:" completion=1-sum(Δs.>ϵ_target)/nparticles num_simulations=nsim ϵ=ϵ_current
        end
    end
    ϵ_current=maximum(Δs)
    verbose && @info "ABCDE Ended:" completion=1-sum(Δs.>ϵ_target)/nparticles num_simulations=nsim ϵ=ϵ_current
    converged = true
    if ϵ_target < ϵ_current
        verbose && @warn "Failed to reach target ϵ.\n   possible fix: increase maximum number of simulations"
        converged = false
    end

    if mcmcsteps>0 && converged
        verbose && @info "Performing additional MCMC-DE steps at tolerance " ϵ_current
        for i in 1:mcmcsteps
            nθs,nΔs=ABCDE_innerloop(plan, ϵ_current, θs[end-nparticles+1:end], Δs[end-nparticles+1:end], 1:nparticles, parallel)
            append!(θs,nθs)
            append!(Δs,nΔs)
            if verbose
                @info "Finished step:" i remaining_steps=mcmcsteps-i
            end
        end
    end
    θs, Δs, converged
end

function DE(plan::ABCplan, ϵ_target; nparticles=100, generations=100, parallel=false, verbose=true)
    @extract_params plan prior distance simulation data params
    θs,Δs=sample_plan(plan,nparticles,false)
    γ = 2.38/sqrt(2*length(prior))
    iters=0
    complete=1-sum(Δs.>ϵ_target)/nparticles
    while iters<generations
        iters+=1
        nθs=identity.(θs)
        nΔs=identity.(Δs)
        @cthreads parallel for i in 1:nparticles
            a=i
            while a==i
                a=rand(1:nparticles)
            end
            b=a
            while b==a || b==i
                b=rand(1:nparticles)
            end
            θp=deperturb(prior,θs[i],θs[a],θs[b],γ)
            w_prior=pdf(prior,θp)/pdf(prior,θs[i])
            rand() > min(1,w_prior) && continue
            xp=simulation(θp,params)
            dp=distance(xp,data)
            if dp <= max(ϵ_target, Δs[i])
                nΔs[i]=dp
                nθs[i]=θp
            end
        end
        θs=nθs
        Δs=nΔs
        ncomplete=1-sum(Δs.>ϵ_target)/nparticles
        if verbose && (ncomplete!=complete)
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

export DE
