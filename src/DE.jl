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

function deperturb(prior::Factored{N},sample,r1,r2,γ) where N
    coeffs=ntuple(Val(N)) do i
        rand((1.0,0.1))
    end
    λ=0.2*(rand()-0.5) # CU(-0.1,0.1)
    corr_γ=γ*sqrt(N/sum(coeffs))*(1+λ)
    ntuple(Val(N)) do i
        return deperturb(prior.p[i],sample[i],r1[i],r2[i],coeffs[i]*corr_γ)
    end
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
    ABCDE(plan, ϵ_target; nparticles=100, generations=500, α=0, parallel=false, earlystop=false, verbose=true)

A population monte carlo algorithm inspired by differential evolution, very efficient (simpler version of B.M.Turner 2012, https://doi.org/10.1016/j.jmp.2012.06.004)

# Arguments:
- `plan`: a plan built using the function ABCplan.
- `ϵ_target`: maximum acceptable distance between simulated datasets and the target dataset
- `nparticles`: number of samples from the approximate posterior that will be returned
- `generations`: total number of simulations per particle
- `α`: controls the ϵ for each simulation round as ϵ = m+α*(M-m) where m,M = extrema(distances)
- `parallel`: when set to `true` multithreaded parallelism is enabled
- `earlystop`: when set to `true` a particle is no longer updated as soon as it has reached ϵ_target, this provides a huge speedup, but it can lead to erroneous posterior distribution
- `verbose`: when set to `true` verbosity is enabled
"""
function ABCDE(plan::ABCplan, ϵ_target; nparticles=100, generations=500, α=0, parallel=false, earlystop=false, verbose=true)
    @assert 0<=α<1 "α must be in 0 <= α < 1."
    @extract_params plan prior distance simulation data params
    θs,Δs=sample_plan(plan,nparticles,parallel)
    nsims=zeros(Int,nparticles)
    γ = 2.38/sqrt(2*length(prior))
    iters=0
    complete=1-sum(Δs.>ϵ_target)/nparticles
    while iters<generations
        iters+=1
        nθs = identity.(θs)
        nΔs = identity.(Δs)
        ϵ_l, ϵ_h = extrema(Δs)
        if earlystop
            ϵ_h<=ϵ_target && break
        end
        ϵ_pop = max(ϵ_target,ϵ_l + α * (ϵ_h - ϵ_l))
        @cthreads parallel for i in 1:nparticles
            if earlystop
                Δs[i] <= ϵ_target && continue
            end
            s = i
            ϵ = ifelse(Δs[i] <= ϵ_target, ϵ_target, ϵ_pop)
            if Δs[i] > ϵ
                s=rand((1:nparticles)[Δs .<= Δs[i]])
            end
            a = s
            while a == s
                a = rand(1:nparticles)
            end
            b = a
            while b == a || b == s
                b = rand(1:nparticles)
            end
            θp = deperturb(prior,θs[s],θs[a],θs[b],γ)
            w_prior = pdf(prior,θp) / pdf(prior,θs[i])
            rand() > min(1,w_prior) && continue
            xp = simulation(θp, params)
            nsims[i]+=1
            dp = distance(xp,data)
            if dp <= max(ϵ, Δs[i])
                nΔs[i] = dp
                nθs[i] = θp
            end
        end
        θs = nθs
        Δs = nΔs
        ncomplete = 1 - sum(Δs .> ϵ_target) / nparticles
        if verbose && (ncomplete != complete || complete >= (nparticles - 1) / nparticles)
            @info "Finished run:" completion=ncomplete nsim = sum(nsims) range_ϵ = extrema(Δs)
        end
        complete=ncomplete
    end
    conv=maximum(Δs) <= ϵ_target
    if verbose
        @info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs)
    end
    θs, Δs, conv
end

"""
    KABCDE(plan, ϵ; nparticles=100, generations=100, parallel=false, verbose=true)

A sequential monte carlo algorithm inspired by differential evolution, very efficient (simpler version of B.M.Turner 2012, https://doi.org/10.1016/j.jmp.2012.06.004).
This method uses a kernel function to accept or reject samples on the hypothesis of Gaussianly distributed errors.

# Arguments:
- `plan`: a plan built using the function ABCplan.
- `ϵ`: target statistical distance between between simulated datasets and the target dataset
- `nparticles`: number of samples from the approximate posterior that will be returned
- `generations`: total number of simulations per particle
- `parallel`: when set to `true` multithreaded parallelism is enabled
- `verbose`: when set to `true` verbosity is enabled
"""
function KABCDE(plan::ABCplan, ϵ; nparticles=100, generations=100, parallel=false, verbose=true)
    @assert ϵ>0 "ϵ must be greater than zero, since ϵ represents the kernel bandwidth"
    @extract_params plan prior distance simulation data params
    θs,Δs=sample_plan(plan,nparticles,parallel)
    nsims=zeros(Int,nparticles)
    γ = 2.38/sqrt(2*length(prior))
    iters=0
    kernel=Normal(oftype(ϵ,0),ϵ)
    logJ(d) = logpdf(kernel,d)
    while iters<generations
        iters+=1
        nθs = identity.(θs)
        nΔs = identity.(Δs)
        @cthreads parallel for i in 1:nparticles
            a = i
            while a == i
                a = rand(1:nparticles)
            end
            b = a
            while b == a || b == i
                b = rand(1:nparticles)
            end
            θp = deperturb(prior,θs[i],θs[a],θs[b],γ)
            xp = simulation(θp, params)
            nsims[i]+=1
            dp = distance(xp,data)
            log_w=logpdf(prior,θp)-logpdf(prior,θs[i])+logJ(dp)-logJ(Δs[i])
            if log(rand()) <= min(0,log_w)
                nΔs[i] = dp
                nθs[i] = θp
            end
        end
        θs = nθs
        Δs = nΔs
        if verbose
            diagnostic=chisq_diagnostic(prior,Δs,ϵ)
            @info "Finished run:" nsim = sum(nsims) range_ϵ = extrema(Δs) reduced_χ²=diagnostic.red_chisq ESS=diagnostic.ess effective_ϵ=diagnostic.eff_ϵ
        end
    end
    if verbose
        diagnostic=chisq_diagnostic(prior,Δs,ϵ)
        @info "Last run:" nsim = sum(nsims) range_ϵ = extrema(Δs) reduced_χ²=diagnostic.red_chisq ESS=diagnostic.ess effective_ϵ=diagnostic.eff_ϵ
    end
    ws=logJ.(Δs)
    goodsamples=isfinite.(ws) .& (!isnan).(ws)
    θs=θs[goodsamples]
    Δs=Δs[goodsamples]
    ws=ws[goodsamples]
    ws = exp.(ws) ./ sum( exp.(ws) )
    (samples=θs, weights=ws, errors=Δs)
end

export ABCDE, KABCDE
