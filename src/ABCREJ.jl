

"""
    ABC(plan, α_target; nparticles = 100, parallel = false)

Classical ABC rejection algorithm.

# Arguments:
- `plan`: a plan built using the function ABCplan.
- `α_target`: target acceptance rate for ABC rejection algorithm, `nparticles/α` will be sampled and only the best `nparticles` will be retained.
- `nparticles`:  number of samples from the approximate posterior that will be returned
- `parallel`: when set to `true` multithreaded parallelism is enabled
"""
function ABC(plan::ABCplan, α_target;
             nparticles=100, parallel=false)
    @extract_params plan prior distance simulation data params
    @assert 0<α_target<=1 "α_target is the acceptance rate, and must be properly set between 0 - 1."
    simparticles=ceil(Int,nparticles/α_target)
    particles,distances=sample_plan(plan,simparticles,parallel)
    idx=sortperm(distances)[1:nparticles]
    (particles=particles[idx],
     distances=distances[idx],
     ϵ=distances[idx[end]])
end
