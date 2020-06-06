# # A gaussian mixture model
# First of all we define our model,
using KissABC
using Distributions

function model(P,N)
    μ_1, μ_2, σ_1, σ_2, prob=P
    d1=randn(N).*σ_1 .+ μ_1
    d2=randn(N).*σ_2 .+ μ_2
    ps=rand(N).<prob
    R=zeros(N)
    R[ps].=d1[ps]
    R[.!ps].=d2[.!ps]
    R
end

# Let's use the model to generate some data, this data will constitute our dataset
parameters = (1.0, 0.0, 0.2, 2.0, 0.4)
data=model(parameters,5000)

# let's look at the data

using Plots
histogram(data)
savefig("ex1_hist1.svg"); nothing # hide

# ![ex1_hist1](ex1_hist1.svg)

# we can now try to infer all parameters using `KissABC`, first of all we need to define a reasonable prior for our model

prior=Factored(
            Uniform(0,3), # there is surely a peak between 0 and 3
            Uniform(-2,2), #there is a smeared distribution centered around 0
            Uniform(0,1), # the peak has surely a width below 1
            Uniform(0,4), # the smeared distribution surely has a width less than 4
            Beta(4,4) # the number of total events from both distributions look about the same, so we will favor 0.5 just a bit
        );

# let's look at a sample from the prior, to see that it works
rand(prior)
# now we need a distance function to compare datasets, this is possibly the worst distance we could use, but it will work out anyway
function D(x,y)
    r=0:0.01:1
    sum(abs2,quantile.(Ref(x),r).-quantile.(Ref(y),r))/length(r)
end

# we can now run ABCDE to get the posterior distribution of our parameters given the dateset `data`
res,Δ=ABCDE(prior,model,data,D,0.02,params=5000,parallel=true,verbose=false);

# let's see the median and 95% confidence interval for the inferred parameters and let's compare them with the true values
function getstats(P,V)
    (
        param=P,
        median=median(V),
        lowerbound=quantile(V,0.05),
        upperbound=quantile(V,0.95)
    )
end

stats=getstats.((:μ_1, :μ_2, :σ_1, :σ_2, :prob),[getindex.(res,i) for i in 1:5])

for is in eachindex(stats)
    println(parameters[is], " → ", stats[is])
end

# we can see that the true values lie inside the confidence interval.
