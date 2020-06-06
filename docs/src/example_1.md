```@meta
EditURL = "<unknown>/example_1.jl"
```

```@example example_1
# A gaussian mixture model
```

First of all we define our model,

```@example example_1
using KissABC

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

data=model((1.0,0.0,0.2,2.0,0.4),10000)
```

let's look at the data

```@example example_1
using PyPlot
hist(data,100)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
