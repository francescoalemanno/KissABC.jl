using KissABC
using Documenter
using Literate
using Plots
ENV["GKSwstype"] = "100"
@show joinpath(@__DIR__,"src")
Literate.markdown(joinpath(@__DIR__,"literate","example_1.jl"),joinpath(@__DIR__,"src"),documenter=true)
Literate.markdown(joinpath(@__DIR__,"literate","index.jl"),joinpath(@__DIR__,"src"),documenter=true)

makedocs(;
    modules=[KissABC],
    authors="Francesco Alemanno <francescoalemanno710@gmail.com> and contributors",
    repo="https://github.com/francescoalemanno/KissABC.jl/blob/{commit}{path}#L{line}",
    sitename="KissABC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://francescoalemanno.github.io/KissABC.jl",
        assets=String[],
    ),
    pages=[
        "Basic Usage" => "index.md",
        "Example: Gaussian Mixture" => "example_1.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/francescoalemanno/KissABC.jl",
)
