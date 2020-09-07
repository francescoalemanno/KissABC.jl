using KissABC
using Documenter
#using Literate
#using Plots
#ENV["GKSwstype"] = "100"
#@show joinpath(@__DIR__, "src")
#=Literate.markdown(
    joinpath(@__DIR__, "literate", "example_1.jl"),
    joinpath(@__DIR__, "src"),
    documenter = true,
)=#
#=Literate.markdown(
    joinpath(@__DIR__, "literate", "index.jl"),
    joinpath(@__DIR__, "src"),
    documenter = true,
)
@info "Literate finished."=#
makedocs(;
    modules = [KissABC],
    authors = "Francesco Alemanno <francescoalemanno710@gmail.com> and contributors",
    repo = "https://github.com/JuliaApproxInference/KissABC.jl/blob/{commit}{path}#L{line}",
    sitename = "KissABC.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliaapproxinference.github.io/KissABC.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
#        "Example: Gaussian Mixture" => "example_1.md",
        #"Reference" => "reference.md",
    ],
)

deploydocs(; repo = "github.com/JuliaApproxInference/KissABC.jl", push_preview = true)
