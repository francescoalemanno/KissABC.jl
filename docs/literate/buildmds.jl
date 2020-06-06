using Literate
d=@__DIR__
od=pwd()
cd(d)
cd("../src")
Literate.markdown(joinpath(@__DIR__,"example_1.jl"))

cd(od)
