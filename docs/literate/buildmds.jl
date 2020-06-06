cd(@__DIR__)

using Literate

Literate.markdown("example_1.jl",outdir="../src/")
