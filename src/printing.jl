using PrettyTables: pretty_table
import Base.show
function print_chain(io, c::AISChain; kwargs...)
    nsamples, nparameters, nchains = size(c)
    println(
        io,
        "Object of type AISChain (total samples $(nsamples*nchains))\nnumber of samples: $nsamples\nnumber of parameters: $nparameters\nnumber of chains: $nchains",
    )
    q = [0.025, 0.25, 0.5, 0.75, 0.975]
    S = (x -> "$x%").(q .* 100)
    M = permutedims(
        hcat([quantile(vec(identity.(c[:, i, :])), q) for i = 1:nparameters]...),
        (2, 1),
    )
    pretty_table(
        io,
        M,
        S;
        row_names = ["Param $i" for i = 1:nparameters],
        crop = :none,
        kwargs...,
    )
end

function Base.show(io::IO, c::AISChain)
    print_chain(io, c)
end
function Base.show(io::IO, ::MIME"text/plain", c::AISChain)
    print_chain(io, c)
end
