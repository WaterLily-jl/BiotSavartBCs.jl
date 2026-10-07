# Tree sum over sources at all levels
Base.@propagate_inbounds @fastmath function tree(ml,Ti::CartesianIndex{Np1}) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)

    #Top level
    ω = first(ml)
    val = zero(eltype(ω))
    domain = inside(size_u(ω)[1])
    Router = remaining(T,domain)
    Rinner = inR(Router,inside(size_u(ω)[1],buff=2))
    # Do everything remaining inside buff=2
    for S in Rinner
        val += weighted(x-SVector{N,Float32}(S.I),S,i,ω)
    end
    # @show Rinner, val

    # Loop down levels
    x = x .- 1.5f0 # adjust origin for scaling
    for l in 2:lastindex(ml)
        ω = ml[l]; T = down(T)
        domain = inside(size_u(ω)[1])
        Rinner = close(T,domain)
        Router = l == lastindex(ml) ? domain : remaining(T,domain)
        Rinner ≠ Router && for S in Router
            S ∉ Rinner && (val += weighted(x-(SVector{N,Float32}(S.I) .- 1.5f0)*2^(l-1),S,i,ω))
        end
        # @show T, Rinner, Router, val
    end; val
end

# Biot-Savart BC using the tree sum
treeBC!(ml,targets,::Tuple{}=()) = @loop ml[1][Ii]=tree(ml,Ii) over Ii ∈ targets
treeBC!(ml,targets,perdir) = throw(ArgumentError("periodic Biot-Savart BCs require fmm=true"))
