# Tree sum over sources at all levels (global windows over this rank's blocks, as in `induced`)
Base.@propagate_inbounds @fastmath function tree(ml,Ti::CartesianIndex{Np1},lv) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)

    #Top level
    ω,(G,o) = first(ml),first(lv)
    val = zero(eltype(ω))
    domain = inside(G)
    Router = remaining(T,domain)
    Rinner = inR(Router,inside(G,buff=2))
    # Do everything remaining inside buff=2
    for S in inR(Rinner,block(ω,o))
        val += weighted(x-SVector{N,Float32}(S.I),S-o,i,ω)
    end
    # @show Rinner, val

    # Loop down levels
    x = x .- 1.5f0 # adjust origin for scaling
    for l in 2:lastindex(ml)
        ω,(G,o) = ml[l],lv[l]; T = down(T)
        domain = inside(G)
        Rinner = close(T,domain)
        Router = l == lastindex(ml) ? domain : remaining(T,domain)
        Rinner ≠ Router && for S in inR(Router,block(ω,o))
            S ∉ Rinner && (val += weighted(x-(SVector{N,Float32}(S.I) .- 1.5f0)*2^(l-1),S-o,i,ω))
        end
        # @show T, Rinner, Router, val
    end; val
end

# Biot-Savart BC using the tree sum
treeBC!(v,ml,targets,lv,::Tuple{}=()) = @loop v[k]=tree(ml,targets[k],lv) over k ∈ eachindex(targets)
treeBC!(v,ml,targets,lv,perdir) = throw(ArgumentError("periodic Biot-Savart BCs require fmm=true"))
