# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3/π/4
@inline weighted(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = (-1)^i*@inbounds(ω[S,1]*r[i%2+1])/(r'*r)/π/2

# Velocity induced at a target by the sources at one level, of global size G. Under MPI the
# windows are global and `ω` holds this rank's block (at offset o) of the sources
Base.@propagate_inbounds @fastmath function induced(ω,Ti::CartesianIndex{Np1},(G,o),l,depth,d...) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    val = zero(eltype(ω))
    domain = inside(G)
    Router,Rinner = remaining(T,domain,d...),close(T,domain,d...)
    l == depth && (Router = domain)
    if l == 1 # Top level
        # Do everything remaining in the source cells
        for S in inR(inR(Router,sources(G,d...)),block(ω,o))
            val += weighted(x-SVector{N,Float32}(S.I),S-o,i,ω)
        end
    elseif Rinner≠Router
        for S in inR(Router,block(ω,o))
            S ∉ Rinner && (val += weighted(x-SVector{N,Float32}(S.I),S-o,i,ω))
        end
    end; val
end
block(ω,o) = inside(size_u(ω)[1]) .+ o # the cells of ω, in global indices
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Induced velocity on targets, including the images across the `symmetry` faces
induced!(v,ml,ftar,idx,lv,perdir=(),symmetry=()) = @loop _induced!(v,ml,ftar,k,lv,perdir,symmetry) over k ∈ idx
@inline _induced!(v,ml,ftar,k,lv,perdir,symmetry) = ((l,T) = ftar[k]; v[k] = isempty(perdir) ? images(ml[l],T,symmetry,lv[l],l,length(ml)) : periodic(ml[l],T,lv[l],l,length(ml),perdir...))

# Symmetry planes on domain `faces`: sum the velocity induced at the target and all of its images
@inline images(ω,T,::Tuple{},args...) = induced(ω,T,args...)
@inline function images(ω,T,faces,lv,args...)
    T′,sgn = image(T,first(lv),first(faces))
    images(ω,T,Base.tail(faces),lv,args...)+sgn*images(ω,T′,Base.tail(faces),lv,args...)
end

# Periodic in d: the domain and its nearest images at every level, then the images |n|≥2 at
# the coarsest level using Σ r/|r|³ ≈ ∫ r/|r|³ ds/L (the 2D kernel) minus the middle three periods
Base.@propagate_inbounds @fastmath function periodic(ω,Ti,(G,o),l,depth,d)
    L = G[d]-2; Δ = L*δ(d,Ti)
    val = induced(ω,Ti-Δ,(G,o),l,depth,d)+induced(ω,Ti,(G,o),l,depth,d)+induced(ω,Ti+Δ,(G,o),l,depth,d)
    l < depth && return val
    i,T,e = last(Ti),front(Ti),SVector{3,Float32}(ntuple(k->k==d,3))
    x = shifted(T,i)+SVector{3,Float32}(T.I)
    for S in block(ω,o)
        r = x-SVector{3,Float32}(S.I); ρ = r-r[d]*e; ρ² = ρ'*ρ
        F(s) = (ρ*s/ρ²-e)/√(ρ²+s^2) # ∫ r/|r|³ ds
        R = (2ρ/ρ²+F(r[d]-1.5f0L)-F(r[d]+1.5f0L))/L
        val += permute((j,k)->ω[S-o,j]*R[k],i)/π/4
    end; val
end
