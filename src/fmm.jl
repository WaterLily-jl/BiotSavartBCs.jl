# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3/π/4
@inline weighted(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = (-1)^i*@inbounds(ω[S,1]*r[i%2+1])/(r'*r)/π/2

# Closed-form lattice tail for a single periodic axis d of period L)
@inline function image_tail(ω,S::CartesianIndex{3},r0::SVector{3,Float32},i,d,L,M)
    ρ2 = zero(eltype(r0)); @inbounds for k in 1:3; k==d || (ρ2 += r0[k]*r0[k]); end
    sd = @inbounds r0[d]; s1 = sd-(M+0.5f0)*L; s2 = sd+(M+0.5f0)*L
    A = (s1/(ρ2*sqrt(ρ2+s1*s1)) - s2/(ρ2*sqrt(ρ2+s2*s2)) + 2f0/ρ2)/L  # Σ_{|n|>M} |r|⁻³
    B = (1f0/sqrt(ρ2+s2*s2) - 1f0/sqrt(ρ2+s1*s1))/L                    # Σ_{|n|>M} r_d|r|⁻³
    j = i%3+1; k = (i+1)%3+1                                           # (ω×r)_i = ω_j r_k - ω_k r_j
    @inbounds Σj = (j==d ? B : r0[j]*A); Σk = (k==d ? B : r0[k]*A)
    @inbounds (ω[S,j]*Σk - ω[S,k]*Σj)/(4f0π)
end

# Sum over sources at one interaction level
Base.@propagate_inbounds @fastmath function interaction(ω,Ti::CartesianIndex{Np1},l,depth) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    val = zero(eltype(ω))
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain),close(T,domain)
    l == depth && (Router = domain)
    if l == 1 # Top level
        # Do everything remaining inside buff=2
        for S in inR(Router,inside(size_u(ω)[1],buff=2))
            val += weighted(x-SVector{N,Float32}(S.I),S,i,ω)
        end
    elseif Rinner≠Router
        for S in Router
            S ∉ Rinner && (val += weighted(x-SVector{N,Float32}(S.I),S,i,ω))
        end
    end; val
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Interaction on targets. `symmetry` is a per-level extension hook that case scripts can override.
interaction!(ml,flat_targets) = @vecloop _interaction!(ml,lT) over lT ∈ flat_targets
@inline _interaction!(ml,lT) = ((l,T) = lT; ml[l][T] = symmetry(ml[l],T,l,length(ml)))
@inline symmetry(ω,T,args...) = interaction(ω,T,args...) # default: no applied symmetry

# Periodic-aware interaction for any level, for a single periodic axis d=perdir[1] (3D only)
Base.@propagate_inbounds @fastmath function pinteraction(ω,Ti::CartesianIndex{Np1},l,depth,perdir,nimages) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    Nu = size_u(ω)[1]
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    per = ntuple(k->Nu[k]-2, N)
    val = zero(eltype(ω))
    # discrete close image sum over the period-extended domain
    lo = ntuple(k-> k in perdir ? 2 - nimages*per[k]       : 2,       N) # domain extended
    hi = ntuple(k-> k in perdir ? Nu[k]-1 + nimages*per[k] : Nu[k]-1, N) # by nimages periods
    pdom = CartesianIndices(ntuple(k->lo[k]:hi[k],N))
    Router = l == depth ? pdom : remaining(T,pdom)
    Rinner = close(T,pdom)
    if !(l != 1 && Rinner == Router)
        b = l == 1 ? 2 : 1   # non-perdir interior buffer (buff=2 finest, buff=1 coarse)
        clip = CartesianIndices(ntuple(k-> k in perdir ? (lo[k]:hi[k]) : (1+b:Nu[k]-b), N))
        for S in Router
            S in clip || continue                  # non-perdir interior clip (no-op in perdir)
            (l != 1 && S in Rinner) && continue    # exclude inner near box (finer levels handle it)
            Sw = CartesianIndex(ntuple(k-> k in perdir ? mod(S[k]-2,per[k])+2 : S[k], N)) # warp S to array indices
            val += weighted(x-SVector{N,Float32}(S.I),Sw,i,ω)
        end
    end
    # closed-form far-field images of every source cell added once at the coarsest level
    if l == depth
        d = perdir[1]; Lf = Float32(per[d])
        for S in inside(Nu)
            val += image_tail(ω,S,x-SVector{N,Float32}(S.I),i,d,Lf,nimages)
        end
    end
    val
end

# Periodic interaction: periodic-aware shell sum at every level (see pinteraction).
periodic_interaction!(ml,flat_targets,perdir,nimages) = @vecloop _periodic_interaction!(ml,lT,perdir,nimages) over lT ∈ flat_targets
@inline function _periodic_interaction!(ml,lT,perdir,nimages)
    l,Ti = lT; ml[l][Ti] = pinteraction(ml[l],Ti,l,length(ml),perdir,nimages)
end

# Biot-Savart BC using FMM. Tuple{} dispatch gives zero overhead on the non-periodic path.
fmmBC!(ml,targets,flat_targets,::Tuple{}=(),nimages=0) = (interaction!(ml,flat_targets); project!(ml,targets))
function fmmBC!(ml,targets,flat_targets,perdir,nimages=0)
    periodic_interaction!(ml,flat_targets,perdir,nimages); project!(ml,targets)
end