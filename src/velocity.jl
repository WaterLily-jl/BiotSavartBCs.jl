# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂,face
fill_ω!(ml::Tuple,u,perdir=(),src=sources(size_u(first(ml))[1],perdir...)) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u,perdir,src); restrict!(ml))
fill_ω!(ml::Tuple,u,perdir,src,R) = (ω=first(ml); fill_ω!(ω,u,perdir,inR(R,src)); restrict!(ml,R)) # refresh R only
fill_ω!(ω::AbstractArray{<:Any,4},u,perdir=(),R=sources(size_u(ω)[1],perdir...)) = @loop (ω[I,1],ω[I,2],ω[I,3]) = WaterLily.ω(I,vel(u)) over I ∈ R
fill_ω!(ω::AbstractArray{<:Any,3},u,perdir=(),R=sources(size_u(ω)[1],perdir...)) = @loop (ω[I,1] = WaterLily.ω(I,vel(u)); ω[I,2] = zero(eltype(ω))) over I ∈ R

# Projected velocity u-μ₀∇p evaluated lazily from u=(u,μ₀,p)
struct Projected{U,M,P}; u::U; μ₀::M; p::P; end
Base.@propagate_inbounds Base.getindex(a::Projected,I::CartesianIndex,i) = a.u[I,i]-a.μ₀[I,i]*∂(i,I,a.p)
@inline vel(u) = u
@inline vel(u::Tuple) = Projected(u...)

# Incompressible & irrotational ghosts on the domain faces. Under MPI `u` is the block at offset o
# of a domain of size G: the edge tests are in global indices and the seams are left to the halos
import WaterLily: CIj
function pflowBC!(u,(G,o)=(size_u(u)[1],zero(CartesianIndex{ndims(u)-1})))
    N,n = size_u(u); lo,hi = map(iszero,o.I),map((o,N,G)->o+N==G,o.I,N,G) # on the domain faces?
    for i ∈ 1:n
        @loop pflow!(u,Ii,i,N,(G,o),lo[i],hi[i]) over Ii ∈ face(CartesianIndices((ntuple(k->k==i ? (1:1) : (lo[k] ? 1 : 2):(hi[k] ? N[k] : N[k]-1),length(N))...,1:n)),i)
    end
end
@inline function pflow!(u,Ii,i,N,(G,o),lo,hi)
    I,j = front(Ii),last(Ii)
    if j≠i
        lo && (u[Ii] = curl0(u,I,i,j,(G,o),true)); hi && (u[CIj(i,Ii,N[i])] = curl0(u,CIj(i,I,N[i]),i,j,(G,o),false))
    elseif any(ntuple(k->k≠i && I.I[k]+o[k]∈(1,G[k]),length(N)))
        lo && (u[Ii] = u[I+δ(i,I),i])
    elseif lo
        u[Ii] = sum(ntuple(k->k==i ? u[I+δ(i,I),i] : curl0(u,I+δ(k,I),i,k,(G,o),true)-curl0(u,I,i,k,(G,o),true),length(N)))
    end
end
@inline function curl0(u,I,i,j,(G,o),lower)
    F = lower ? I+δ(i,I) : I; J = lower ? F : I-δ(i,I)
    u[J,j] - (lower ? 1 : -1)*(2<I.I[j]+o[j]<G[j] ? u[F,i]-u[F-δ(j,F),i] : zero(eltype(u)))
end

using Atomix
# Biot-Savart BCs + residual update. FMM sums only the targets in `idx`: the rest keep their `own` sums
function biotBC!(r,u,U,ml,tg;fmm=true,perdir=(),symmetry=(),idx=tg.blk)
    (;lv,tar,ftar,pmap,ltar,own,tot) = tg
    fmm ? induced!(own,ml,ftar,idx,lv,perdir,symmetry) : treeBC!(own,ml,tar[1],lv,perdir)
    global_allreduce!(copyto!(tot,own)); fmm && project!(tot,pmap,tar) # Fill tot=uᵥ
    @loop _biotBC!(r,u,U,tot,kI) over kI ∈ ltar                      # Update the u,r
    fix_resid!(r,u,ltar)                                              # Fix u,r
end
@inline function _biotBC!(r,u,U,uᵥ,(k,Ii))
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uₙ = U[i]+uᵥ[k]
    uI = lower ? Ii+δ(i,Ii) : Ii; uₙ⁰ = u[uI]; u[uI] = uₙ
    Atomix.@atomic r[I+(lower ? δ(i,I) : -δ(i,I))] += (uₙ-uₙ⁰)*(lower ? -1 : 1)
end

# Correct the global residual s.t. sum(r)=0
fix_resid!(r,u,ltar,fix=global_sum(r)/global_length(ltar)) = @loop _fix_resid!(r,u,fix,kI) over kI ∈ ltar
@inline function _fix_resid!(r,u,fix,(_,Ii))
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    u[I+ (lower ? δ(i,I) : zero(I)),i] += fix*(lower ? 1 : -1)
    Atomix.@atomic r[I+ (lower ? δ(i,I) : -δ(i,I))] -= fix
end
