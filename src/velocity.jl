# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂,face
fill_ω!(ml::Tuple,u,perdir=()) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u,perdir); restrict!(ml))
fill_ω!(ml::Tuple,u,perdir,R) = (ω=first(ml); fill_ω!(ω,u,perdir,inR(R,sources(size_u(ω)[1],perdir...))); restrict!(ml,R)) # refresh R only
fill_ω!(ω::AbstractArray{<:Any,4},u,perdir=(),R=sources(size_u(ω)[1],perdir...)) = @loop (ω[I,1],ω[I,2],ω[I,3]) = WaterLily.ω(I,vel(u)) over I ∈ R
fill_ω!(ω::AbstractArray{<:Any,3},u,perdir=(),R=sources(size_u(ω)[1],perdir...)) = @loop (ω[I,1] = WaterLily.ω(I,vel(u)); ω[I,2] = zero(eltype(ω))) over I ∈ R

# Projected velocity u-μ₀∇p evaluated lazily from u=(u,μ₀,p)
struct Projected{U,M,P}; u::U; μ₀::M; p::P; end
Base.@propagate_inbounds Base.getindex(a::Projected,I::CartesianIndex,i) = a.u[I,i]-a.μ₀[I,i]*∂(i,I,a.p)
@inline vel(u) = u
@inline vel(u::Tuple) = Projected(u...)

# Incompressible & irrotational ghosts
import WaterLily: CIj
function pflowBC!(u)
    N,n = size_u(u)
    for i ∈ 1:n
        @loop pflow!(u,Ii,i,N) over Ii ∈ face(CartesianIndices((ntuple(k->k==i ? (1:1) : (1:N[k]),length(N))...,1:n)),i)
    end
end
@inline function pflow!(u,Ii,i,N)
    I,j = front(Ii),last(Ii)
    if j≠i
        u[Ii] = curl0(u,I,i,j,N,true); u[CIj(i,Ii,N[i])] = curl0(u,CIj(i,I,N[i]),i,j,N,false)
    elseif any(ntuple(k->k≠i && I.I[k]∈(1,N[k]),length(N)))
        u[Ii] = u[I+δ(i,I),i]
    else
        u[Ii] = sum(ntuple(k->k==i ? u[I+δ(i,I),i] : curl0(u,I+δ(k,I),i,k,N,true)-curl0(u,I,i,k,N,true),length(N)))
    end
end
@inline function curl0(u,I,i,j,N,lower)
    F = lower ? I+δ(i,I) : I; J = lower ? F : I-δ(i,I)
    u[J,j] - (lower ? 1 : -1)*(2<I.I[j]<N[j] ? u[F,i]-u[F-δ(j,F),i] : zero(eltype(u)))
end

using Atomix
# Biot-Savart BCs + residual update. FMM sums only the targets in `idx`: the rest keep their `own` sums
function biotBC!(r,u,U,ml,tg;fmm=true,perdir=(),symmetry=(),idx=tg.blk)
    (;tar,ftar,pmap,ltar,own,tot) = tg
    fmm ? induced!(own,ml,ftar,idx,perdir,symmetry) : treeBC!(own,ml,tar[1],perdir)
    copyto!(tot,global_allreduce(own)); fmm && project!(tot,pmap,tar) # Fill tot=uᵥ
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
