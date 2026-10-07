# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂,face
fill_ω!(ml::Tuple,u,perdir=()) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u,perdir); restrict!(ml))
fill_ω!(ω::AbstractArray{<:Any,4},u,perdir=()) = @loop (ω[I,1] = centered_curl(1,I,u); ω[I,2] = centered_curl(2,I,u); ω[I,3] = centered_curl(3,I,u)) over I ∈ sources(size_u(ω)[1],perdir...)
fill_ω!(ω::AbstractArray{<:Any,3},u,perdir=()) = @loop (ω[I,1] = centered_curl(3,I,u); ω[I,2] = zero(eltype(ω))) over I ∈ sources(size_u(ω)[1],perdir...)
Base.@propagate_inbounds centered_curl(i,I,u) = (j=i%3+1; k=(i+1)%3+1; ∂(k,j,I,u)-∂(j,k,I,u))

# Incompressible & irrotational ghosts
function pflowBC!(u)
    N,n = size_u(u)
    @inline edge(I,j,val) = 2<I.I[j]<N[j] ? val : zero(val)
    for i ∈ 1:n # loops launch over the faces normal to i, see WaterLily.face
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j] - edge(I,j,∂(j,CartesianIndex(I+δ(i,I),i),u)) over I ∈ face(slice_u(N,i,j,1),i)
            @loop u[I,j] = u[I-δ(i,I),j] + edge(I,j,∂(j,CartesianIndex(I,i),u)) over I ∈ face(slice_u(N,i,j,N[i]),i)
        end # Normal direction ghosts, div=0
        @loop u[I,i] += WaterLily.div(I,u) over I ∈ face(WaterLily.slice(N.-1,1,i,2),i)
    end
end
slice_u(N::NTuple{n},i,j,s) where n = CartesianIndices(ntuple(k-> k==i ? (s:s) : k==j ? (2:N[k]) : (2:N[k]-1),n))

# Biot-Savart BCs
function biotBC!(u,U,ml,targets,flat_targets;fmm=true,perdir=(),symmetry=())
    fmm ? fmmBC!(ml,targets,flat_targets,perdir,symmetry) : treeBC!(ml,targets[1],perdir) # Fill ml[targets]=uᵥ
    @loop _biotBC!(u,U,ml[1],Ii) over Ii ∈ targets[1]           # Set u = uᵥ+U
end
@inline function _biotBC!(u,U,uᵥ,Ii)
    i,I = last(Ii),front(Ii); lower = I.I[i]==1
    u[I+(lower ? δ(i,I) : zero(I)),i] = U[i]+uᵥ[Ii]
end

using Atomix
# Biot-Savart BCs + residual update
function biotBC_r!(r,u,U,ml,targets,flat_targets;fmm=true,perdir=(),symmetry=())
    fmm ? fmmBC!(ml,targets,flat_targets,perdir,symmetry) : treeBC!(ml,targets[1],perdir) # Fill ml[targets]=uᵥ
    @loop _biotBC_r!(r,u,U,ml[1],Ii) over Ii ∈ targets[1]       # Update the u,r
    fix_resid!(r,u,targets[1])                                     # Fix u,r
end
@inline function _biotBC_r!(r,u,U,uᵥ,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uₙ = U[i]+uᵥ[Ii]
    uI = lower ? Ii+δ(i,Ii) : Ii; uₙ⁰ = u[uI]; u[uI] = uₙ
    Atomix.@atomic r[I+(lower ? δ(i,I) : -δ(i,I))] += (uₙ-uₙ⁰)*(lower ? -1 : 1)
end

# Correct the global residual s.t. sum(r)=0
fix_resid!(r,u,targets,fix=sum(r)/length(targets)) = @loop _fix_resid!(r,u,fix,Ii) over Ii ∈ targets
@inline function _fix_resid!(r,u,fix,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    u[I+ (lower ? δ(i,I) : zero(I)),i] += fix*(lower ? 1 : -1)
    Atomix.@atomic r[I+ (lower ? δ(i,I) : -δ(i,I))] -= fix
end
