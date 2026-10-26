# Vector multi-level constructor (top level points to u, doesn't copy)
using WaterLily: size_u
# @loop takes its backend from the first variable of the loop body, which can be a multi-level tuple
WaterLily.get_backend(ml::NTuple) = WaterLily.get_backend(first(ml))
function MLArray(u,perdir=())
    N,n = size_u(u)
    isempty(perdir) || (n==3 && length(perdir)==1) || throw(ArgumentError("periodic Biot-Savart BCs require 3D and a single periodic direction"))
    levels = []
    I = CartesianIndex(ntuple(i-> i==1 ? 1 : 2, n))
    while true
        N = @. 1+N÷2; R = inside(N)
        close(I,R,perdir...) == R && break
        push!(levels,N)
        any(N .%2 .≠0) && break
    end
    zeros_like_u(N,n) = (y = similar(u,N...,n); fill!(y,0); y)
    return (u,map(N->zeros_like_u(N,n),levels)...)
end

# Restrict(!) source data to a coarser level by pooling (summation)
# (only within the range R of the top level, coarsened level by level)
restrict!(ml::NTuple,R=inside(size_u(first(ml))[1])) = for l ∈ 2:lastindex(ml)
    R = down(R); restrict!(ml[l],ml[l-1],inR(R,inside(size_u(ml[l])[1])))
end
using WaterLily: @loop
restrict!(a,b,R=inside(size_u(a)[1])) = @loop a[Ii] = restrict(Ii,b) over Ii ∈ CartesianIndices((R.indices...,axes(a,ndims(a))))
@fastmath @inline function restrict(Ii::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(front(Ii))
     s += @inbounds(b[J,last(Ii)])
    end; s
end

# Project(!) the flat target data `v` from each coarser level by (bi)linear interpolation
project!(v,pmap,tar) = for l ∈ reverse(1:lastindex(tar)-1)
    @loop v[k] += interp(v,pmap[k]) over k ∈ sum(length,tar[1:l-1];init=0) .+ eachindex(tar[l])
end
@fastmath @inline interp(v,(a,b,c,d)::NTuple{4}) = 0.015625f0@inbounds(9v[a]+3v[b]+3v[c]+v[d])
@fastmath @inline interp(v,(a,b)::NTuple{2}) = 0.125f0@inbounds(3v[a]+v[b])
# The flat indices of the coarse targets each flat target interpolates from (0 on the coarsest level)
function project_map(ftar,dims)
    k = Dict(lT=>j for (j,lT) in enumerate(ftar))
    map(((l,T),)->map(C->get(k,(l+1,C),0),coarse(T,dims[min(l+1,end)])),ftar)
end
function coarse(Ii::CartesianIndex{4},N)
    I,i = front(Ii),last(Ii)
    dj,dk = step(I,i%3+1,N),step(I,(i+1)%3+1,N)
    map(J->CartesianIndex(J,i),down(I) .+ (zero(I),dj,dk,dj+dk))
end
function coarse(Ii::CartesianIndex{3},N)
    I,i = front(Ii),last(Ii); d = step(I,i%2+1,N)
    map(J->CartesianIndex(J,i),down(I) .+ (zero(I),d))
end
step(I,j,N,Ij=I.I[j]) = (Ij % 2 == 1 ?    # positive step,
    Ij ÷ 2 == N[j]-2 ? zero(I) : δ(j,I) : # don't step...
    Ij ÷ 2 == 1      ? zero(I) : -δ(j,I)) # past either edge