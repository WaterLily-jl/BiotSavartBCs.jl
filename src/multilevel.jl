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
restrict!(ml::NTuple) = for l ∈ 2:lastindex(ml)
    restrict!(ml[l],ml[l-1])
end
using WaterLily: @loop
restrict!(a,b) = @loop a[Ii] = restrict(Ii,b) over Ii ∈ inside_u(a)
@fastmath @inline function restrict(Ii::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(front(Ii))
     s += @inbounds(b[J,last(Ii)])
    end; s
end

# Project(!) target data to a coarser level by (bi)linear interpolation
project!(ml::Tuple,mltargets::Tuple) = for l ∈ reverse(1:lastindex(ml)-1)
    project!(ml[l],ml[l+1],mltargets[l])
end
project!(a,b,targets) = @loop a[Ii] += project(Ii,b) over Ii ∈ targets
@fastmath function project(Ii::CartesianIndex{4},b)
    I,i,N = front(Ii),last(Ii),size_u(b)[1]
    dj,dk = step(I,i%3+1,N),step(I,(i+1)%3+1,N)
    I,I2,I3,I4 = down(I) .+ (zero(I),dj,dk,dj+dk)
    0.015625f0@inbounds(9b[I,i]+3b[I2,i]+3b[I3,i]+b[I4,i])
end
@fastmath function project(Ii::CartesianIndex{3},b)
    I,i,N = front(Ii),last(Ii),size_u(b)[1]
    d = step(I,i%2+1,N)
    I,I2 = down(I) .+ (zero(I),d)
    0.125f0@inbounds(3b[I,i]+b[I2,i])
end
step(I,j,N,Ij=I.I[j]) = (Ij % 2 == 1 ?    # positive step,
    Ij ÷ 2 == N[j]-2 ? zero(I) : δ(j,I) : # don't step...
    Ij ÷ 2 == 1      ? zero(I) : -δ(j,I)) # past either edge