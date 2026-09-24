# Extend Multi-level up/down indexing for CartesianRanges
using Base: front,last
using WaterLily: up,down
WaterLily.up(R::CartesianIndices) = first(up(first(R))):last(up(last(R)))
WaterLily.down(R::CartesianIndices) = down(first(R)):down(last(R))

# Generalize inside(array) for any thickness of buffer cells
using WaterLily: inside
WaterLily.inside(ndims::NTuple{n};buff=1) where n = CartesianIndices(map(N->(1+buff:N-buff),ndims))
# Interior u-indices with ghost buffer `buff`; along periodic directions (`perdir`) use buff=1
inside_u(a;buff=1,perdir=()) = inside_u(size_u(a)[1],buff,perdir)
inside_u(ndims::NTuple{n},buff,perdir=()) where n = CartesianIndices((ntuple(k-> k in perdir ? (2:ndims[k]-1) : (1+buff:ndims[k]-buff), n)...,1:n))

# Local CartesianRange around a target T, with size specialized for 2D and 3D
# note: These sources are too "close" to T for interaction at this level (unless we're at the top level)
close(T::CartesianIndex{2}) = T-4oneunit(T):T+4oneunit(T)
close(T::CartesianIndex{3}) = T-2oneunit(T):T+2oneunit(T)
close(T,R) = inR(close(T),R)
inR(x,R) = max(first(x),first(R)):min(last(x),last(R))

# CartesianRange corresponding to close(T,R) on the next coarser level
# note: These are the only remaining contributions missing from the FMM sum (unless we're at the bottom level)
remaining(T,R) = up(close(down(T),down(R)))

# Collect "targets" on the faces of a MLArray
using Base.Iterators
slice(dims::NTuple{N},i,s) where N = CartesianIndices((ntuple( k-> k==i ? (s:s) : (2:dims[k]-1), N-1)...,(i:i)))
faces(dims::NTuple{N},off) where N = flatmap(i->flatmap(s->slice(dims,i,s), ((-i∈off ? () : (1,))...,(i∈off ? () : (dims[i],))...)),1:N-1)
collect_targets(ω,off=(),perdir=()) = map(ωᵢ->collect(faces(size(ωᵢ),(off...,perdir...,(-).(perdir)...))),ω)
flatten_targets(targets) = mapreduce(((level,targets),)->map(T->(level,T),targets),vcat,enumerate(targets))

"""
   image(T::CartesianIndex,dims,face=2)

Reflect target `T` across the specified domain face of an array with dimensions `dims`.
The `face` argument specifies which face to reflect across, where `±i` corresponds to the low/high side of dimension `i`.
Returns a tuple containing the reflected target index and the contriution sign.
"""
@inline function image(T::CartesianIndex,dims,face=2)
    i = abs(face)
    d = face>0 ? 2dims[i]-2T.I[i]-1 : 3-2T.I[i]
    return T+d*WaterLily.δ(i,T), T.I[end]==i ? -1 : 1
end