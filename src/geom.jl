# Extend Multi-level up/down indexing for CartesianRanges
using Base: front,last
using WaterLily: up,down
WaterLily.up(R::CartesianIndices) = first(up(first(R))):last(up(last(R)))
WaterLily.down(R::CartesianIndices) = down(first(R)):down(last(R))

# Generalize inside(array) for any thickness of buffer cells
using WaterLily: inside
WaterLily.inside(ndims::NTuple{n};buff=1) where n = CartesianIndices(map(N->(1+buff:N-buff),ndims))
inside_u(a;buff=1) = inside_u(size_u(a)[1],buff)
inside_u(ndims::NTuple{n},buff) where n = CartesianIndices((map(N->(1+buff:N-buff),ndims)...,1:n))
# Cells holding vorticity: buff=2, except along a periodic direction d
sources(ndims::NTuple{n},d...) where n = CartesianIndices(ntuple(k-> k∈d ? (2:ndims[k]-1) : (3:ndims[k]-2),n))

# Local CartesianRange around a target T, with size specialized for 2D and 3D
# note: These sources are too "close" to T for interaction at this level (unless we're at the top level)
close(T::CartesianIndex{2}) = T-4oneunit(T):T+4oneunit(T)
close(T::CartesianIndex{3}) = T-2oneunit(T):T+2oneunit(T)
# periodic in d with period L: once the window spans the period the images act 2D, so use the 2D size in-plane
close(T::CartesianIndex{3},d::Int,L::Int) = (w=CartesianIndex(ntuple(k->k==d || L>4 ? 2 : 4,3)); T-w:T+w)
close(T,R) = inR(close(T),R)
close(T,R,d) = inR(close(T,d,size(R,d)),R)
inR(x,R) = max(first(x),first(R)):min(last(x),last(R))

# CartesianRange corresponding to close(T,R) on the next coarser level
# note: These are the only remaining contributions missing from the FMM sum (unless we're at the bottom level)
remaining(T,R,d...) = up(close(CartesianIndex(fld.(T.I .+ 2,2)),down(R),d...))

# Collect "targets" on the faces of a MLArray
using Base.Iterators
slice(dims::NTuple{N},i,s) where N = CartesianIndices((ntuple( k-> k==i ? (s:s) : (2:dims[k]-1), N-1)...,(i:i)))
faces(dims::NTuple{N},off) where N = flatmap(i->flatmap(s->slice(dims,i,s), ((-i∈off ? () : (1,))...,(i∈off ? () : (dims[i],))...)),1:N-1)
collect_targets(ω,off=(),dims=map(a->size_u(a)[1],ω)) = map(N->collect(faces((N...,length(N)),off)),dims)
flatten_targets(targets) = mapreduce(((level,targets),)->map(T->(level,T),targets),vcat,enumerate(targets))

# The targets of `ω` (in global indices under MPI) and their data: `lv` (see `grids`), the flat list
# `ftar` and its interpolation map `pmap`, this rank's level 1 targets `ltar` as (flat index, index),
# the flat target sums `own` over this rank's sources and their totals `tot`, the targets `blk` that
# these sources reach, and the source cells `src`
function target_data(ω,off=(),perdir=();mem=Array)
    lv = grids(ω); (G,o),N = first(lv),size_u(first(ω))[1]
    tar = collect_targets(ω,(off...,perdir...,(-).(perdir)...),map(first,lv)); ftar = flatten_targets(tar)
    k,o′ = Dict(T=>j for (j,T) in enumerate(tar[1])),CartesianIndex(o,0)
    ltar = [(k[T+o′],T) for T in faces(size(first(ω)),()) if haskey(k,T+o′)] # not the seams
    own = fill!(similar(first(ω),length(ftar)),0)
    (;lv,tar=mem.(tar),ftar=mem(ftar),pmap=mem(project_map(ftar,map(first,lv))),ltar=mem(ltar),own,tot=similar(own),
      blk=mem(near_targets(CartesianIndices(N),lv,ftar,perdir)),src=inR(sources(G,perdir...) .- o,inside(N)))
end

@inline function image(T::CartesianIndex,dims,face=2)
    i = abs(face); normal = T.I[end]==i
    d = face>0 ? 2dims[i]-2T.I[i]-1 : 3-2T.I[i]+(normal && T.I[i]>1)
    return T+d*WaterLily.δ(i,T), normal ? -1 : 1
end