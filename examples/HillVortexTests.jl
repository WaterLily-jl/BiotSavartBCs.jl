using WaterLily,BiotSavartBCs,StaticArrays
using BiotSavartBCs: MLArray,collect_targets,flatten_targets,fill_ω!,biotBC!

hill_vortex(N;D=3N/4) = function uλ(i,xyz)
    q = xyz .- (N-2)/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
    v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*cos(θ)
    v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*sin(θ)
    i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
    i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
    cos(θ)*v_r-sin(θ)*v_θ
end
function fill_hill(N,D;T=Float64,mem=Array)
    u = zeros(T,(N,N,N,3))|>mem; apply!(hill_vortex(N;D),u)
    p = zeros(T,(N,N,N))|>mem;
    f = zeros(T,(N,N,N,3))|>mem # ML points to this
    ω = MLArray(f); fill_ω!(ω,u)
    return p,u,ω
end

# Test serial vs multi-threaded vs GPU
using CUDA,BenchmarkTools
CUDA.allowscalar(false)
pow = 6; N,D = 2^pow+2,2^(pow-3)
p,u,ω = fill_hill(N,D,T=Float32);
pC,uC,ωC = fill_hill(N,D,T=Float32,mem=CuArray);

# test scaling
U = SA[0.,0.,1.]
tar = collect_targets(ω); ftar = flatten_targets(tar)
tarC = CUDA.CuArray.(collect_targets(ω)); ftarC = flatten_targets(tarC)
@btime biotBC!($p,$u,$U,$ω,$tar,$ftar;fmm=false); # 2.501ms 3.77KiB
@btime biotBC!($p,$u,$U,$ω,$tar,$ftar;fmm=true); # 0.671ms 3.84KiB
@btime CUDA.@sync biotBC!($pC,$uC,$U,$ωC,$tarC,$ftarC;fmm=false); # 1.102ms 4.84KiB
@btime CUDA.@sync biotBC!($pC,$uC,$U,$ωC,$tarC,$ftarC;fmm=true); # 0.319ms 15.61KiB 

# timmings
btime(b) = minimum(b).time
function btime_biotBC!(u,U,ω,targs,ftargs,dist;fmm=false)
    # we have to owerwrite the function before we call btime
    @eval BiotSavartBCs.close(T::CartesianIndex{3}) = T-$dist*oneunit(T):T+$dist*oneunit(T)
    # we do not want the compilation of the above function in the time
    r = fill!(similar(u,size(u)[1:end-1]),0)
    btime(@benchmark CUDA.@sync biotBC!($r,$u,$U,$ω,$targs,$ftargs;fmm=$fmm))
end

# run some benchmark
duration_tree =  [btime_biotBC!(u,U,ω,tar,ftar,dist;fmm=false) for dist ∈ 2 .^ collect(0:pow-1)]
duration_fmm =   [btime_biotBC!(u,U,ω,tar,ftar,dist;fmm=true) for dist ∈ 2 .^ collect(0:pow-1)]
duration_treeC = [btime_biotBC!(uC,U,ωC,tarC,ftarC,dist;fmm=false) for dist ∈ 2 .^ collect(0:pow-1)]
duration_fmmC =  [btime_biotBC!(uC,U,ωC,tarC,ftarC,dist;fmm=true) for dist ∈ 2 .^ collect(0:pow-1)]

using Plots,JLD2
figdir = joinpath(@__DIR__,"..","tex","fig") # paper figures are written here
reds = colormap("Reds",8)[3:end]
blues = colormap("Blues",8)[3:end]
plot(2.0.^collect(0:5),duration_fmmC[end]./duration_tree,xlabel="S",ylabel="speedup",
     label="Tree - CPU",lw=2,c=reds[4], yaxis=:log,xaxis=:log, xlims=(1,32),ylims=(0.1,10000))
     plot!(2.0.^collect(0:5),duration_fmmC[end]./duration_treeC,lw=2,c=reds[6],ls=:dashdot,label="Tree - GPU")
     plot!(2.0.^collect(0:5),duration_fmmC[end]./duration_fmm,lw=2,c=blues[4],label="FMℓM - CPU")
plot!(2.0.^collect(0:5),duration_fmmC[end]./duration_fmmC,lw=2,c=blues[6],ls=:dashdot,label="FMℓM - GPU")
jldsave("Hill_speedup.jld2";times=[duration_tree,duration_fmm,duration_treeC,duration_fmmC])
savefig(joinpath(figdir,"Hill_speedup_dists.png"))

# Check error scaling with dist
@inline d(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2;
@inline J(I) = I+CartesianIndex(0,N÷2,0)
function hill_error(u,ω,U=(0,0,1))
    ue = copy(u); u .= 0 # make sure it's empty
    # targets are the full domain, but we only care about a single slice
    R = map(ωᵢ->vcat(CartesianIndices(ωᵢ)...),ω); fR = flatten_targets(R)
    biotBC!(zeros(eltype(u),size(u)[1:end-1]),u,U,ω,R,fR;fmm=false) # can only use tree
    p = zeros(eltype(ue),(N,1,N))
    WaterLily.@loop (p[I] = d(J(I))>1 ? √sum(abs2,ue[J(I),:].-u[J(I),:]) : 0) over I ∈ CartesianIndices(p)
    return p
end

pow = 7; N,D = 2^pow+2,2^(pow-3)
pmap(p) = log10(p+10^-6.5)
dis = range(1,N÷2,length=30)
stats(p,i) = pmap(maximum(p[I] for I in CartesianIndices(p) if dis[i-1]<d(J(I))≤dis[i]))
data = [];
for dist ∈ 2 .^ collect(0:pow-3)
    @show dist
    @eval BiotSavartBCs.close(T::CartesianIndex{3}) = T-$dist*oneunit(T):T+$dist*oneunit(T)
    _,u,ω = fill_hill(N,D); p = hill_error(u,ω)
    flood(pmap.(p[:,1,:]),clims=(-6,-1),border=:none,cfill=:Greens)
    savefig("Hill_error_dist$(dist).png")
    push!(data,[stats(p,i) for i in 2:lastindex(dis)])
end
save_object("Hill_error.jld2",data) 

# data = load_object("Hill_error.jld2")[1]

colors = colormap("Blues",pow+2)
plt = plot(xlabel="d/2R",ylabel="max(|uₑ|/U)",ylims=(10^-6,.1));
for (dist,vec) in enumerate(data)
    plot!(plt,collect(dis)[2:end]./D,10.0.^vec,label="S=$(2 .^(dist-1))",
          c=colors[2+dist],yaxis=:log,legendtitle = "Multilevel")
end
plt
savefig(joinpath(figdir,"Hill_error_dists.png"))