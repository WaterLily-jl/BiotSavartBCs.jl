using WaterLily,BiotSavartBCs,StaticArrays
using BiotSavartBCs: MLArray,collect_targets,flatten_targets,fill_ω!,biotBC!
using SpecialFunctions,ForwardDiff,Plots
figdir = joinpath(@__DIR__,"..","tex","fig") # paper figures are written here
function lamb_dipole(N;D=3N/4,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    return function uλ(i,xy)
        x,y = xy .- (N-2)/2
        i==1 && return ForwardDiff.derivative(y->ψ(x,y),y)+1+U
        -ForwardDiff.derivative(x->ψ(x,y),x)
    end
end
function fill_lamb(N,D;T=Float64,mem=Array)
    uλ = lamb_dipole(N;D)
    u = zeros(T,(N,N,2))|>mem; apply!(uλ,u)
    f = zeros(T,(N,N,2))|>mem
    ω = MLArray(f); fill_ω!(ω,u)
    return u,ω
end

using BenchmarkTools,CUDA
function btime_biotBC!(u,U,ω,targs,ftargs,dist;fmm=false)
    @eval BiotSavartBCs.close(T::CartesianIndex{2}) = T-$dist*oneunit(T):T+$dist*oneunit(T)
    # we do not want the compilation of the above function in the time
    r = fill!(similar(u,size(u)[1:end-1]),0)
    btime(@benchmark CUDA.@sync biotBC!($r,$u,$U,$ω,$targs,$ftargs;fmm=$fmm))
end

btime(b) = minimum(b).time
import BiotSavartBCs: close,remaining
function lamb_test(N,D,dist,U=(1,0);fmm=false)
    u,ω = fill_lamb(N,D); ue = copy(u)
    tar = collect_targets(ω); ftar = flatten_targets(tar)
    r = zeros(eltype(u),size(u)[1:end-1])
    time = btime(@benchmark CUDA.@sync biotBC!($r,$u,$U,$ω,$tar,$ftar;fmm=$fmm))
    # targets are the full domain
    R = map(ωᵢ->vcat(CartesianIndices(ωᵢ)...),ω); fR = flatten_targets(R)
    biotBC!(r,u,U,ω,R,fR;fmm=fmm)
    # error
    p = zeros(Float64,(N,N));
    sdf(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2
    WaterLily.@loop (p[I] = sdf(I)>1 ? √sum(abs2,ue[I,:].-u[I,:]) : 0) over I ∈ inside(p,buff=0)
    return p,time
end

# # Check dependency on dist = size of kernel
pow = 6; N,D = 2^(pow+3)+2,2^pow
pmap(p) = log10(p+10^-6.5)
dis = range(1,N÷2,length=30)
d(I) = √sum(abs2,loc(0,I) .- (N-2)/2) - D/2
stats(p,i) = pmap(maximum(p[I] for I in CartesianIndices(p) if dis[i-1]<d(I)≤dis[i]))

data = []; duration = [];
for dist ∈ 2 .^ collect(0:pow+1)
    @show dist
    @eval BiotSavartBCs close(T::CartesianIndex{2}) = T-$dist*oneunit(T):T+$dist*oneunit(T)
    p,time = lamb_test(N,D,Int(dist)); push!(duration,time)
    flood(pmap.(p),clims=(-6,-1),border=:none,cfill=:Greens)
    savefig("lamb_dipole_error_dist$(dist).png")
    push!(data,[stats(p,i) for i in 2:lastindex(dis)])
end
using JLD2
save_object("error_dists.jld2",[data,duration])

colors = colormap("Blues",pow+5)
plt = plot(xlabel="d/2R",ylabel="max(|uₑ|/U)",ylims=(10^-6,0.1));
for (dist,vec) in enumerate(data)
    plot!(plt,collect(dis)[2:end]./D,10.0.^vec,label="S=$(2 .^(dist-1))",
          c=colors[2+dist],yaxis=:log,legendtitle="Multilevel")
end
plt
savefig(joinpath(figdir,"lamb_dipole_error_dists.png"))
 
# speed up on boundary only
pow = 8; N,D = 2^(pow+3)+2,2^pow
u,ω = fill_lamb(N,D); U=SA[1,0]
uC,ωC = fill_lamb(N,D,mem=CuArray);
tar = collect_targets(ω); ftar = flatten_targets(tar)
tarC = CUDA.CuArray.(collect_targets(ω)); ftarC = flatten_targets(tarC)
duration_fmm =   [btime_biotBC!(u,U,ω,tar,ftar,dist;fmm=true) for dist ∈ 2 .^ collect(0:pow-1)]
duration_tree =  [btime_biotBC!(u,U,ω,tar,ftar,dist;fmm=false) for dist ∈ 2 .^ collect(0:pow-1)]
duration_fmmC =  [btime_biotBC!(uC,U,ωC,tarC,ftarC,dist;fmm=true) for dist ∈ 2 .^ collect(0:pow-1)]
duration_treeC = [btime_biotBC!(uC,U,ωC,tarC,ftarC,dist;fmm=false) for dist ∈ 2 .^ collect(0:pow-1)]
jldsave("lamb_dipole_speedup.jld2";times=[duration_tree,duration_fmm,duration_treeC,duration_fmmC])

using Plots
reds = colormap("Reds",8)[3:end]
blues = colormap("Blues",8)[3:end]
plot(2.0.^collect(0:pow-1),duration_fmmC[end]./duration_tree,xlabel="S",ylabel="speedup",
label="Tree - CPU",lw=2,c=reds[4],yaxis=:log,xaxis=:log)
plot!(2.0.^collect(0:pow-1),duration_fmmC[end]./duration_treeC,lw=2,c=reds[6],ls=:dashdot,label="Tree - GPU")
plot!(2.0.^collect(0:pow-1),duration_fmmC[end]./duration_fmm,lw=2,c=blues[4],label="FMℓM - CPU")
plot!(2.0.^collect(0:pow-1),duration_fmmC[end]./duration_fmmC,lw=2,c=blues[6],ls=:dashdot,label="FMℓM - GPU")
xlims!(1,2^pow); ylims!(.1,10^3)
savefig(joinpath(figdir,"lamb_dipole_speedup.png"))
