using WaterLily,StaticArrays,CUDA,BiotSavartBCs
const poisson_kwargs = (;tol=5e-5) # the paper's results need a tighter pressure tolerance than the 2e-3 default
using JLD2,Plots
using Logging; disable_logging(Logging.Debug)
figdir = joinpath(@__DIR__,"..","tex","fig") # paper figures are written here
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest
function make_sim_acc(dims; N=128, R=32, a=0.5, U=1, Re=1e3, thk=2, mem=Array, use_biotsavart=false, fmm=true)
    disk2D(x,t) = (z=x-SA[-R,0].-N/2; Rt=R-thk; y=z.-SA[0,clamp(z[2],-Rt,Rt)]; √sum(abs2,y)-thk)
    disk3D(x,t) = (z=x-SA[-R,0,0].-N/2; Rt=R-thk; r=√(z[2]^2+z[3]^2); √(z[1]^2+(r-min(r,Rt))^2)-thk)
    Ut(i,x,t::T) where T = i==1 ? convert(T,min(a*t/R,U)) : zero(T) # velocity BC
    body = length(dims)==2 ? AutoBody(disk2D) : AutoBody(disk3D)
    use_biotsavart && return BiotSimulation(dims, Ut, R; U, ν=U*R/Re, body, mem, fmm)
    Simulation(dims, Ut, R; U, ν=U*R/Re, body, mem)
end
slice(a::AbstractArray{T,3}) where T = Array(dropdims(a,dims=3))
slice(a::AbstractArray{T,2}) where T = Array(a)

# (Dim,BiotSavart,dis,FMM)
params = [(2,false,4,true),(3,false,2,true),(2,true,4,true), # 2D and 3D reflection + 2D Biot-Savart fmm
          (3,true,1,true),(3,true,2,true),(3,true,4,true), # 3D fmm dist ∈ [1,2,4]
          (3,true,1,false),(3,true,2,false),(3,true,4,false)]# 3D fmm dist ∈ [1,2,4]

          # size of the domain
N = 2^7; R = N/3
for (Dim,use_biotsavart,dist,fmm) ∈ params
    @show Dim,use_biotsavart,dist,fmm
    @eval BiotSavartBCs.close(T::CartesianIndex{$Dim}) = T-$dist*oneunit(T):T+$dist*oneunit(T) #overwrite the function
    sim = make_sim_acc(ntuple(i->N,Dim);N,R,mem=CUDA.CuArray,use_biotsavart,fmm);
    Rslice = Dim==2 ? inside(sim.flow.p) : CartesianIndices((2:N+1,2:N+1,N÷2+1:N÷2+1))
    forces = []; σ = []; p = [];
    fname = "disk_$(Dim)D_$(N)D_$(use_biotsavart)"
    WaterLily.logger(fname*".log")
    @time for t in 1:6
        while sim_time(sim)<t
            sim_step!(sim;remeasure=false,poisson_kwargs)
            f = 2WaterLily.pressure_force(sim)/R^(Dim-1)
            push!(forces,[sim_time(sim),f[1]])
        end
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        push!(p,slice(sim.flow.p[Rslice]))
        push!(σ,slice(sim.flow.σ[Rslice]))
        plot_logger(fname*".log"); savefig(fname*".png")
    end
    jldsave("disk_$(Dim)D_$(N)D_$(use_biotsavart)_"*ifelse(fmm,"FMM","TREE")*"_$dist.jld2"; f=forces, σ=σ, p=p)
end
let
    # make the figures
    f_f = plot(dpi=600,xlabel="Convective time",ylabel="2F/ρU²RⁿCa")
    f_t = plot(dpi=600,xlabel="Convective time",ylabel="2F/ρU²R²Ca")
    for (Dim,use_biotsavart,dist,fmm) ∈ params
        @show Dim,use_biotsavart,dist,fmm
        BCs = use_biotsavart ? "biot" : "reflect"
        jldopen("disk_$(Dim)D_$(N)D_$(use_biotsavart)_"*ifelse(fmm,"FMM","TREE")*"_$dist.jld2") do file
            df = mapreduce(permutedims, vcat, file["f"])
            t = df[:,1]; idx = t .< 3
            t = t[idx]; f = df[idx,2]
            Ca=Dim==2 ? π : 8/3
            ls = ifelse(Dim==3,:solid,:dashdot)
            c = ifelse(use_biotsavart,colormap("Blues",8),colormap("Reds",8))[dist+4]
            @show -f[1]/Ca
            ((Dim==3 && dist==2 && fmm) || Dim==2) && plot!(f_f,t,-f./Ca,label="$(Dim)D "*(use_biotsavart ? "Biot-Savart (FMℓM S=$dist)" : "Reflection");lw=2,ls,c)
            if (Dim==3 && use_biotsavart) # method comparison
                ls,c = ifelse(fmm,(:solid,c),(:dashdot,colormap("Greens",8)[dist+4]));
                plot!(f_t,t,-f./Ca,label=ifelse(fmm,"FMℓM","Tree")*" (S=$(dist))";lw=2,ls,c)
            end
            ((Dim==3 && dist==2 && fmm) || Dim==2) && for t ∈ 1:6
                p = file["p"][t]; σ = file["σ"][t]
                flood(p,clims=(-2,2),cfill=:viridis)
                savefig("Disk_$(Dim)D_"*BCs*"_press_$(t).png")
                flood(σ,clims=(-20,20))
                savefig("Disk_$(Dim)D_"*BCs*"_omega_$(t).png")
                flood(σ,clims=(-20,20))
                contour!(clamp.(p',-2,2),levels=range(-2,2,length=10),color=:black,
                        linewidth=0.5,legend=false)
                savefig("Disk_$(Dim)D_"*BCs*"_press_omega_$(t).png")
            end
        end
    end
    xlims!(f_t,0,3); ylims!(f_t,0,4); savefig(f_t,joinpath(figdir,"Disk_force_comparison_methods.png"))
    xlims!(f_f,0,3); ylims!(f_f,0,10); savefig(f_f,joinpath(figdir,"Disk_force_comparison.png"))
end