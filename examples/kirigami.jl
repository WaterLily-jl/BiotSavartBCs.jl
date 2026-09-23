using WaterLily,BiotSavartBCs,CUDA,StaticArrays

# linear acceleration profile
linear(t)=min(t,one(t))
WaterLily.CFL(a::Flow) = WaterLily.CFL(a;Δt_max=1) # good idea when accelerating from rest

function kirigami(N;H=0,rings=16,U=1,Re=1e4,mem=Array,T=Float32,Ux=linear,R=T(2N/3),ϵ=T(1/2),half_thk=ϵ+1/T(√2),fall=false)
    δR = R/rings; δH = R*H/rings^2; x₀ = max(R*(1-H)/2,δR+half_thk-min(0,R*H))
    @inline mapped(f) = AutoBody(f,(x,t)->x-SA[x₀,0,0])
    @inline ring(R₀,R₁,x₀,x₁,ϕ) = mapped() do (x,y,z),t
        r,θ = hypot(y,z),atan(z,y)
        δx = x₀+tanh(π*r/δR)*(x₁-x₀)*(1+cos(4θ+ϕ))/2
        hypot(x-δx,r-clamp(r,R₀+half_thk,R₁-half_thk))-half_thk
    end
    body = sum(i -> ring(δR*(i-1), δR*i, δH*(i-1)^2, δH*i^2, π*(i%2)), 1:rings)
    H == 0 && (body = ring(0,R,0,0,0))
    Ut = fall ? (0,0,0) : (i,x,t)->(i==1 ? U*Ux(U*t/2R) : zero(t)) # velocity BC
    BiotSimulation((3N,N,N),Ut,R;U,ν=U*2R/Re,body,mem,T,ϵ,nonbiotfaces=(-2,-3))
end

import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₂,sgn₂ = image(T,size(ω),-2)  # image target and sign in y
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    T₂₃,_   = image(T₃,size(ω),-2) # image of image!
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)+
     sgn₂*(interaction(ω,T₂,args...)+sgn₃*interaction(ω,T₂₃,args...))
end
using TypedTables
drag!(sim,times,R=sim.L,x₀=SA[R,0,0];remeasure=false) = map(times) do t
    @show t; flush(stdout)
    sim_step!(sim,t;remeasure)
    Cd,Cl = -8WaterLily.total_force(sim)[1:2]/R^2
    Cm = 8WaterLily.pressure_moment(x₀,sim)[3]/R^3
    (;t,Cd,Cl,Cm)
end |> Table

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
using WriteVTK
import WaterLily: @loop,ω,λ₂
vtk_ω(a::AbstractSimulation) = (@loop a.flow.f[I,:] .= ω(I,a.flow.u) over I in inside(a.flow.p); a.flow.f |> Array)
vtk_d(a::AbstractSimulation) = (measure_sdf!(a.flow.σ,a.body,WaterLily.time(a)); a.flow.σ |> Array)
vtk_λ₂(a::AbstractSimulation) = (@inside a.flow.σ[I] = λ₂(I,a.flow.u); a.flow.σ |> Array)

# Convergence sweep
using JLD2
H = 1; times = 0.05:0.05:3
for N in (3*2^4, 2^6, 3*2^5, 2^7, 3*2^6) # 3*2^7 10GiB # do 2^8 below
    @show N; flush(stdout)
    sim = kirigami(N;H,rings=8,mem=CUDA.CuArray)
    data = drag!(sim,times)
    save_object("kirigami_N$(N)_H$(H)_rings8_hist.jld2",data)
    writer = vtkWriter("kirigami_N$(N)_H$(H)_rings8"; attrib=Dict("ω"=>vtk_ω,"λ₂"=>vtk_λ₂,"d"=>vtk_d))
    save!(writer,sim); close(writer)
end

# Rings sweep
N = 2^8; times = 0.05:0.05:3
for H in (0.5,1,2,4), rings in 4:4:20
    @show rings; flush(stdout)
    sim = kirigami(N;H,rings,mem=CUDA.CuArray)
    data = drag!(sim,times)
    save_object("kirigami_N$(N)_H$(H)_rings$(rings)_hist.jld2",data)
    writer = vtkWriter("kirigami_N$(N)_H$(H)_rings$(rings)"; attrib=Dict("ω"=>vtk_ω,"λ₂"=>vtk_λ₂,"d"=>vtk_d))
    save!(writer,sim); close(writer)
end

# H sweep
Hs = 0.5 .^ (-2:2)
Hs = [-Hs; 0; reverse(Hs)] # include negative H for checking symmetry
for H ∈ Hs
    @show H; flush(stdout)
    sim = kirigami(N;H,mem=CUDA.CuArray)
    data = drag!(sim,times)
    save_object("kirigami_N$(N)_H$(H)_hist.jld2",data)
    writer = vtkWriter("kirigami_N$(N)_H$(H)"; attrib=Dict("ω"=>vtk_ω,"λ₂"=>vtk_λ₂,"d"=>vtk_d))
    save!(writer,sim); close(writer)
end
