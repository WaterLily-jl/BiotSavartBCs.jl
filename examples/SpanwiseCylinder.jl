using WaterLily, StaticArrays, BiotSavartBCs, WriteVTK

# simulation constructor
function spanwise_cylinder(;dims=(4,2,1), D=32, Re=1_000, U=1, T=Float32, mem=Array)
    body = AutoBody((x,t) -> √sum(abs2,SA[x[1]-D,x[2]-D]) - D/3.f0)
    length(dims)==3 && return BiotSimulation(dims.*D, (U,0,0), D; ν=U*D/Re, body, T, mem, perdir=(3,), nimages=1)
    BiotSimulation(dims.*D, (U,0), D; ν=U*D/Re, body, T, mem)
end

# function for vtk writing
vtk_sdf(a::Simulation)      = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array)
vtk_velocity(a::Simulation) = a.flow.u  |> Array
vtk_mu0(a::Simulation)      = a.flow.μ₀ |> Array
vtk_pressure(a::Simulation) = a.flow.p  |> Array
# write vtk fields
attrib = Dict("d"=>vtk_sdf, "u"=>vtk_velocity, "μ₀"=>vtk_mu0, "p"=>vtk_pressure)

# make 2D and 3D sim
using CUDA
sim2D = spanwise_cylinder(;dims=(4,2),   D=128, mem=CuArray)
sim3D = spanwise_cylinder(;dims=(4,2,3), D=128, mem=CuArray, Re=12_000)

# spread after a few steps
println("Running 2D sim for 100 steps...")
sim_step!(sim2D, 100.0; remeasure=false)
println("Done, spreading to 3D...")
WaterLily.spread!(sim3D, sim2D; dim=3, ϵ=0.0)

# writer for the 3D sim
writer = vtkWriter("SpanwiseCylinder"; attrib)

# run the 3D sim
println("Running 3D sim for 30 time units...")
for t in range(0, 30.0; step=0.05)
    sim_step!(sim3D, t; remeasure=false)
    sim_info(sim3D)
    t≥25.0 && save!(writer, sim3D)
end
close(writer)
