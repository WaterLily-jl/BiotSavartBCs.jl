module BiotSavartBCs

using WaterLily

include("geom.jl")
include("multilevel.jl")

include("fmm.jl")
include("tree.jl")
include("velocity.jl")

include("BiotSavartPoisson.jl")
export BiotSavartPoisson

"""
   BiotSimulation((WaterLily.Simulation inputs)...; fmm=true, nonbiotfaces=(), symmetry=(), mem=Array)

Constructor for a WaterLily.Simulation that uses Biot-Savart boundary conditions.
Returns a plain `WaterLily.Simulation` with a `BiotSavartPoisson` solver injected via `pois_ctor`.

- `fmm`: Use the Fast Multi-level Method (`true`, default) or tree-sum (`false`).
- `nonbiotfaces`: tuple of face indices to exclude from Biot-Savart BCs (e.g. `(-2,)` for the negative-y face).
- `symmetry`: tuple of face indices that are symmetry planes (e.g. `(-2,)`). Like `nonbiotfaces` these faces don't get Biot-Savart BCs, but the images of the vorticity across them are included.
- `mem`: memory backend (`Array`, `CuArray`, etc.).

See: `Using Biot-Savart boundary conditions for unbounded external flow on Eulerian meshes,
https://arxiv.org/abs/2404.09034` and `WaterLily.Simulation`.
"""
function BiotSimulation(args...; nonbiotfaces=(), fmm=true, symmetry=(), mem=Array, kwargs...)
    Simulation(args...; mem,
        pois_ctor=flow->BiotSavartPoisson(flow; nonbiotfaces, fmm, mem, symmetry),
        kwargs...)
end
export BiotSimulation

end # module