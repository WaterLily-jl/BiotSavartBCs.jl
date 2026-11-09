# Live whale-fluke demo for slide 5. In a REPL beside the deck:
#   julia --project=talk/demo -i talk/demo/whale.jl
#   show_whale()            # a new sweep angle Λ (degrees) from the room: one stroke, ~17 s on the A2000
#   show_whale(40; L=32)    # finer, ~40 s per stroke
# Run show_whale() once before the talk: the first call compiles for a minute or two.
# From CPC 2024 examples/Whale.jl: a flapping half-planform, mirrored about its root.
using WaterLily, ParametricBodies, StaticArrays, CUDA, GLMakie, Meshing
Makie.inline!(false)
CUDA.allowscalar(false)

# Planform (span × chord, units of 1/200 span) whose tip sits Λ degrees aft of the root
function whale(Λ=25; L=32, U=1, Re=1e4, T=Float32, mem=CUDA.functional() ? CuArray : Array)
    s = 190tand(Λ)/8
    pnts = SA{T}[0 40 190   200   190   170   100   0 -10 0
                 0  0  8s 8s+40 5s+70 5s+50 5s+70 100  80 0]
    planform = BSplineCurve(reverse(pnts), degree=3)
    function map(x, t)
        θ, h = π/6*sin(U*t/L), 1+0.5cos(U*t/L)
        Ry = SA[cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
        Ry*100*(x/L-SA[0.75, 0, h])
    end
    body = PlanarBody(planform, (0, 1); map, T, mem)   # mem puts its lookup table on the GPU
    Simulation((5L, 3L, 2L), (U, 0, 0), L; U, ν=U*L/Re, body, T, mem)
end

# viz! plots |ω| in grid units; ωL/U from 1 to 30 matches the CPC figure
show_whale(Λ=25; L=24, duration=2π, kw...) = viz!(whale(Λ; L); duration, step=0.1, sym=(0, 1, 0),   # one stroke
    body2mesh=false, algorithm=:mip, colormap=:dense, colorrange=(1, 30)./L, azimuth=-0.6π, elevation=π/6,
    hidedecorations=true, kw...)

println("Whale demo ready on ", CUDA.functional() ? "the GPU" : "the CPU", ". Run show_whale() to warm up, then show_whale(Λ) with a sweep angle in degrees.")
