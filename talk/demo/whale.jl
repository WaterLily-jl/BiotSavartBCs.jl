using WaterLily, ParametricBodies, BiotSavartBCs, WaterLilyNarrowBand, LilyPad, StaticArrays, CUDA, GLMakie
Makie.inline!(false)
CUDA.allowscalar(false)

function whale(Λ=25; L=32, U=1, Re=1e4, T=Float32, mem=CUDA.functional() ? CuArray : Array, Δt=1.5)
    0 ≤ Λ ≤ 35 || throw(ArgumentError("this planform is only valid for sweep angles of 0–35°, got Λ=$Λ"))
    s = 190tand(Λ)/8
    pnts = SA{T}[-20 40 190   200   190   170   100 -20 -30 -20
                 0  0  8s 8s+40 5s+70 5s+50 5s+70 100  80 0]
    function map(x, t)
        θ, h = π/6*sin(U*t/L), 1.25+0.5cos(U*t/L)
        SA[cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]*100*(x/L-SA[0.3, 0, h])
    end
    dims = (4L, 5L÷2, 5L÷2)
    plate = PlanarBody(BSplineCurve(reverse(pnts), degree=3), (0, 1); map, T, mem, buffer=64)
    body = NarrowBand(plate, dims; mem, interior=false, k=5)
    LilyBiotSim(dims, (U, 0, 0), L; U, ν=U*L/Re, body, T, mem, symmetry=(-2,), Δt)
end

show_whale(Λ=25; L=32, duration=2π, kw...) = viz!(whale(Λ; L); duration, step=0.1, sym=(0, 1, 0),
    body2mesh=false, algorithm=:mip, colormap=:dense, colorrange=(1, 30)./L, azimuth=-0.6π, elevation=π/6,
    hidedecorations=true, kw...)

println("Whale demo ready. Run show_whale() to warm up, then show_whale(Λ) with Λ = 0–35°.")
