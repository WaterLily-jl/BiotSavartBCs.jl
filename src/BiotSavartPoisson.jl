"""
    BiotSavartPoisson <: WaterLily.AbstractPoisson

Custom Poisson type for Biot-Savart boundary conditions. Wraps a `MultiLevelPoisson`
(which holds all pressure system state) and layers Biot-Savart state on top.

Fields:
- `ml`      : wrapped standard pressure solver
- `ω`       : multi-level vorticity (top level aliases `flow.f`)
- `tar`     : domain boundary target index arrays per multigrid level
- `ftar`    : flattened target list for kernel dispatch
- `p`       : pressure solution accumulator
- `fmm`     : use Fast Multi-level Method (`true`) or tree-sum (`false`)
- `perdir`  : periodic directions (forwarded from `flow.perdir`); those faces are excluded from Biot-Savart targets
- `nimages` : fixed discrete image-core half-width per side in `perdir`; the |n|>nimages remainder is summed in closed form (`image_tail`), so accuracy does not depend on it
"""
struct BiotSavartPoisson{T,S,V} <: AbstractPoisson{T,S,V}
    ml      :: MultiLevelPoisson{T,S,V}
    ω       :: NTuple
    tar     :: NTuple
    ftar    :: AbstractVector
    p       :: AbstractArray
    fmm     :: Bool
    perdir  :: NTuple
    nimages :: Int
    function BiotSavartPoisson(flow; nonbiotfaces=(), fmm=true, mem=Array, nimages=2)
        ml = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ; perdir=flow.perdir)
        ω  = MLArray(flow.f)   # top level aliases flow.f — no copy
        tar  = mem.(collect_targets(ω, nonbiotfaces, flow.perdir))
        ftar = flatten_targets(tar)
        p   = copy(flow.p)
        new{eltype(flow.p),typeof(flow.p),typeof(flow.μ₀)}(ml,ω,tar,ftar,p,fmm,flow.perdir,nimages)
    end
end
WaterLily.update!(b::BiotSavartPoisson) = WaterLily.update!(b.ml)

"""
    mom_project!(a::AbstractFlow, b::BiotSavartPoisson, w, t; U=BCTuple(a.uBC,t,N), tol=1e-4, itmx=32)

Custom project method for Biot-Savart BCs. Solves for pressure with a multigrid V-cycle, applying biot_BC! to update the boundary velocity and residual at each iteration.
Note: a.p is used as the incremental pressure solution for each V-cycle, while b.p accumulates the total pressure solution.
`U` defaults to the domain BC tuple, but can be overridden (e.g. for a moving reference frame) by callers that bypass `WaterLily.mom_step!`.
"""
function WaterLily.mom_project!(a::AbstractFlow{N}, b::BiotSavartPoisson, w, t; U=BCTuple(a.uBC,t,N), tol=1e-4, itmx=32) where N
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    b.p .= 0; project_update!(a,b)                              # Project out initial μ₀∇p
    periodicBC!(a.u,b.perdir)   # sync periodic u-ghosts so fill_ω!'s curl at the perdir boundary layers (buff=1) is periodic
    fill_ω!(b.ω,a.u,b.perdir); biotBC!(a.u,U,b.ω,b.tar,b.ftar,b.perdir,b.nimages;fmm=b.fmm)

    # Set residual
    top = b.ml.levels[1]; top.r .= 0
    @inside top.r[I] = ifelse(top.iD[I]==0,0,WaterLily.div(I,a.u))
    fix_resid!(top.r,a.u,b.tar[1]) # only fix on the boundaries

    nᵖ,nᵇ,r₂ = 0,0,L₂(top)
    @log ", $nᵖ, $(WaterLily.L∞(top)), $r₂, $nᵇ\n"
    while nᵖ<itmx
        # V-cycle with fixed BCs until the residual drops >10x
        rtol = max(tol,0.1r₂)
        while nᵖ<itmx
            WaterLily.Vcycle!(b.ml); WaterLily.smooth!(top)
            r₂ = L₂(top); nᵖ+=1
            r₂<rtol && break
        end
        # Update the BCs with Biot-Savart (which requires updating u,p,ω) and repeat until convergence
        project_update!(a,b) # Update u,p
        periodicBC!(a.u,b.perdir)   # sync periodic ghosts so ω=∇×u is periodic before Biot-Savart image sum
        fill_ω!(b.ω,a.u,b.perdir); biotBC_r!(top.r,a.u,U,b.ω,b.tar,b.ftar,b.perdir,b.nimages;fmm=b.fmm)
        r₂ = L₂(top); nᵇ+=1
        @log ", $nᵖ, $(WaterLily.L∞(top)), $r₂, $nᵇ\n"
        r₂<tol && break
    end
    push!(b.ml.n,nᵖ)
    pflowBC!(a.u,b.perdir)     # Update ghost BCs (domain is already correct)
    periodicBC!(a.u,b.perdir)  # Sync periodic ghost cells (pflowBC! skips perdir)
    a.p .= b.p/dt     # rescale pressure solution and copy to Flow
end
BCTuple(f::Function,t::T,N) where T = ntuple(i->f(i,zero(SVector{N,T}),t),N)
BCTuple(f::Tuple,t,N) = f

# Apply u-=μ₀∇p & accumulate p
function project_update!(a::AbstractFlow, b::BiotSavartPoisson)
    WaterLily.perBC!(a.p,b.perdir) # refresh periodic ghost of p before gradient computation
    @loop a.u[Ii] -= a.μ₀[Ii]*∂(last(Ii),front(Ii),a.p) over Ii ∈ inside_u(a.u)
    b.p .+= a.p; fill!(a.p,0) # accumulate total pressure solution, reset increment
end
