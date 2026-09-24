using BiotSavartBCs
using Test
using WaterLily

using BiotSavartBCs: @vecloop,inside_u,restrict!,project!,down,front,step

using StaticArrays
@testset "util.jl" begin
    a = zeros(Int,(4,4,6,3))
    @vecloop a[I] += 1 over I in inside_u(a,buff=2)
    @test sum(a) == 0
    @vecloop a[I] += 1 over I in inside_u(a)
    @test sum(a) == length(inside_u(a)) == 2*2*4*3

    a = zeros(Int,(10,10,18,3))
    ml=MLArray(a)
    @test length(ml)==3
    @vecloop a[I] += 1 over I in inside_u(a)
    @test sum(first(ml)) == length(inside_u(a))
    restrict!(ml)
    @test sum(last(ml)) == length(inside_u(a))

    tar = collect_targets(ml)
    @test length(tar[1]) == 4length(tar[2]) == 16length(tar[3])
    tar2 = collect_targets(ml,(-1,-2))
    @test first(tar[3]) ∉ tar2[3]
    @test length(tar2[3])+2*2*4 == length(tar[3])

    Ti = last(tar[2])
    T,i = front(Ti),last(Ti)
    @test CartesianIndex(down(T),i)==last(tar[3])

    @vecloop ml[3][I] += 16 over I in tar[3]
    project!(ml,tar)
    @test ml[2][Ti] == 4

    @test length(flatten_targets(tar)) == sum(length,tar)
    @test flatten_targets(tar)[sum(length,tar[1:2])] == (2,Ti)

    a = zeros(Int,(34,34,2))
    ml=MLArray(a)
    @test length(ml)==3 # much bigger dis in 2D
    @vecloop a[I] += 1 over I in inside_u(a)
    @test sum(first(ml)) == length(inside_u(a))
    restrict!(ml)
    @test sum(last(ml)) == length(inside_u(a))

    tar = collect_targets(ml)
    @test length(tar[1]) == 2length(tar[2]) == 4length(tar[3])
    Ti = last(tar[2])
    T,i = front(Ti),last(Ti)
    @test CartesianIndex(down(T),i)==last(tar[3])

    @vecloop ml[3][I] += 4 over I in tar[3]
    project!(ml,tar)
    @test ml[2][Ti] == 2
end

using SpecialFunctions,ForwardDiff
function lamb_dipole(N;D=3N/4,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    return function uλ(i,xy)
        x,y = xy .- (N-2)/2
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
end
function hill_vortex(N;D=3N/4)
    return function uλ(i,xyz)
        q = xyz .- (N-2)/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
        v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*cos(θ)
        v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*sin(θ)
        i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
        i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
        cos(θ)*v_r-sin(θ)*v_θ
    end
end

using BiotSavartBCs: slice
@testset "velocity.jl" begin
    # Hill ring vortex in 3D
    N = 2+2^5
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u)
    ω = zeros(Float32,N,N,N,3)

    fill_ω!(ω,u) # Ideally, ω₃=0 & |ωᵩ|N/U≤20, but ω is discontinuous...
    @test all(-0.25 .< extrema(ω[:,:,:,3]) .*N .< 0.25) # roughly 0
    @test 18 < maximum(ω)*N < 20 # roughly |20|
    @test abs(sum(ω)) < 1e-4 # zero total circulation

    N = 2+3*2^3; U=(0,0,1)
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u); u₀ = copy(u)
    ω = MLArray(zeros(Float32,N,N,N,3)); tar = collect_targets(ω); ftar = flatten_targets(tar);
    fill_ω!(ω,u)
    BC!(u,U) # mess up BCs

    # Check domain uₙ using FMM-version of Biot-Savart BCs
    biotBC!(u,U,ω,tar,ftar)
    tol = (0.0222,0.0222,0.05) # Hill vortex has largest uₙ on z faces
    for i in 1:3, s in (2,N)
        mx = maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,s))
        # @show i,s,mx
        @test mx < tol[i]
    end

    # Tangential ghosts are great
    pflowBC!(u) # fix ghosts
    @test maximum(abs,(u.-u₀)[3:end-1,2:end-1,1,1])<0.02
    @test maximum(abs,(u.-u₀)[3:end-1,2:end-1,end,1])<0.02
    @test maximum(abs,(u.-u₀)[2:end-1,3:end-1,1,2])<0.02
    @test maximum(abs,(u.-u₀)[2:end-1,3:end-1,end,2])<0.02
    @test maximum(abs,(u.-u₀)[1,2:end-1,3:end-1,3])<0.023
    @test maximum(abs,(u.-u₀)[end,2:end-1,3:end-1,3])<0.023

    # Normal ghost has lower accuracy (but it's the least important)
    for i in 1:3
        @test maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,1)) < 0.06
    end

    # Check domain uₙ using tree-version of Biot-Savart BCs
    BC!(u,U) # mess up BCs
    biotBC!(u,U,ω,tar,ftar,fmm=false) # tree
    tol = (0.004,0.004,0.02) # No target interpolation error!
    for i in 1:3, s in (2,N)
        mx = maximum(I->abs(u[I]-u₀[I]),slice(size(u),i,s))
        # @show i,s,mx
        @test mx < tol[i]
    end

    pow = 5; N = 2+2^pow; U = (1,0)
    u = Array{Float32}(undef,(N,N,2)); apply!(lamb_dipole(N),u); u₀ = copy(u)
    ω = MLArray(zeros(Float32,N,N,2)); tar = collect_targets(ω); ftar = flatten_targets(tar);

    fill_ω!(ω,u)
    @test all(ω[1][:,:,2].==0) # we don't use the second component
    @test all(ω[1][[2,N-1],:,1].==0) # no vorticity outside the bubble
    @test all(@. abs(sum(ω))<12e-5) # zero-sum at every level

    BC!(u,U) # mess up boundaries
    biotBC!(u,U,ω,tar,ftar;fmm=true) # fix domain velocities
    @test maximum(abs,(u.-u₀)[2:end,2:end-1,1])<0.028
    @test maximum(abs,(u.-u₀)[2:end-1,2:end,2])<0.025

    BC!(u,U) # mess up boundaries
    biotBC!(u,U,ω,tar,ftar;fmm=false) # fix domain velocities
    @test maximum(abs,(u.-u₀)[2:end,2:end-1,1])<0.0063 # No target interpolation error
    @test maximum(abs,(u.-u₀)[2:end-1,2:end,2])<0.003
    pflowBC!(u) # fix ghosts
    @test maximum(abs,(u.-u₀)[3:end-1,1,1])<0.0044 # tangential
    @test maximum(abs,(u.-u₀)[1,3:end-1,2])<0.003 # tangential
    @test maximum(abs,(u.-u₀)[1,3:end-2,1])<0.0064 # normal
    @test maximum(abs,(u.-u₀)[3:end-2,1,2])<0.003 # normal
end

@testset "BiotSavartPoisson.jl" begin
    circ(D;fmm,U=1,m=2D) = BiotSimulation((m,m), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4,fmm)
    for fmm in (true,false)
        sim = circ(256;fmm)
        sim_step!(sim;remeasure=false)
        u_max = maximum(sim.flow.u[:,:,1])
        v_max = maximum(sim.flow.u[:,:,2])
        u_inf = minimum(sim.flow.u[1,:,1])
        @show fmm,u_max,v_max,u_inf
        @test abs(u_max-2)<0.02 # circle u_max = 2
        @test abs(v_max-1)<0.02 # circle v_max = 1
        @test abs(u_inf-0.75)<0.02 # upstream slow down
        @time sim_step!(sim;remeasure=false)
        @show sim.pois.ml.n
        @test !isempty(sim.pois.ml.n) # iteration count recorded after step
    end

    sphere(D;fmm,m=3D÷2) = BiotSimulation((m,m,m), (1,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=D/1e4,fmm)
    for fmm in (true,false)
        sim = sphere(128;fmm)
        sim_step!(sim;remeasure=false)
        u_max = maximum(sim.flow.u[:,:,:,1])
        v_max = maximum(sim.flow.u[:,:,:,2:3])
        u_inf = minimum(sim.flow.u[2,:,:,1])
        @show fmm,u_max,v_max,u_inf
        @test abs(u_max-1.5)<0.012    # u_max = 3/2
        @test abs(v_max-0.75)<0.035   # v,w_max = 3/4
        @test abs(u_inf-19/27)<0.033  # upstream slow down
        @time sim_step!(sim;remeasure=false)
        @show sim.pois.ml.n
    end

    # Spanwise-periodic cylinder: stagnation velocity should match 2D result (fmm=true only; tree has no periodic support)
    cyl_span(D;m=2D,Lz=D÷2) = BiotSimulation((m,m,Lz),(1,0,0),D;
                                              body=AutoBody((x,t)->√sum(abs2,(x.-m/2)[1:2])-D/2),
                                              ν=D/1e4,fmm=true,perdir=(3,))
    let sim = cyl_span(128)
        sim_step!(sim;remeasure=false)
        u_max = maximum(sim.flow.u[:,:,:,1])
        v_max = maximum(sim.flow.u[:,:,:,2])
        u_inf = minimum(sim.flow.u[1,:,:,1])
        @show u_max,v_max,u_inf
        @test abs(u_max-2)<0.15 # 2D cylinder stagnation: u_max = 2 (loose tol: 3D first-step accuracy)
        @test abs(v_max-1)<0.05 # circle v_max = 1
        @test abs(u_inf-0.75)<0.10 # upstream slow down
        # z-reflection symmetry: z-uniform body/inflow with w(0)=0 -> spanwise w must stay ~0.
        # A z-varying periodic FMM BC (the bug pinteraction fixes) injects w via the pressure solve.
        @test maximum(abs,sim.flow.u[:,:,:,3]) < 1e-3  # spanwise w ~ 0 (≈3e-5 here)
        z_uniform = maximum(k->maximum(abs,sim.flow.u[:,:,k,1].-sim.flow.u[:,:,2,1]), 3:size(sim.flow.u,3)-2)
        @test z_uniform < 1e-2                          # streamwise u is z-uniform
        @show sim.pois.ml.n
        @test !isempty(sim.pois.ml.n)
    end
end

@testset "periodic BCs" begin
    # collect_targets excludes periodic faces
    ml = MLArray(zeros(Float32,10,10,18,3))
    tar   = collect_targets(ml)
    tar_p = collect_targets(ml,(),(3,))
    @test !any(T->last(T)==3, tar_p[1])                         # no z-direction targets
    @test length(tar_p[1]) == length(tar[1]) - 2*(10-2)*(10-2) # correct count

    # periodicBC! sets ghost cells to match the opposing interior face (perBC! for vector fields)
    using BiotSavartBCs: periodicBC!
    N=10; u = randn(Float32,N,N,N,3)
    periodicBC!(u,(3,))
    @test u[:,:,1,:] == u[:,:,N-1,:]   # lower ghost = upper interior
    @test u[:,:,N,:] == u[:,:,2,:]     # upper ghost = lower interior

    # pflowBC! leaves periodic ghost cells untouched
    u2 = randn(Float32,N,N,N,3)
    z_lo,z_hi = copy(u2[:,:,1,:]),copy(u2[:,:,N,:])
    pflowBC!(u2,(3,))
    @test u2[:,:,1,:] == z_lo
    @test u2[:,:,N,:] == z_hi

    # z-uniform ω: periodic images reduce spurious z-variation at x-face boundaries
    pow=4; N2=2+2^pow; Lz=8
    u2 = Array{Float32}(undef,(N2,N2,2)); apply!(lamb_dipole(N2),u2)
    u3 = zeros(Float32,N2,N2,Lz,3)
    for k in 1:Lz; u3[:,:,k,1].=u2[:,:,1]; u3[:,:,k,2].=u2[:,:,2]; end
    ω3 = MLArray(zeros(Float32,N2,N2,Lz,3)); fill_ω!(ω3,u3,(3,)); U3 = (1,0,0)

    # 3D spanwise-periodic Lamb dipole: for a z-uniform flow, biotBC! with perdir=(3,)
    # must produce z-UNIFORM x,y face velocities (the defining property of a periodic BC).
    tar3p = collect_targets(ω3,(),(3,)); ftar3p = flatten_targets(tar3p)
    for k in 1:Lz; u3[:,:,k,1].=u2[:,:,1]; u3[:,:,k,2].=u2[:,:,2]; end
    BC!(u3,U3); biotBC!(u3,U3,ω3,tar3p,ftar3p,(3,),4;fmm=true)
    z_var_x = maximum(z->abs(u3[2,N2÷2,z,1]-u3[2,N2÷2,Lz÷2,1]), 3:Lz-2)
    z_var_y = maximum(z->abs(u3[N2÷2,2,z,2]-u3[N2÷2,2,Lz÷2,2]), 3:Lz-2)
    @test z_var_x < 1e-3
    @test z_var_y < 1e-3

    # After mom_project!, periodic ghost cells must be fresh (perdir=(3,) — periodic BCs are 3D-only).
    # Without periodicBC! at the end of mom_project!, pflowBC! skips perdir and
    # ghosts remain stale, causing spurious div(u) in the next step.
    sim3d = BiotSimulation((16,16,8),(1,0,0),8; perdir=(3,), ν=0.01)
    sim_step!(sim3d; remeasure=false)
    u = sim3d.flow.u
    @test u[:,:,1,:] == u[:,:,end-1,:]  # lower ghost = upper interior
    @test u[:,:,end,:] == u[:,:,2,:]    # upper ghost = lower interior

    # fill_ω! must use buff=1 along perdir: the periodic Biot-Savart source integrates the
    # full period (indices 2:N-1), so the first interior vorticity layers (z=2, Nz-1) must be
    # populated
    Nx=Ny=10; P=6; φ0=0.9f0
    fx = Float32[sin(2π*(i-1)/Nx) for i in 1:Nx, j in 1:Ny]
    fy = Float32[cos(2π*(j-1)/Ny) for i in 1:Nx, j in 1:Ny]
    φz(k) = 2π*mod(k-2,P)/P + φ0   # period P; bit-identical at matching cells of either domain
    build_u(Nz) = (v=zeros(Float32,Nx,Ny,Nz,3);
        for k in 1:Nz; φ=φz(k)
            v[:,:,k,1] .= fx.*sin(φ); v[:,:,k,2] .= fy.*cos(φ); v[:,:,k,3] .= (fx.+fy).*sin(φ)
        end; v)
    Nz_s = P+2                                                      # single period
    u_s = build_u(Nz_s); ω_s = MLArray(zeros(Float32,Nx,Ny,Nz_s,3)); fill_ω!(ω_s,u_s,(3,))
    Nz_t = 4P+2                                                     # four periods
    u_t = build_u(Nz_t); ω_t = MLArray(zeros(Float32,Nx,Ny,Nz_t,3)); fill_ω!(ω_t,u_t) # buff=2 ok deep inside
    xy = (3:Nx-2, 3:Ny-2)                                          # x,y interior (skip x,y buff zeros)
    for k in 2:Nz_s-1                                              # shift one period into tall interior
        @test ω_s[1][xy...,k,:] == ω_t[1][xy...,k+P,:]
    end
    @test maximum(abs, ω_s[1][xy...,2,:]) > 0.01                  # boundary layer genuinely nonzero
    @test maximum(abs, ω_s[1][xy...,Nz_s-1,:]) > 0.01            # (so the match above isn't 0==0)
    ω_b2 = MLArray(zeros(Float32,Nx,Ny,Nz_s,3)); fill_ω!(ω_b2,u_s)  # no perdir -> buff=2 in z (old path)
    @test all(iszero, ω_b2[1][:,:,2,:]) && all(iszero, ω_b2[1][:,:,Nz_s-1,:])  # what the fix guards against

    # Spanwise z-reflection symmetry over several steps, check that z-velocity doesn't grow spuriously, test pinteraction
    let sim = BiotSimulation((48,48,8),(1,0,0),24; body=AutoBody((x,t)->√sum(abs2,(x.-24)[1:2])-12),
                             ν=24/1e3, fmm=true, perdir=(3,))
        for _ in 1:6; sim_step!(sim;remeasure=false); end
        @test maximum(abs,sim.flow.u[:,:,:,3]) < 1e-3   # spanwise w stays ∼1e-4
    end
end