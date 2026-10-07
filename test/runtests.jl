using BiotSavartBCs
using Test
using WaterLily

using WaterLily: @loop
using BiotSavartBCs: inside_u,restrict!,project!,down,front,step
using BiotSavartBCs: MLArray,collect_targets,flatten_targets,fill_ω!,biotBC!,pflowBC!
@testset "util.jl" begin
    a = zeros(Int,(4,4,6,3))
    @loop a[I] += 1 over I in inside_u(a,buff=2)
    @test sum(a) == 0
    @loop a[I] += 1 over I in inside_u(a)
    @test sum(a) == length(inside_u(a)) == 2*2*4*3

    a = zeros(Int,(10,10,18,3))
    ml=MLArray(a)
    @test length(ml)==3
    @loop a[I] += 1 over I in inside_u(a)
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
    
    @loop ml[3][I] += 16 over I in tar[3]
    project!(ml,tar)
    @test ml[2][Ti] == 4

    @test length(flatten_targets(tar)) == sum(length,tar)
    @test flatten_targets(tar)[sum(length,tar[1:2])] == (2,Ti)

    a = zeros(Int,(34,34,2))
    ml=MLArray(a)
    @test length(ml)==3 # much bigger dis in 2D
    @loop a[I] += 1 over I in inside_u(a)
    @test sum(first(ml)) == length(inside_u(a))
    restrict!(ml)
    @test sum(last(ml)) == length(inside_u(a))

    tar = collect_targets(ml)
    @test length(tar[1]) == 2length(tar[2]) == 4length(tar[3])
    Ti = last(tar[2])
    T,i = front(Ti),last(Ti)
    @test CartesianIndex(down(T),i)==last(tar[3])
    
    @loop ml[3][I] += 4 over I in tar[3]
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
end
@testset "periodic BCs" begin
    # Slab test: a z-uniform Lamb dipole in a z-periodic domain must induce the 2D velocity
    N = 2+2^5; u₀ = Array{Float32}(undef,(N,N,2)); apply!(lamb_dipole(N),u₀)
    u = zeros(Float32,N,N,10,3); for k in 1:10; u[:,:,k,1:2] .= u₀; end
    ω = MLArray(zeros(Float32,N,N,10,3),(3,)); tar = collect_targets(ω,(3,-3)); ftar = flatten_targets(tar)
    fill_ω!(ω,u,(3,)); BC!(u,(1,0,0)); biotBC!(u,(1,0,0),ω,tar,ftar;perdir=(3,))
    Δu = u[:,:,2:end-1,1:2] .- reshape(u₀,N,N,1,2)
    @test maximum(abs,Δu[2:end,2:end-1,:,1])<0.04 && maximum(abs,Δu[2:end-1,2:end,:,2])<0.02 # as good as 2D

    # Spanwise-periodic cylinder: no spurious spanwise velocity
    sim = BiotSimulation((48,48,8),(1,0,0),24; body=AutoBody((x,t)->√sum(abs2,(x.-24)[1:2])-12),ν=24/1e3,perdir=(3,))
    for _ in 1:6; sim_step!(sim;remeasure=false); end
    @test maximum(abs,sim.flow.u[:,:,:,3]) < 1e-3
end

using BiotSavartBCs: induced,image,images
@testset "symmetry" begin
    # images mirror the target positions across the low (1.5) or high (N-½) face, and flip the normal component
    @test image(CartesianIndex(10,5,1),(10,12,2),-1) == (CartesianIndex(-6,5,1),-1) # normal: x=9.5 ↦ -6.5
    @test image(CartesianIndex(5,12,2),(10,12,2),-2) == (CartesianIndex(5,-8,2),-1) # normal: y=11.5 ↦ -8.5
    @test image(CartesianIndex(5,1,2),(10,12,2),2) == (CartesianIndex(5,22,2),-1)   # normal: y=1.5 ↦ 21.5
    @test image(CartesianIndex(5,1,6,2),(10,12,14,3),1) == (CartesianIndex(14,1,6,2),1)     # tangential: x=5 ↦ 14
    @test image(CartesianIndex(5,6,14,3),(10,12,14,3),-3) == (CartesianIndex(5,6,-10,3),-1) # normal: z=13.5 ↦ -10.5
    @test image(CartesianIndex(5,6,1,3),(10,12,14,3),3) == (CartesianIndex(5,6,26,3),-1)    # normal: z=1.5 ↦ 25.5

    # images matches a hand-written sum, including the image of the image
    @inline function sym_yz(ω,T,args...)
        T₂,sgn₂ = image(T,size(ω),-2); T₃,sgn₃ = image(T,size(ω),-3); T₂₃,_ = image(T₃,size(ω),-2)
        induced(ω,T,args...)+sgn₃*induced(ω,T₃,args...)+sgn₂*(induced(ω,T₂,args...)+sgn₃*induced(ω,T₂₃,args...))
    end
    N = 2+3*2^3
    u = Array{Float32}(undef,(N,N,N,3)); apply!(hill_vortex(N),u)
    ω = MLArray(zeros(Float32,N,N,N,3)); fill_ω!(ω,u)
    ftar = flatten_targets(collect_targets(ω,(-2,-3)))
    @test all(((l,T),)->images(ω[l],T,(-2,-3),l,length(ω)) ≈ sym_yz(ω[l],T,l,length(ω)), ftar)

    # Half-domain circle with a symmetry plane on y=0 matches the full-domain circle
    D = 64; m = 2D
    stats(sim) = (sim_step!(sim;remeasure=false); [maximum(sim.flow.u[:,:,1]),maximum(abs,sim.flow.u[:,:,2]),minimum(sim.flow.u[1,:,1])])
    half(;kw...) = BiotSimulation((m,m÷2), (1,0), D; body=AutoBody((x,t)->√((x[1]-m/2)^2+x[2]^2)-D/2),ν=D/1e4,kw...)
    sim = half(symmetry=(-2,))
    @test !any(Ti->Ti.I[2]==1 && last(Ti)==2, sim.pois.tar[1]) # symmetry plane has no Biot-Savart targets
    full = stats(BiotSimulation((m,m), (1,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=D/1e4))
    sym,nosym = stats(sim),stats(half(nonbiotfaces=(-2,)))
    @show full,sym,nosym
    @test all(abs.(sym .- full) .< 0.002)
    @test all(abs.(sym .- full) .< abs.(nosym .- full)/4) # images are essential

    @test_throws ArgumentError BiotSimulation((32,16), (1,0), 8; symmetry=(-2,), fmm=false)
    @test_throws ArgumentError BiotSimulation((32,16,8), (1,0,0), 8; symmetry=(-2,), perdir=(3,))
    @test_throws ArgumentError BiotSimulation((32,16), (1,0), 8; symmetry=(-2,2))
end
