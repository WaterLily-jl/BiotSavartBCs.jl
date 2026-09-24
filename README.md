# BiotSavartBCs

[![Build Status](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/weymouth/BiotSavartBCs.jl/actions/workflows/CI.yml?query=branch%3Amaster)

![disk](tex/fig/disk_high_re_7.png)

This repository defines an extension to the [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) flow solver, adding "external flow" boundary conditions based on the [Biot-Savart equation](https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law#Aerodynamics_applications). This equation is used to update the velocity *on the boundaries* of the simulation domain based on the vorticity *within* the domain. The resulting boundary conditions are an excellent model for external flow, allowing you to use very small domains around any immersed bodies. [See the paper for the detailed methodology and validation.](https://physics.paperswithcode.com/paper/using-biot-savart-to-shrink-eulerian-domains)

### WaterLily.jl Simulations with BiotSavartBCs.jl

Using these new boundary conditions within a `WaterLily` simulation is really straightforward; this requires changing only two (😱) lines of code. The first one is obviously

```julia
using WaterLily,BiotSavartBCs
```
The second line that you have to modify creates the Biot-Savart Simulation structure
```julia
sim = BiotSimulation((4L,2L,2L), Ut, L; body=AutoBody(sdf,map), ν=U*L/Re, T, mem=CUDA.CuArray)
```
You can update and plot the simulation structure exactly the same at with a standard WaterLily `Simulation`
```julia
sim_step!(sim, t_end; remeasure::Bool)
```
There are numerous examples in the `examples` folder of this repository that show how to use these new boundary conditions in practice.

### Method

This package takes a practical approach to avoid the two fundamental issues with applying the Biot-Savart equation to set the boundary conditions of a projection-based Navier-Stokes solver:
 1. A naive weighted sum over the $N_s$ vorticity sources at every cell for all $N_t$ targets at the domain cell faces would make the boundary condition update take $O(N_s N_t)$ operations, making it *orders of magnitude slower* than the rest of the solver. We accelerate the BC update by clustering the vorticity sources using a tree method (oct-tree in 3D and quad-tree in 2D). This reuses the pooling method in WaterLily's Multigrid pressure solver and reduces the cost to $O(\log(N_s) N_t)$. We can further accelerate the BC update by also clustering the target faces - making this an $O(N_t)$ Fast Multi*level* Method FMℓM - a variant of the classic [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method). Finally, we parallelize over all the targets using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) which works on the GPU or multi-threaded CPU.
 2. The pressure projection step depends sensitively on the boundaries conditions, but these *cannot be set* since the unknown pressure generates vorticity on immersed bodies. We solve this problem using a matrix partition method, similar to the approach used for partitioned Fluid-Structure-Interaction (FSI) methods. In practise we see the Multigrid pressure solver actually converges *faster* with `BiotSavartBcs` than with reflection BCs.

The resulting simulation update is very fast, especially with large 3D grids on the GPU - exactly where the ability to use a snug domain is the most important. See the paper for detailed methods, examples, and computational benchmarks.

### Mixed domain boundary conditions

You can turn off the Biot-Savart update to a domain face by passing the face index to the optional `nonbiotfaces` keyword argument (-3 is the negative z domain face, 2 is the positive y face, etc). In this case, the normal velocity at this face remains zero. Using this we can, for example, model a square plate abutting two slip-walls using:
```julia
function sym_square(N;Re=5e2,mem=Array,U=1,T=Float32,thk=2,L=T(N/2))
    body = AutoBody() do (x,y,z),t
        hypot(x-L,y-min(y,L-thk),z-min(z,L-thk))-thk
    end
    BiotSimulation((2N,N,N), (U,0,0),L;ν=U*2L/Re,body,mem,T,nonbiotfaces=(-2,-3))
end
sim_slip_walls = sym_square(96,mem=CuArray);
sim_step!(sim_slip_walls,2,remeasure=false) # or whatever
```

If we instead want to model a square plate (of twice the width) in an unbounded domain using a symmetric flow condition on the y & z planes, then we _also_ need to add the influence of the images of the vortices to the Biot-Savart boundaries. This is done by overwritting the `symmetry` function before running the simulation:
```julia
import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₂,sgn₂ = image(T,size(ω),-2)  # image target and sign in y
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    T₂₃,_   = image(T₃,size(ω),-2) # image of image!
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)+
     sgn₂*(interaction(ω,T₂,args...)+sgn₃*interaction(ω,T₂₃,args...))
end
sim_sym_walls = sym_square(96,mem=CuArray); # no difference!
sim_step!(sim_sym_walls,2,remeasure=false) # BiotBCs now see reflected domain
```

#### Periodic boundary conditions

If we want to model a flow that is periodic in one direction, we can use the `perdir` keyword argument to specify the periodic direction. For example, if we want to model a flow that is periodic in the z-direction, we can do:
```julia
sim_periodic = BiotSimulation((2N,N,N÷2),(U,0,0),L;
                              ν=U*2L/Re,body,mem,T,
                              perdir=(3,),nimages=4)
```
which creates a simulation with periodic boundary conditions in the z-direction. Internally, this switches off the Biot-Savart boundary update on the ±`dir` domain walls, similar to the symmetric case. Periodic conditions are enforced there in the classical way.

There is no need to _manually_ add the periodic contributions as we did for the symmetry case above; the `perdir` keyword argument automatically adds the periodic contributions to the Biot-Savart update. The `nimages` keyword argument specifies how many periodic images to use in the Biot-Savart update. The more images, the more accurate the update, but also the more expensive. We find that `nimages=4` is a good compromise between accuracy and cost.


> [!NOTE]
> Currently, periodic and symmetric conditions are __not__ compatible with each other, so you __cannot__ use both at the same time.

### Gallery

Here are a few renderings of the cool things you can do with [`WaterLily.jl`](https://github.com/weymouth/WaterLily.jl) and these new Biot-Savart BCs

#### Flow behind a square plate at Re=125,000
[![square1](https://img.youtube.com/vi/CNQqI5rRdug/0.jpg)](https://www.youtube.com/shorts/CNQqI5rRdug)

[![square2](https://img.youtube.com/vi/tbf06uhnAEQ/0.jpg)](https://www.youtube.com/shorts/tbf06uhnAEQ)

#### Flow behind a Doritos at Re=25,000
[![doritos](https://img.youtube.com/vi/spFlx2YW0pg/0.jpg)](https://www.youtube.com/shorts/spFlx2YW0pg)

## Reproducing the results

To reproduce the results presented [here](https://arxiv.org/abs/2404.09034), you will need a working `Julia` kernel, preferably `v.1.10.x`. You will also need a version of the [`WaterLily.jl`](https://github.com/weymouth/WaterLily.jl) flow solver.

The first step before using these new boundary conditions is to clone this repository somewhere on your machine; I will assume that this repository is to be cloned into a `Workspace` folder on your machine. The first thing to do is to go into this directory
```bash
cd ~/Workspace
```
Then, you can clone the `BiotSavartBCs.jl` (and the `WaterLily.jl` solver if you don't already have it, note that you cannot use the official `WaterLily` release from the `Julia` package manager to run these test due to some function incompatibility)
```bash
git clone https://github.com/weymouth/BiotSavartBCs.jl
(git clone https://github.com/weymouth/WaterLily.jl)
```
The next step is to `dev` the `BiotSavartBCs.jl` into the `julia` environment you are using to run the `WaterLily` simulations.

### Activate the `BiotSavartBCs` environment

I will assume you are using `BiotSavartBCs.jl` with `WaterLily` and that you have an `examples` folder on your machine which contains a `Project.toml` file that sets the environment used to run the simulations. 

Start by opening `julia` and activate this environment
```bash
julia --project=/PATH/TO/BiotSavartBCs.jl/examples
```
This will open julia; use the package manager to `dev` the `BiotSavartBCs.jl`
```julia
julia> ]
(examples) pkg> dev /PATH/TO/BiotSavartBCs.jl
...
(examples) pkg> instantiate
...
```
This will install and precompile some of the packages required to use this new package. If you have elected to use the github version of WaterLily, you can also `dev` it in the same way
```julia
(examples) pkg> dev /PATH/TO/WaterLily.jl
...
(examples) pkg> instantiate
...
```
You can then import the `BiotSavartBCs.jl` package and use it within a `WaterLily` simulation.
```julia
using BiotSavartBCs
```

### Citing

We simply ask you to cite the references below in any publication in which you have made use of the `BiotSavartBCs` project. If you are using other `WaterLily` packages, please cite them as indicated in their repositories.

```bibtex
@article{weymouth2024biot,
    title={Using Biot-Savart to shrink Eulerian domains while maintaining or improving external flow accuracy}, 
    author={Gabriel D. Weymouth and Marin Lauber},
    year={2024},
    eprint={2404.09034},
    archivePrefix={arXiv},
    primaryClass={physics.flu-dyn}
}
```
