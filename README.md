# Pseudo-transient Hydro-Mechanical-Chemical

[![Build Status](https://github.com/PTsolvers/PseudoTransientHMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PTsolvers/PseudoTransientHMC.jl/actions/workflows/CI.yml?query=branch%3Amain)

This repository contains Pseudo-Transient (PT) routines resolving chemical reactions coupled to fluid flow in viscously defroming solid pourous matrix, so-called Hydro-Mechanical-Chemical (HMC) coupling. Example of such multi-physical processes relate to, e.g, the brucite-periclase reactions [(Schmalholz et al., 2020)](https://doi.org/10.1029/2020GC009351) and could explain the formation of olivine veins by dehydration of ductile serpentinite [(Schmalholz et al., 2022 - submitted)]().

Pseudo-Transient approach relies in using physics-inspired transient terms within differential equations in order to iteratively converge to an accurate solution. The PT HMC routines are written using the [Julia programming language](https://julialang.org) and build upon the high-performance [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) package to enable for optimal execution on graphical processing units (GPUs) and multi-threaded CPUs.

## Content
* [Script list](#script-list)
* [Usage](#usage)
* [Output](#output)
* [References](#references)

## Script list
The scripts are located in two distinct folders. The folders contain the Julia (and early Matlab) routines and the `.mat` file with the corresponding thermodynamic data to be loaded as look-up tables.

- The [scripts_2020](scripts_2020) folder relates to the [(Schmalholz et al., 2020)](https://doi.org/10.1029/2020GC009351) study. The "analytical" [`PT_HMC_bru_analytical.jl`](scripts_2020/PT_HMC_bru_analytical.jl) script includes a paramtetrisation of the solid and fluid densities and composition as function of fluid pressure to circumvent costly interpolation operations.

- The [scripts_2022](scripts_2022) folder contains the routines for the [(Schmalholz et al., 2022)]() study. The main scipt is [`PT_HMC_atg.jl`](scripts_2022/PT_HMC_atg.jl). _The "rand" version implements resolving HMC coupling given a random initial porosity distribution._

## Usage
If not stated otherwise, all the routines are written in Julia and can be executed from the [Julia REPL], or from the terminal for improved performance. Output is produced using the [Julia Plots package].

The either multi-threaded CPU or GPU backend can be selected by adding the appropriate flag to the `USE_GPU` constant, modifying either the default behaviour in the top-most lines of the codes
```julia
const USE_GPU  = haskey(ENV, "USE_GPU" ) ? parse(Bool, ENV["USE_GPU"] ) : false
```
or by setting/exporting the desired environment variable.

- Selecting `false` will use the `Base.threads` backend. Multi-threading can be enabled by defining and exporting the `JULIA_NUM_THREADS` environment variable (e.g. `export JULIA_NUM_THREADS=2` prior to launching Julia will enable the code to run on 2 CPU threads).
- Selecting `true` will use the [CUDA.jl] GPU backend and will succeed if a CUDA-capable GPU is available.

### Example running the routine from the REPL

1. Launch Julia
```sh
% julia --project
```
2. Activate and instantiate the environment to download all required dependencies:
```julia-repl
julia> ]

(PseudoTransientHMC) pkg> activate .

(PseudoTransientHMC) pkg> add https://github.com/luraess/ParallelRandomFields.jl

(PseudoTransientHMC) pkg> instantiate
```
3. Run the script
```julia-repl
julia> include("PT_HMC_atg.jl")
```

### Example running the routine from the terminal

1. Launch the Julia executable using the project's dependencies `--project`, disabling array bound checking for enhanced performance `--check-bounds=no`, and using optimization level 3 `-O3`.
```sh
julia --project --check-bounds=no -O3 PT_HMC_atg.jl
```
Additional startup flag infos can be found [here](https://docs.julialang.org/en/v1/manual/getting-started/#man-getting-started)

## Output
The output of running the [`PT_HMC_bru_analytical.jl`](scripts_2020/PT_HMC_bru_analytical.jl) script on an Nvidia TitanXp GPU with `nx=1023, ny=1023`:

![PT-HMC code predicting brucite-periclase reaction](docs/PT_HMC_1023x1023.png)

The output of running the [`PT_HMC_atg.jl`](scripts_2022/PT_HMC_atg.jl) script on an Nvidia Tesla V100 GPU with `nx=1023, ny=1023`:

![](docs/PT_HMC_Atg_1023x1023.gif)


## References
[Schmalholz, S. M., Moulas, E., Plümper, O., Myasnikov, A. V., & Podladchikov, Y. Y. (2020). 2D hydro‐mechanical‐chemical modeling of (De)hydration reactions in deforming heterogeneous rock: The periclase‐brucite model reaction. Geochemistry, Geophysics, Geosystems, 21, 2020GC009351. https://doi.org/10.1029/2020GC009351](https://doi.org/10.1029/2020GC009351)

[Schmalholz, S. M., Moulas, E., Räss, L., & Müntener, O. (2022). Shear-driven formation of olivine veins by dehydration of ductile serpentinite: a numerical study. Submitted to XYZ]()

[CUDA.jl]: https://github.com/JuliaGPU/CUDA.jl
[Julia Plots package]: https://github.com/JuliaPlots/Plots.jl
[Julia REPL]: https://docs.julialang.org/en/v1/stdlib/REPL/
