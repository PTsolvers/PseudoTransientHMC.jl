using Test
using PseudoTransientHMC

using ReferenceTests, BSON
import ParallelStencil: @reset_parallel_stencil

ENV["USE_GPU"]  = false
ENV["DO_VIZ"]   = false
ENV["DO_SAVE"]  = false
ENV["RUN_TEST"] = true

# Reference test using ReferenceTests.jl
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2) for (v1,v2) in zip(values(d1), values(d2))])
