## 2D tests
include("./shared.jl")

ENV["NX"] = 63
ENV["NY"] = 63

include("../scripts_2020/PT_HMC_bru.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d1  = Dict(:X=> xc[indsx], :Pf=>Pf[indsx,indsy], :Phi=>Phi[indsx,indsy])

include("../scripts_2020/PT_HMC_bru_analytical.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d2  = Dict(:X=> xc[indsx], :Pf=>Pf[indsx,indsy], :Phi=>Phi[indsx,indsy])

@testset "Reference-tests PT HMC Brucite 2D" begin
    @test_reference "reftest-files/test_PT_HMC_bru.bson" d2d1 by=comp
    @test_reference "reftest-files/test_PT_HMC_bru_analytical.bson" d2d2 by=comp
end

@reset_parallel_stencil()
