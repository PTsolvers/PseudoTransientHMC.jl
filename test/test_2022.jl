## 2D tests
include("./shared.jl")

ENV["NX"] = 255
ENV["NY"] = 255

include("../scripts_2022/PT_HMC_atg.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d1  = Dict(:X=> xc[indsx], :Pf=>Pf[indsx,indsy], :Phi=>Phi[indsx,indsy])

include("../scripts_2022/PT_HMC_atg_rand.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d2  = Dict(:X=> xc[indsx], :Pf=>Pf[indsx,indsy], :Phi=>Phi[indsx,indsy])

@testset "Reference-tests PT HMC Antigorite 2D" begin
    @test_reference "reftest-files/test_PT_HMC_atg.bson" d2d1 by=comp
    @test_reference "reftest-files/test_PT_HMC_atg_rand.bson" d2d2 by=comp
end

@reset_parallel_stencil()
