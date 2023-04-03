## 2D tests
include("./shared.jl")

include("../scripts_2023/DeHy.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d1  = Dict(:X=> xc[indsx], :Pf=>Pf[indsx,indsy], :Phi=>Phi[indsx,indsy])

@testset "Reference-tests PT HMC Antigorite 2D" begin
    @test_reference "reftest-files/test_DeHy.bson" d2d1 by=comp
end

@reset_parallel_stencil()
