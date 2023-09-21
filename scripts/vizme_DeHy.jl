using CairoMakie
using MAT
using Printf

@views avx(A) = 0.5 * (A[1:end-1, :] + A[2:end, :])
@views avy(A) = 0.5 * (A[:, 1:end-1] + A[:, 2:end])

fnum = (1, 500, 1000, 2000) # user input - which steps to save

label = ("a)", "b)", "c)", "d)")
fig = Figure(resolution=(1000,400), fontsize=20)
axs = Array{Any}(undef, 4)

for il in eachindex(fnum)
    nsave = @sprintf("%04d", fnum[il])
    vars  = matread("dehy_$(nsave).mat")
    Chi   = get(vars, "Ch_ti_fluid_dif_phi", missing)
    xc    = get(vars, "xc", missing)
    yc    = get(vars, "yc", missing)
    timeP = get(vars, "timeP", missing)
    # Scales
    rho_0  = 3000          # Density scale [kg/m^3]
    Pi_Pa  = 1 / 12.75e8   # Stress scale  [1/Pa]
    t_char = Chi           # Characteristc time
    nx     = length(xc)
    st     = 20
    rng    = 1:st:nx
    # read vars
    Pf    = get(vars, "Pf", missing)
    Phi   = get(vars, "Phi", missing)
    Rho_s = get(vars, "Rho_s", missing)
    Vx    = get(vars, "Vx", missing)
    Vy    = get(vars, "Vy", missing)
    Vx, Vy = avx(Vx), avy(Vy)

    axs[il] = Axis(fig[1,il][1,1]; aspect=DataAspect(), xlabel="x/r", width = 250, height = 250)

    plt = ( p1 = heatmap!(axs[il], xc, yc, Rho_s * rho_0   ; colorrange = (2550, 3150), colormap=Reverse(:lapaz)),
            p2 = contour!(axs[il], xc, yc, Pf / Pi_Pa / 1e8, levels=[12.7, 12.7], color = :black),
            p3 = contour!(axs[il], xc, yc, Phi             , levels=[0.15, 0.15], color = :magenta),
            p4 =  arrows!(axs[il], xc[rng], yc[rng], Vx[rng, rng], Vy[rng,rng]; lengthscale=2.0, arrowsize=9, color=:gray),
    )

    axs[il].title = "$(label[il]) time/tc = $(@sprintf("%1.4f", timeP/t_char))"
    limits!(axs[il], -7, 7, -7, 7)

    (il > 1) && hideydecorations!(axs[il], grid=false)
    axs[1].ylabel = "y/r"

    subgrid = GridLayout(fig[1, 4][1, 2], tellheight = false)
    Label(subgrid[1, 1], "ρₛ [kg⋅m⁻¹]")
    Colorbar(subgrid[2, 1], plt.p1; halign=:left)

    e1, e2 = LineElement(color = :black, linestyle = nothing), LineElement(color = :magenta, linestyle = nothing)
    Legend(fig[1, 1], [e1, e2], ["p_f = 12.7 kbar", "ϕ = 0.15"],
        halign = :right, valign = :top, margin = (7, 7, 7, 7), labelsize = 14)
end

resize_to_layout!(fig)
display(fig)

# save("viz_dehy.png", fig)