const USE_GPU  = false  # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID   = 0
# using ParallelStencil
# using ParallelStencil.FiniteDifferences2D
# @static if USE_GPU
#     @init_parallel_stencil(CUDA, Float64, 2)
#     CUDA.device!(GPU_ID) # select GPU
# else
#     @init_parallel_stencil(Threads, Float64, 2)
# end
using Plots, Printf, Statistics, LinearAlgebra
using MAT, Interpolations
################################ Macros from cuda_scientific
# import ParallelStencil: INDICES
# ix,  iy  = INDICES[1], INDICES[2]
# ixi, iyi = :($ix+1), :($iy+1)

# macro av_xya(A) esc(:( ($A[$ix, $iy] + $A[($ix+1), $iy] + $A[$ix,($iy+1)] + $A[($ix+1),($iy+1)])*0.25 )) end
# ################################
# @parallel_indices (ix,iy) function initialize!(xc, yc, lx::Data.Number, ly::Data.Number, μsi::Data.Number, ρgi::Data.Number, Mus::Data.Array, Rog::Data.Array)
    
#     if (ix<=size(xc, 1) && iy<=size(yc ,1)) radc = ((xc[ix]-lx*0.5))*((xc[ix]-lx*0.5)) + ((yc[iy]-ly*0.5))*((yc[iy]-ly*0.5)); end
#     if (ix<=size(Mus,1) && iy<=size(Mus,2)) if (radc<1.0)  Mus[ix,iy] = μsi; end; end
#     if (ix<=size(Rog,1) && iy<=size(Rog,2)) if (radc<1.0)  Rog[ix,iy] = ρgi; end; end
    
#     return
# end

# @parallel function timesteps!(Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, max_nxy::Int, dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Mus::Data.Array)

#     @all(dτVx) = Vsc*min_dxy2/@av_xi(Mus)/4.1
#     @all(dτVy) = Vsc*min_dxy2/@av_yi(Mus)/4.1
#     @all(dτPt) = Ptsc*4.1*@all(Mus)/max_nxy

#     return
# end

# @parallel function compute_PT!(divV::Data.Array, Vx::Data.Array, Vy::Data.Array, Pt::Data.Array, dτPt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Mus::Data.Array, _dx::Data.Number, _dy::Data.Number)

#     @all(divV) = _dx*@d_xa(Vx) + _dy*@d_ya(Vy)
#     @all(Pt)   = @all(Pt)  -  @all(dτPt)*@all(divV) 

#     @all(τxx) = 2.0*@all(Mus)*(_dx*@d_xa(Vx)  - 0.333*@all(divV))  
#     @all(τyy) = 2.0*@all(Mus)*(_dy*@d_ya(Vy)  - 0.333*@all(divV))
#     @all(τxy) =  @av_xya(Mus)*(_dy*@d_yi(Vx) + _dx*@d_xi(Vy))  # Free slip BC; 2*0.5 = 1 DEBUG: free slip BC

#     return
# end

# @parallel function compute_dV!(dampX::Data.Number, dampY::Data.Number, Pt::Data.Array, Rog::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, _dx::Data.Number, _dy::Data.Number)

#     @all(Rx)    = _dx*@d_xi(τxx) + _dy*@d_ya(τxy) - _dx*@d_xi(Pt)
#     @all(Ry)    = _dy*@d_yi(τyy) + _dx*@d_xa(τxy) - _dy*@d_yi(Pt) + @av_yi(Rog)
#     @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx) 
#     @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)

#     return
# end

# @parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)

#     @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
#     @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    
#     return
# end

# @parallel_indices (ix,iy) function bc_X!(A::Data.Array)

#     if (ix==1          && iy<=size(A ,2)) A[ix, iy] = A[ix+1, iy] end
#     if (ix==size(A, 1) && iy<=size(A ,2)) A[ix, iy] = A[ix-1, iy] end

#     return
# end

# @parallel_indices (ix,iy) function bc_Y!(A::Data.Array)

#     if (ix<=size(A ,1) && iy==1         ) A[ix, iy] = A[ix, iy+1] end
#     if (ix<=size(A, 1) && iy==size(A ,2)) A[ix, iy] = A[ix, iy-1] end

#     return
# end

##################################################
@views function PT_HMC_v0()

    # read in mat file
    vars = matread("LOOK_UP_HMC_Pub.mat")
    RHO_s_LU = get(vars, "Rho_s_07",1)
    Rho_f_LU = get(vars, "Rho_f",1)
    X_LU     = get(vars, "X_s_vec",1)
    P_LU     = get(vars, "P_vec",1)*1e8

    # Independent parameters
    radius          = 1.0          # Radius of initial P-perturbation [m]
    eta_mat         = 1.0          # Viscosity scale [Pa s]
    P_ini           = 1.0          # Initial ambient pressure [Pa]
    rho_0           = 3000.0       # Density scale [kg/m^3]
    # Nondimensional parameters
    ellipse_factor  = 3.0          # Use 1 for circle
    angle           = 30.0         # Counterclockwise angle of long axis with respect to vertical direction
    phi_ini         = 2e-3         # Initial porosity
    eta_incl_fac    = 1e-3         # Factor, how much solid SHEAR viscosity of inclusion is larger (factor>1) or smaller than surrounding
    lambda_incl_fac = 1.0          # Factor, how much solid BULK viscosity of inclusion is larger (factor>1) or smaller than surrounding
    n_exp           = 3.0          # Stress exponent of matrix; n=1 means linear viscous
    lam_eta         = 1e0          # lam_eta = lambda / eta_mat; []; Ratio of bulk to shear viscosity
    Lc_rad2         = 1e8          # Lc_rad2 = k_etaf*eta_mat/radius^2; []; Ratio of hydraulic fluid extraction to compaction extraction
    Da              = 0.0024       # Da   = eb*eta_mat/P_ini; []; Ratio of viscous stress to initial stress
    sig_yield       = 0.024        # Stress_ref / P_ini; []; Reference stress used for power-law viscous flow law
    Lx_rad          = 10.0         # Model width divided by inclusion radius
    Ly_Lx           = 1.0          # Model height divided by model width
    Pini_Pappl      = P_ini/8.5e8  # Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
    # Dependant parameters
    beta_eff        = 1e-2/P_ini               # Effective compressibility used only to determine PT time step [1/Pa]
    k_etaf          = Lc_rad2*radius^2/eta_mat # Permeability divided by fluid viscosity; [m^2/(Pa s)]
    P_pert          = 0.2*P_ini                # Pressure perturbation [Pa]
    lambda          = lam_eta*eta_mat          # Bulk viscosity [Pa s]
    eb              = Da*P_ini/eta_mat         # Background strain rate in matrix [1/s]
    stress_ref      = sig_yield *P_ini         # Stress reference for power-law viscosity
    Lx              = Lx_rad*radius            # Model width [m]
    Ly              = Ly_Lx*Lx                 # Model height [m]
    P_LU            = P_LU*Pini_Pappl          # Transform look-up table stress to PT stress scale
    # Numerical resolution
    nx              = 101                              # Numerical resolution width
    ny              = nx+1                             # Numerical resolution height
    tol             = 10^(-5)                          # Tolerance for pseudo-transient iterations
    cfl             = 1/16.1                           # CFL parameter for PT-Stokes solution
    dtp             = 2e0*radius^2/(k_etaf/beta_eff)   # Time step physical
    time_tot        = 1.0*dtp                          # Total time of simulation
    # Configuration of grid, matrices and numerical parameters
    dx              = Lx/(nx-1)                     # Grid spacing
    dy              = Ly/(ny-1)                     # Grid spacing
    xc              = -Lx/2:dx:Lx/2                 # Coordinate vector
    yc              = -Ly/2:dy:Ly/2                 # Coordinate vector
    (Xc2, Yc2)      = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    Pf              = P_ini*ones(nx,  ny)           # Initial ambient fluid pressure
    Ptot            = Pf                            # Initial total pressure
    rad_a           = radius
    rad_b           = rad_a*ellipse_factor
    X_ROT           =  Xc2*cosd(angle)+Yc2*sind(angle)
    Y_ROT           = -Xc2*sind(angle)+Yc2*cosd(angle)
    Pf[sqrt.(X_ROT.^2.0./rad_a.^2.0 .+ Y_ROT.^2.0./rad_b.^2) .< 1.0] .= P_ini - P_pert  # Fluid pressure petubation
    X_Ellipse       =  rad_a.*cos.(0:0.01:2*pi).*cosd(angle).+rad_b.*sin.(0:0.01:2*pi).*sind(angle)
    Y_Ellipse       = -rad_a.*cos.(0:0.01:2*pi).*sind(angle).+rad_b.*sin.(0:0.01:2*pi).*cosd(angle)
    ELLIPSE_XY      = [-X_Ellipse; Y_Ellipse]
    for smo=1:3 # Smooting of perturbation
        Pf[2:end-1,:] .= Pf[2:end-1,:] .+ 0.4.*(Pf[3:end,:].-2.0.*Pf[2:end-1,:].+Pf[1:end-2,:])
        Pf[:,2:end-1] .= Pf[:,2:end-1] .+ 0.4.*(Pf[:,3:end].-2.0.*Pf[:,2:end-1].+Pf[:,1:end-2])
    end
    Pf_inip         = Pf
    # Density, compressibility and gamma from concentration and pressure from thermodynamic data base
    itp1 = interpolate( (P_LU[:,1],), RHO_s_LU[:,1], Gridded(Linear()))
    itp2 = interpolate( (P_LU[:,1],), Rho_f_LU[:,1], Gridded(Linear()))
    itp3 = interpolate( (P_LU[:,1],),     X_LU[:,1], Gridded(Linear()))
    RHO_S     = itp1.(Pf)./rho_0
    RHO_F     = itp2.(Pf)./rho_0
    X_S       = itp3.(Pf)
    RHO_S_ini = itp1.(P_ini*ones(nx, ny))./rho_0
    X_S_ini   = itp3.(P_ini*ones(nx, ny))
    # Initialize required matrices (or arrays)
    PHI_INI         = phi_ini*ones(nx, ny)
    PHI             = PHI_INI
    K_DARCY         = k_etaf*ones(nx, ny)
    DIVV            = zeros(nx, ny)
    DIVV_RHO_X      = zeros(nx, ny)
    DIVV_RHO_T      = zeros(nx, ny)
    TAUXX           = zeros(nx, ny)               # Deviatoric stress
    TAUYY           = zeros(nx, ny)               # Deviatoric stress
    TAUXY           = zeros(nx-1,ny-1)            # Deviatoric stress
    # Centered coordinates needed for staggered grid for solid velocities
    xv              = [xc[1]-dx/2, (xc[1:end-1]+xc[2:end])/2.0, xc[end]+dx/2]  # Horizontal vector for Vx which is one more than basic grid
    yv              = [yc[1]-dy/2, (yc[1:end-1]+yc[2:end])/2.0, yc[end]+dy/2]  # Verical    vector for Vy which is one more than basic grid
    (Xc2vx, Yc2vx)  = ([x for x=xv, y=yc], [y for x=xv, y=yc])
    (Xc2vy, Yc2vy)  = ([x for x=xc, y=yv], [y for x=xc, y=yv])
    VX              = -eb*Xc2vx                   # Pure shear, shortening in x
    VY              =  eb*Yc2vy                   # Pure shear, extension in y
    # Parameters for time loop and pseudo-transient iterations
    time            = 0.0                         # Initial time
    it              = 0                           # Integer count for iteration loop
    itp             = 0                           # Integer count for time loop
    save_count      = 0 
    Time_vec        = [0]



    # # Physics
    # lx, ly    = 10.0, 10.0
    # μs0       = 1.0
    # μsi       = 0.1
    # ρgi       = 1.0
    # # Numerics
    # nitr      = 10000
    # nout      = 200
    # BLOCK_X   = 16
    # BLOCK_Y   = 16 
    # GRID_X    = 8
    # GRID_Y    = 8
    # Vdmp      = 4.0
    # Vsc       = 1.0
    # Ptsc      = 1.0/4.0
    # ε         = 1e-5
    # nx        = GRID_X*BLOCK_X - 1 # -1 due to overlength of array nx+1
    # ny        = GRID_Y*BLOCK_Y - 1 # -1 due to overlength of array ny+1
    # cuthreads = (BLOCK_X, BLOCK_Y, 1)
    # cublocks  = (GRID_X , GRID_Y , 1)
    # dx, dy    = lx/nx, ly/ny
    # _dx, _dy  = 1.0/dx, 1.0/dy 
    # # Initialisation
    # Pt        = @zeros(nx  ,ny  )
    # dτPt      = @zeros(nx  ,ny  )
    # divV      = @zeros(nx  ,ny  )
    # Vx        = @zeros(nx+1,ny  )
    # Vy        = @zeros(nx  ,ny+1)
    # τxx       = @zeros(nx  ,ny  )
    # τyy       = @zeros(nx  ,ny  )
    # τxy       = @zeros(nx-1,ny-1)
    # Rx        = @zeros(nx-1,ny-2)
    # Ry        = @zeros(nx-2,ny-1)
    # dVxdτ     = @zeros(nx-1,ny-2)
    # dVydτ     = @zeros(nx-2,ny-1)
    # dτVx      = @zeros(nx-1,ny-2)
    # dτVy      = @zeros(nx-2,ny-1)
    # Rog       = @zeros(nx  ,ny  )
    # Mus       = μs0*@ones(nx,ny)
    # xc        = LinRange(dx/2, lx-dx/2, nx)
    # yc        = LinRange(dy/2, ly-dy/2, ny)
    # min_dxy2  = min(dx,dy)^2
    # max_nxy   = max(nx,ny)
    # dampX     = 1.0-Vdmp/nx
    # dampY     = 1.0-Vdmp/ny
    # # visualisation
    # gr() #pyplot()
    # ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    # println("Animation directory: $(anim.dir)")
    # # Action
    # @parallel cublocks cuthreads initialize!(xc, yc, lx, ly, μsi, ρgi, Mus, Rog)

    # @parallel cublocks cuthreads timesteps!(Vsc, Ptsc, min_dxy2, max_nxy, dτVx, dτVy, dτPt, Mus)

    # err=2*ε; err_evo1=[]; err_evo2=[]; warmup=10
    # for itr = 1:nitr
    #     if (itr==warmup+1) global t0 = Base.time(); end

    #     @parallel cublocks cuthreads compute_PT!(divV, Vx, Vy, Pt, dτPt, τxx, τyy, τxy, Mus, _dx, _dy)

    #     @parallel cublocks cuthreads compute_dV!(dampX, dampY, Pt, Rog, τxx, τyy, τxy, Rx, Ry, dVxdτ, dVydτ, _dx, _dy)

    #     @parallel cublocks cuthreads compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

    #     @parallel cublocks cuthreads bc_Y!(Vx)
    #     @parallel cublocks cuthreads bc_X!(Vy)

    #     # convergence check
    #     if mod(itr, nout)==0
    #         global mean_Rx, mean_Ry, mean_divV
    #         mean_Rx = mean(abs.(Rx[:])); mean_Ry = mean(abs.(Ry[:])); mean_divV = mean(abs.(divV[:]))
    #         err = maximum([mean_Rx, mean_Ry, mean_divV])
    #         push!(err_evo1, maximum([mean_Rx, mean_Ry, mean_divV])); push!(err_evo2,itr)
    #         @printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_divV=%1.3e] \n", itc, err, mean_Rx, mean_Ry, mean_divV)
    #     end
    #     if (err<=ε) break; end
    #     global itc=itr
    # end
    # global time_s = Base.time() - t0
    # A_eff = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    # t_it  = time_s/(itc-warmup)                  # Execution time per iteration [s]
    # T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]
    # @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ %1.2f GB/s) \n",itc,err,time_s,T_eff)
    # # Plotting
    # yv = LinRange(0, ly, ny+1)
    # p1 = heatmap(xc,yc,Pt', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:inferno, title="Pressure")
    # p2 = heatmap(xc,yv,Vy', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yv[1], yv[end]), c=:inferno, title="Vy")
    # p4 = heatmap(xc[2:end-1],yv[2:end-1],log10.(abs.(Ry')), aspect_ratio=1, xlims=(xc[2], xc[end-1]), ylims=(yc[2], yc[end-1]), c=:inferno, title="log10(Ry)");
    # p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    # # display(plot( p1, p2, p4, p5 ))
    # plot( p1, p2, p4, p5 ); frame(anim)
    # gif(anim, "Stokes2D.gif", fps = 15)

    return
end

@time PT_HMC_v0()
